"""
Usage:
    # Option A: Precompute first, then run parallel training jobs
    python defense_memory_efficient.py --precompute-only \\
        --dataset arxiv --ref-model pythia-2.8b --finetune-ref --gpu 0
    
    # Then launch multiple jobs with same model but different alphas
    # (can run in parallel on different GPUs)
    python defense_memory_efficient.py --model pythia-2.8b --dataset arxiv \\
        --ref-model pythia-2.8b --finetune-ref --alpha 0.2 --gpu 0 &
    python defense_memory_efficient.py --model pythia-2.8b --dataset arxiv \\
        --ref-model pythia-2.8b --finetune-ref --alpha 0.5 --gpu 1 &
    python defense_memory_efficient.py --model pythia-2.8b --dataset arxiv \\
        --ref-model pythia-2.8b --finetune-ref --alpha 0.8 --gpu 2 &

    # Option B: Single run (auto-caches if not already cached)
    python defense_memory_efficient.py --model pythia-2.8b --dataset arxiv \\
        --ref-model pythia-2.8b --finetune-ref --alpha 0.8 --gpu 0

Cache location: $REF_LOSS_CACHE_DIR/{dataset}/{ref_model}_{type}_maxlen{N}_losses.pt
                (default: $HF_HOME/ref_loss_cache/...)
"""

import os
import json
import argparse
from datetime import datetime
from typing import Optional, Dict, List
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_wandb(project: str, run_name: str, group: str, config: dict) -> bool:
    """Login (if needed) and start a wandb run. Returns False to disable logging.

    Key resolution order: $WANDB_API_KEY -> ~/.wandb_key -> fail-soft (return False).
    Calling wandb.init with reinit=True closes any prior run cleanly so this is
    safe to call once per Trainer phase (ref FT, then target FT).
    """
    if not WANDB_AVAILABLE:
        print("[wandb] package not installed; skipping logging")
        return False
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        keyfile = os.path.expanduser("~/.wandb_key")
        if os.path.exists(keyfile):
            with open(keyfile) as f:
                key = f.read().strip()
            os.environ["WANDB_API_KEY"] = key
    if not key:
        print("[wandb] no API key (set WANDB_API_KEY or write ~/.wandb_key); skipping")
        return False
    try:
        wandb.login(key=key, relogin=False)
        wandb.init(project=project, name=run_name, group=group,
                   config=config, reinit=True)
        return True
    except Exception as e:
        print(f"[wandb] init failed: {e}; continuing without logging")
        return False


# ============================================================================
# Configuration
# ============================================================================

import os as _os
HF_DIR = _os.environ.get("HF_HOME", "./hf_cache")
DATASET_DIR = _os.environ.get("DATASET_DIR", _os.path.join(HF_DIR, "datasets", "parameterlab"))
OUTPUT_DIR = _os.environ.get("DEFENDED_MODELS_ROOT", "./defended_models")
CACHE_DIR = _os.environ.get("REF_LOSS_CACHE_DIR", _os.path.join(HF_DIR, "ref_loss_cache"))
MODEL_MAX_LENGTH = 2048

DATASETS = [
    "Pile-CC", "arxiv", "FreeLaw", "Github", "HackerNews",
    "OpenWebText2", "USPTO_Backgrounds", "wiki"
]


# ============================================================================
# Data Loading
# ============================================================================

def load_defense_data(dataset: str):
    """Load parameterlab Pile data for continual fine-tuning defense evaluation."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {DATASETS}")

    data_dir = f"{DATASET_DIR}/scaling_mia_the_pile_00_{dataset}/data"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset not found at {data_dir}")

    ds = load_dataset("parquet", data_dir=data_dir,
                      cache_dir=os.path.join(HF_DIR, "datasets", "cache"))

    train_ds = ds["validation"]
    eval_ds = ds["test"]

    train_ds = train_ds.filter(lambda x: len(x["text"]) > 100)
    eval_ds = eval_ds.filter(lambda x: len(x["text"]) > 100)

    print(f"Defense data ({dataset}):")
    print(f"  Fine-tuning (validation/members): {len(train_ds)}")
    print(f"  Held-out (test/non-members):       {len(eval_ds)}")
    return train_ds, eval_ds


def tokenize_and_chunk(dataset, tokenizer, max_length=MODEL_MAX_LENGTH):
    """Tokenize and chunk documents into fixed-length segments."""
    def chunk_fn(examples):
        all_input_ids = []
        all_attention_mask = []
        for text in examples["text"]:
            ids = tokenizer(text, truncation=False, padding=False)["input_ids"]
            for i in range(0, len(ids), max_length):
                chunk = ids[i:i + max_length]
                all_input_ids.append(chunk)
                all_attention_mask.append([1] * len(chunk))
        return {"input_ids": all_input_ids, "attention_mask": all_attention_mask}

    return dataset.map(
        chunk_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )


# ============================================================================
# Model Loading
# ============================================================================

def load_pythia_model(model_name: str, device, dtype=torch.float16):
    """Load a Pythia model and tokenizer from local HF cache."""
    path = f"{HF_DIR}/models/EleutherAI__{model_name}"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")

    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        return_dict=True,
        torch_dtype=dtype,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


# ============================================================================
# Reference Loss Caching
# ============================================================================

class CachedRefLossDataset(Dataset):
    """Dataset wrapper that includes cached reference losses."""

    def __init__(self, tokenized_dataset, ref_losses_path: str):
        self.dataset = tokenized_dataset
        # weights_only=False is safe here since we control the cache file
        self.ref_losses = torch.load(ref_losses_path, weights_only=False)
        assert len(self.ref_losses) == len(self.dataset), \
            f"Mismatch: {len(self.ref_losses)} cached losses vs {len(self.dataset)} samples"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Create a copy to avoid mutating cached dataset items
        result = {
            "input_ids": list(item["input_ids"]) if not isinstance(item["input_ids"], list) else item["input_ids"],
            "attention_mask": list(item["attention_mask"]) if not isinstance(item["attention_mask"], list) else item["attention_mask"],
        }
        # Add cached reference losses (convert to list for consistent handling)
        ref_losses = self.ref_losses[idx]
        result["ref_token_losses"] = ref_losses.tolist() if isinstance(ref_losses, torch.Tensor) else ref_losses
        return result


class CachedRefLossCollator:
    """Data collator that handles cached reference losses."""

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Find max length in batch
        max_len = max(len(f["input_ids"]) for f in features)
        max_len = min(max_len, self.max_length)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_ref_losses = []

        for f in features:
            # Ensure we're working with lists
            input_ids = list(f["input_ids"]) if not isinstance(f["input_ids"], list) else f["input_ids"]
            attention_mask = list(f["attention_mask"]) if not isinstance(f["attention_mask"], list) else f["attention_mask"]
            ref_losses = list(f["ref_token_losses"]) if not isinstance(f["ref_token_losses"], list) else f["ref_token_losses"]

            # Truncate to max_len
            input_ids = input_ids[:max_len]
            attention_mask = attention_mask[:max_len]
            # ref_losses is for shifted sequence (seq_len - 1), truncate accordingly
            ref_losses = ref_losses[:max_len - 1]

            # Pad input_ids and attention_mask to max_len
            pad_len = max_len - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

            # Create labels (same as input_ids, but mask padding with -100)
            labels = [tok if mask == 1 else -100 for tok, mask in zip(input_ids, attention_mask)]

            # Pad ref_losses to (max_len - 1)
            ref_pad_len = (max_len - 1) - len(ref_losses)
            ref_losses = ref_losses + [0.0] * ref_pad_len

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            batch_ref_losses.append(ref_losses)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "ref_token_losses": torch.tensor(batch_ref_losses, dtype=torch.float32),
        }


def precompute_reference_losses(
    ref_model,
    tokenizer,
    train_dataset,
    output_path: str,
    max_length: int = MODEL_MAX_LENGTH,
    batch_size: int = 32,
    device: str = "cuda",
):
    """Precompute per-token losses from frozen reference model and save to disk.

    Args:
        ref_model: Frozen reference model
        tokenizer: Tokenizer
        train_dataset: Raw training dataset (will be tokenized)
        output_path: Path to save cached losses
        max_length: Max sequence length
        batch_size: Batch size for inference
        device: Device to run on

    Saves:
        List of tensors, one per sample, each of shape (seq_len - 1,)
        representing per-token cross-entropy losses (shifted for next-token prediction)
    """
    print(f"\n{'='*60}")
    print("Precomputing Reference Model Losses")
    print(f"{'='*60}")
    print(f"Samples: {len(train_dataset)}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Tokenize and chunk
    train_tok = tokenize_and_chunk(train_dataset, tokenizer, max_length)
    print(f"Total chunks after tokenization: {len(train_tok)}")

    ref_model.eval()
    ref_model.to(device)

    all_ref_losses = []

    # Simple collator for inference (no labels needed during caching)
    def simple_collate(features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids = []
        batch_attention_mask = []
        orig_lengths = []

        for f in features:
            # Ensure we're working with lists
            input_ids = list(f["input_ids"]) if not isinstance(f["input_ids"], list) else f["input_ids"]
            attention_mask = list(f["attention_mask"]) if not isinstance(f["attention_mask"], list) else f["attention_mask"]
            
            input_ids = input_ids[:max_len]
            attention_mask = attention_mask[:max_len]
            orig_len = len(input_ids)
            orig_lengths.append(orig_len)

            pad_len = max_len - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "orig_lengths": orig_lengths,
        }

    loader = DataLoader(
        train_tok,
        batch_size=batch_size,
        shuffle=False,  # IMPORTANT: maintain order
        collate_fn=simple_collate,
    )

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing reference losses"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            orig_lengths = batch["orig_lengths"]

            # Forward pass
            outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Compute per-token CE loss (shifted)
            # logits: (batch, seq, vocab) -> (batch, seq-1, vocab)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            # Per-token loss
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            # (batch, seq-1)
            token_losses = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(shift_logits.size(0), -1)

            # Extract per-sample, removing padding
            for i, orig_len in enumerate(orig_lengths):
                # Only keep losses for actual tokens (not padding)
                # orig_len tokens -> orig_len - 1 losses (shifted)
                sample_losses = token_losses[i, :orig_len - 1].cpu()
                all_ref_losses.append(sample_losses)

    # Save as list of tensors (variable length)
    torch.save(all_ref_losses, output_path)
    print(f"\nSaved {len(all_ref_losses)} cached reference losses to {output_path}")

    return output_path


# ============================================================================
# Memory-Efficient DuoLearn Trainer
# ============================================================================

class CachedDuoLearnTrainer(Trainer):
    """DuoLearn trainer using cached reference losses (no ref model in memory)."""

    def __init__(self, *args, alpha: float, top_k: float, bottom_k: float, 
                 eval_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.top_k = top_k
        self.bottom_k = bottom_k
        # Store separate collator for evaluation (doesn't need ref_token_losses)
        self.eval_collator = eval_collator

    def get_eval_dataloader(self, eval_dataset=None):
        """Override to use standard collator for evaluation (no ref losses needed)."""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        # Temporarily swap collator for evaluation
        original_collator = self.data_collator
        if self.eval_collator is not None:
            self.data_collator = self.eval_collator

        eval_dataloader = super().get_eval_dataloader(eval_dataset)

        # Restore original collator
        self.data_collator = original_collator

        return eval_dataloader

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """DuoLearn dual-purpose loss with cached reference per-token losses.

        At training time the per-token cross-entropy is split into a "hard"
        bucket (top-k by `target_loss - ref_loss`, learned normally) and a
        "memorised" bucket (bottom-k, unlearned with weight `-self.alpha`):
        ``L_dual = mean(CE_hard) - alpha * mean(CE_mem)``. At eval time the
        cached ref losses are dropped and standard CE is returned.

        Args:
            model: The target HF causal-LM being fine-tuned.
            inputs: HF batch dict; in train mode must include
                ``ref_token_losses`` (per-token CE under the reference model,
                aligned with ``inputs["labels"]``).
            num_items_in_batch: Accepted for HF Trainer signature compatibility;
                unused.
            return_outputs: If True, return ``(loss, model_outputs)``;
                otherwise just the scalar loss tensor.
        """
        # During evaluation, use standard loss (no ref_token_losses in eval data)
        if not model.training:
            # Remove ref_token_losses if present (shouldn't be, but safety check)
            inputs.pop("ref_token_losses", None)
            outputs = model(**inputs)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        # Extract cached reference losses
        ref_token_losses = inputs.pop("ref_token_losses").to(model.device)

        # Forward pass on target model
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Mask out padding tokens
        mask = shift_labels != -100

        # Get valid tokens
        valid_logits = shift_logits[mask]
        valid_labels = shift_labels[mask]
        valid_ref_losses = ref_token_losses[mask].float()  # Ensure float32
        num_tokens = valid_labels.numel()

        if num_tokens == 0:
            zero = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return (zero, outputs) if return_outputs else zero

        # Compute target model per-token loss
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        token_loss = loss_fn(valid_logits.float(), valid_labels)  # Ensure float32 for stability

        # Calibrated score: s(ti) = target_loss - ref_loss
        # High score = hard token (model struggles more than ref)
        # Low score = memorized token (model predicts better than ref)
        calibrated_score = token_loss - valid_ref_losses

        # Select hard tokens (top-k by calibrated score) for learning
        num_hard = max(1, int(self.top_k * num_tokens))
        _, hard_idx = torch.topk(calibrated_score, num_hard)
        hard_loss = token_loss[hard_idx].mean()

        # Select memorized tokens (bottom-k by calibrated score) for unlearning
        num_mem = max(1, int(self.bottom_k * num_tokens))
        _, mem_idx = torch.topk(calibrated_score, num_mem, largest=False)
        mem_loss = token_loss[mem_idx].mean()

        # Dual-purpose loss: learn hard tokens, unlearn memorized tokens
        # L_dual = L_CE(hard) - α * L_CE(memorized)
        dual_loss = hard_loss - self.alpha * mem_loss

        return (dual_loss, outputs) if return_outputs else dual_loss


# ============================================================================
# Memory-Efficient DuoLearn Defense
# ============================================================================

class DuoLearnMemoryEfficient:
    """Memory-efficient DuoLearn that precomputes reference losses.

    Two-phase approach:
    1. Load ref model -> compute losses -> save to disk -> unload ref model
    2. Load target model -> train using cached losses

    This avoids having both models in GPU memory simultaneously.
    """

    def __init__(
        self,
        model_name: str,
        device,
        alpha: float = 0.8,
        top_k: float = 0.6,
        bottom_k: float = 0.2,
        ref_model_name: str = "pythia-2.8b",
        # Bumped above the original 1.75e-6 Pythia recipe so the target
        # actually memorizes members enough for the DuoLearn signal to bite.
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        grad_accumulation_steps: int = 3,
        num_epochs: int = 10,
        max_length: int = MODEL_MAX_LENGTH,
        finetune_ref: bool = False,
        dataset_name: str = "",
        use_wandb: bool = True,
        wandb_project: str = "duolearn-defense",
    ):
        self.model_name = model_name
        self.device = device
        self.alpha = alpha
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.ref_model_name = ref_model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.grad_accumulation_steps = grad_accumulation_steps
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.finetune_ref = finetune_ref
        self.dataset_name = dataset_name
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

        # Will be loaded later
        self.model = None
        self.tokenizer = None

    def _get_cache_path(self) -> str:
        """Get path for cached reference losses.
        
        Cache key includes: dataset, ref_model, finetune_ref, max_length
        
        IMPORTANT: The reference model must match the target model architecture
        (per DuoLearn paper: "Reference model shares an identical architecture 
        with the training model"). The calibrated score s(ti) = L_ref - L_target
        only makes sense when comparing same-architecture models.
        
        You can reuse the same cache for different:
        - alpha values
        - learning rates, batch sizes, epochs
        - top_k, bottom_k fractions
        
        You CANNOT reuse the cache for:
        - Different target model architectures (must match ref_model)
        - Different datasets
        - Different max_length values
        """
        ref_type = "finetuned" if self.finetune_ref else "pretrained"
        return (f"{CACHE_DIR}/{self.dataset_name}/"
                f"{self.ref_model_name}_{ref_type}_maxlen{self.max_length}_losses.pt")

    def _finetune_reference(self, device) -> str:
        """Fine-tune reference model on the dataset's train split and return its save path."""
        # Short-circuit if the per-token loss cache already exists: those losses
        # are the only thing the downstream target-FT consumes, so re-fine-tuning
        # the ref model produces no usable artifact and just wastes ~45 min/dataset.
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            print(f"\nLoss cache exists at {cache_path} -- "
                  f"skipping ref fine-tune (no model weights needed downstream).")
            return ""  # _compute_and_cache_ref_losses will hit the same cache next.

        ref_save_dir = f"{OUTPUT_DIR}/duolearn/ref_models/{self.ref_model_name}/{self.dataset_name}"
        best_path = f"{ref_save_dir}/best"

        if os.path.exists(best_path):
            print(f"\nFound existing fine-tuned reference at {best_path}")
            return best_path

        print(f"\n{'='*60}")
        print(f"Fine-tuning reference model: {self.ref_model_name}")
        print(f"{'='*60}")

        # Load train split as T_aux
        data_dir = f"{DATASET_DIR}/scaling_mia_the_pile_00_{self.dataset_name}/data"
        ds = load_dataset("parquet", data_dir=data_dir,
                          cache_dir=os.path.join(HF_DIR, "datasets", "cache"))
        train_aux = ds["train"].filter(lambda x: len(x["text"]) > 100)

        # Subsample: 1000 docs (chunked to 2048-token paragraphs below).
        if len(train_aux) > 1000:
            train_aux = train_aux.shuffle(seed=42).select(range(1000))

        val_aux = ds["validation"].filter(lambda x: len(x["text"]) > 100)
        if len(val_aux) > 200:
            val_aux = val_aux.shuffle(seed=42).select(range(200))

        print(f"T_aux: {len(train_aux)} samples, Val: {len(val_aux)} samples")

        # Load reference model
        ref_model, ref_tokenizer = load_pythia_model(
            self.ref_model_name, device, dtype=torch.bfloat16
        )

        # Adjust batch size for model size
        model_size = sum(p.numel() for p in ref_model.parameters())
        if model_size > 5e9:  # 6.9B+
            ref_batch_size, ref_grad_accum = 24, 4
            print(f"Large model -- -- batch={ref_batch_size}, grad_accum={ref_grad_accum}, gradient checkpointing OFF")
        elif model_size > 2e9:  # 2.8B
            ref_batch_size, ref_grad_accum = 48, 2
            ref_model.gradient_checkpointing_enable()
            print(f"Medium model -- batch={ref_batch_size}, grad_accum={ref_grad_accum}, gradient checkpointing ON")
        else:
            ref_batch_size, ref_grad_accum = 32, 3
            print(f"Small model -- batch=32, grad_accum=3")

        # Match the target's context: chunk full docs to self.max_length so the
        # ref fine-tune covers every position the target will train on (avoids
        # the asymmetry where ref losses for positions >256 came from a model
        # whose own fine-tune never saw beyond 256).
        ref_max_length = self.max_length
        ref_lr = 1.75e-6
        ref_epochs = 10

        train_tok = tokenize_and_chunk(train_aux, ref_tokenizer, ref_max_length)
        val_tok = tokenize_and_chunk(val_aux, ref_tokenizer, ref_max_length)
        print(f"Ref fine-tune chunks: train={len(train_tok)}, val={len(val_tok)} "
              f"(max_length={ref_max_length})")

        data_collator = DataCollatorForLanguageModeling(tokenizer=ref_tokenizer, mlm=False)

        ref_run_name = f"refFT_{self.dataset_name}_{self.ref_model_name}"
        wb_active = self.use_wandb and setup_wandb(
            project=self.wandb_project,
            run_name=ref_run_name,
            group=f"{self.dataset_name}_{self.ref_model_name}",
            config={
                "phase": "ref_finetune",
                "dataset": self.dataset_name,
                "ref_model": self.ref_model_name,
                "lr": ref_lr,
                "epochs": ref_epochs,
                "batch_size": ref_batch_size,
                "grad_accum": ref_grad_accum,
                "max_length": ref_max_length,
            },
        )

        training_args = TrainingArguments(
            output_dir=ref_save_dir,
            num_train_epochs=ref_epochs,
            per_device_train_batch_size=ref_batch_size,
            gradient_accumulation_steps=ref_grad_accum,
            per_device_eval_batch_size=ref_batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            save_only_model=True,
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,
            logging_steps=10,
            learning_rate=ref_lr,
            report_to="wandb" if wb_active else "none",
            run_name=ref_run_name,
            bf16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=ref_model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model(best_path)
        ref_tokenizer.save_pretrained(best_path)

        # Close the ref-FT wandb run cleanly so the upcoming target-FT run is separate.
        if wb_active and WANDB_AVAILABLE:
            wandb.finish()

        # Cleanup
        del ref_model, trainer
        torch.cuda.empty_cache()

        print(f"\nReference model saved to {best_path}")
        return best_path

    def _compute_and_cache_ref_losses(self, train_dataset, ref_model_path: Optional[str] = None):
        """Phase 1: Compute reference losses and cache to disk."""
        cache_path = self._get_cache_path()

        if os.path.exists(cache_path):
            print(f"\nFound cached reference losses at {cache_path}")
            return cache_path

        print(f"\n{'='*60}")
        print("Phase 1: Computing Reference Losses")
        print(f"{'='*60}")

        # Load reference model
        if ref_model_path and os.path.exists(ref_model_path):
            print(f"Loading fine-tuned reference from {ref_model_path}")
            ref_model = AutoModelForCausalLM.from_pretrained(
                ref_model_path, local_files_only=True, torch_dtype=torch.float16
            ).to(self.device)
            ref_tokenizer = AutoTokenizer.from_pretrained(ref_model_path, local_files_only=True)
        else:
            print(f"Loading pretrained reference: {self.ref_model_name}")
            ref_model, ref_tokenizer = load_pythia_model(
                self.ref_model_name, self.device, dtype=torch.float16
            )

        if ref_tokenizer.pad_token is None:
            ref_tokenizer.pad_token = ref_tokenizer.eos_token
        ref_tokenizer.pad_token_id = ref_tokenizer.eos_token_id

        # Store tokenizer for later use
        self.tokenizer = ref_tokenizer

        # Compute and cache losses
        precompute_reference_losses(
            ref_model=ref_model,
            tokenizer=ref_tokenizer,
            train_dataset=train_dataset,
            output_path=cache_path,
            max_length=self.max_length,
            batch_size=self.batch_size,
            device=self.device,
        )

        # Cleanup reference model
        del ref_model
        torch.cuda.empty_cache()
        print("Reference model unloaded from GPU")

        return cache_path

    def train(self, train_dataset, val_dataset, output_dir: str):
        """Run memory-efficient DuoLearn training.

        Phase 1: Fine-tune ref (optional) -> compute losses -> cache -> unload
        Phase 2: Load target -> train with cached losses
        """
        # Validate that target model matches reference model (required by DuoLearn)
        if self.model_name != self.ref_model_name:
            raise ValueError(
                f"Target model ({self.model_name}) must match reference model ({self.ref_model_name}). "
                f"Per DuoLearn paper: 'Reference model shares an identical architecture with the training model.' "
                f"The calibrated score s(ti) = L_ref - L_target only makes sense for same-architecture models."
            )
        
        print(f"\n{'='*60}")
        print("Memory-Efficient DuoLearn Training")
        print(f"{'='*60}")
        print(f"Target model: {self.model_name}")
        print(f"Reference model: {self.ref_model_name}")
        print(f"Fine-tune reference: {self.finetune_ref}")
        print(f"{'='*60}\n")

        # ================================================================
        # Phase 1: Reference model processing
        # ================================================================
        ref_model_path = None
        if self.finetune_ref:
            ref_model_path = self._finetune_reference(self.device)

        cache_path = self._compute_and_cache_ref_losses(train_dataset, ref_model_path)

        # ================================================================
        # Phase 2: Target model training with cached losses
        # ================================================================
        print(f"\n{'='*60}")
        print("Phase 2: Training Target Model")
        print(f"{'='*60}")

        # Load target model (GPU is now free)
        self.model, self.tokenizer = load_pythia_model(
            self.model_name, self.device, dtype=torch.bfloat16
        )

        # Full fine-tune (matches original DuoLearn). Enable gradient
        # checkpointing on medium models to keep VRAM in budget.
        for p in self.model.parameters():
            p.requires_grad = True

        model_size = sum(p.numel() for p in self.model.parameters())
        if model_size > 5e9:
            print(f"Large model ({model_size/1e9:.1f}B) -- gradient checkpointing OFF")
        elif model_size > 2e9:
            self.model.gradient_checkpointing_enable()
            print(f"Medium model ({model_size/1e9:.1f}B) -- gradient checkpointing ON")
        else:
            print(f"Small model ({model_size/1e9:.1f}B)")

        total = sum(p.numel() for p in self.model.parameters())
        print(f"Full FT: {total:,} trainable params")

        # Prepare datasets with cached losses
        train_tok = tokenize_and_chunk(train_dataset, self.tokenizer, self.max_length)
        val_tok = tokenize_and_chunk(val_dataset, self.tokenizer, self.max_length)

        # Wrap training data with cached ref losses
        train_cached = CachedRefLossDataset(train_tok, cache_path)

        # Collators
        train_collator = CachedRefLossCollator(self.tokenizer, self.max_length)
        val_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        print(f"\nAlpha (unlearning weight): {self.alpha}")
        print(f"Top-k (hard tokens): {self.top_k*100:.0f}%")
        print(f"Bottom-k (memorized): {self.bottom_k*100:.0f}%")
        print(f"LR: {self.learning_rate}, Epochs: {self.num_epochs}")
        print(f"{'='*60}\n")

        target_run_name = (
            f"target_{self.dataset_name}_{self.model_name}"
            f"_a{self.alpha}_topk{self.top_k}_botk{self.bottom_k}"
        )
        wb_active = self.use_wandb and setup_wandb(
            project=self.wandb_project,
            run_name=target_run_name,
            group=f"{self.dataset_name}_{self.model_name}",
            config={
                "phase": "duolearn_target",
                "dataset": self.dataset_name,
                "model": self.model_name,
                "ref_model": self.ref_model_name,
                "finetune_ref": self.finetune_ref,
                "alpha": self.alpha,
                "top_k": self.top_k,
                "bottom_k": self.bottom_k,
                "lr": self.learning_rate,
                "epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "grad_accum": self.grad_accumulation_steps,
                "max_length": self.max_length,
            },
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accumulation_steps,
            per_device_eval_batch_size=self.batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            logging_steps=10,
            learning_rate=self.learning_rate,
            load_best_model_at_end=True,
            report_to="wandb" if wb_active else "none",
            run_name=target_run_name,
            bf16=torch.cuda.is_available(),
            remove_unused_columns=False,  # Keep ref_token_losses for our custom collator
        )

        trainer = CachedDuoLearnTrainer(
            model=self.model,
            alpha=self.alpha,
            top_k=self.top_k,
            bottom_k=self.bottom_k,
            eval_collator=val_collator,  # Separate collator for evaluation
            args=training_args,
            train_dataset=train_cached,
            eval_dataset=val_tok,
            data_collator=train_collator,
        )

        trainer.train()
        trainer.save_model(f"{output_dir}/best")
        self.tokenizer.save_pretrained(f"{output_dir}/best")

        if wb_active and WANDB_AVAILABLE:
            wandb.finish()

        # Cleanup checkpoints
        import glob, shutil
        for ckpt in glob.glob(f"{output_dir}/checkpoint-*"):
            shutil.rmtree(ckpt)

        # Save config
        with open(f"{output_dir}/duolearn_config.json", "w") as f:
            json.dump({
                "alpha": self.alpha,
                "top_k": self.top_k,
                "bottom_k": self.bottom_k,
                "ref_model": self.ref_model_name,
                "finetune_ref": self.finetune_ref,
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs,
                "full_fine_tune": True,
                "memory_efficient": True,
            }, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Training complete! Model saved to {output_dir}/best")
        print(f"{'='*60}")

        return self.model


# ============================================================================
# CLI
# ============================================================================

def finalize_checkpoint(output_dir: str, base_model_name: str) -> str:
    """Promote the latest checkpoint-N in output_dir to <output_dir>/best.

    Use this when training plateaus and you want to stop early -- scancel the
    job, then run with --finalize to salvage the latest checkpoint instead of
    waiting for the natural end-of-train save_model call.

    Steps:
      1. Find the highest checkpoint-N under output_dir.
      2. Copy it to <output_dir>/best (overwriting if present).
      3. Drop the tokenizer in (HF Trainer does not save tokenizer in checkpoints).
      4. Remove the remaining checkpoint-* dirs.
    """
    import glob, shutil, re
    ckpts = glob.glob(f"{output_dir}/checkpoint-*")
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint-* directories in {output_dir}")
    latest = max(ckpts, key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
    best_dir = f"{output_dir}/best"
    print(f"Promoting {latest} -> {best_dir}")

    if os.path.exists(best_dir):
        shutil.rmtree(best_dir)
    shutil.copytree(latest, best_dir)

    # Tokenizer files are not stored in HF checkpoints -- copy from the base model.
    base_path = f"{HF_DIR}/models/EleutherAI__{base_model_name}"
    tok = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    tok.save_pretrained(best_dir)

    # Drop intermediate checkpoints to match the post-train cleanup behavior.
    for c in ckpts:
        shutil.rmtree(c, ignore_errors=True)
    return best_dir


def main():
    parser = argparse.ArgumentParser(
        description="Memory-Efficient DuoLearn Defense",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Precompute reference losses once
  python defense.py --precompute-only --dataset arxiv --ref-model pythia-2.8b --finetune-ref --gpu 0

  # Step 2: Run multiple training jobs with different alphas (can run in parallel)
  python defense_memory_efficient.py --dataset arxiv --ref-model pythia-2.8b --finetune-ref --alpha 0.2 --gpu 0
  python defense_memory_efficient.py --dataset arxiv --ref-model pythia-2.8b --finetune-ref --alpha 0.5 --gpu 1
  python defense_memory_efficient.py --dataset arxiv --ref-model pythia-2.8b --finetune-ref --alpha 0.8 --gpu 2
        """
    )
    parser.add_argument("--model", type=str, default="pythia-2.8b",
                        help="Target model to defend (must match --ref-model)")
    parser.add_argument("--dataset", type=str, default="Pile-CC")
    parser.add_argument("--max-length", type=int, default=MODEL_MAX_LENGTH)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Target full fine-tune LR. Above the original "
                             "1.75e-6 Pythia recipe so the target memorizes "
                             "enough for DuoLearn to have a signal to defend.")

    # DuoLearn specific
    parser.add_argument("--alpha", type=float, default=0.8,
                        help="Unlearning weight (higher = more unlearning)")
    parser.add_argument("--top-k", type=float, default=0.6,
                        help="Fraction of hard tokens for learning")
    parser.add_argument("--bottom-k", type=float, default=0.2,
                        help="Fraction of memorized tokens for unlearning")
    parser.add_argument("--ref-model", type=str, default="pythia-2.8b",
                        help="Reference model for calibration (must match --model per DuoLearn paper)")
    parser.add_argument("--finetune-ref", action="store_true",
                        help="Fine-tune reference on train split before computing losses")

    parser.add_argument("--gpu", type=int, default=None)

    # wandb logging
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging (default: enabled). "
                             "Reads $WANDB_API_KEY or ~/.wandb_key for the API key.")
    parser.add_argument("--wandb-project", type=str, default="duolearn-defense")

    # Precompute-only mode for efficient hyperparameter sweeps
    parser.add_argument("--precompute-only", action="store_true",
                        help="Only compute and cache reference losses, then exit. "
                             "Use this to precompute once, then run multiple training "
                             "jobs with different alphas in parallel.")

    # Finalize-only mode: promote latest checkpoint of an interrupted run to best/
    parser.add_argument("--finalize", type=str, default=None, metavar="OUTPUT_DIR",
                        help="Promote the latest checkpoint-N in OUTPUT_DIR to "
                             "<OUTPUT_DIR>/best and write duolearn_config.json, "
                             "then exit. Use after scancel'ing a plateaued run.")

    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Finalize-only mode: no model loading, no data, no GPU work -- pure file ops.
    if args.finalize:
        best_dir = finalize_checkpoint(args.finalize, args.model)
        with open(f"{args.finalize}/duolearn_config.json", "w") as f:
            json.dump({
                "alpha": args.alpha,
                "top_k": args.top_k,
                "bottom_k": args.bottom_k,
                "ref_model": args.ref_model,
                "finetune_ref": args.finetune_ref,
                "learning_rate": args.lr,
                "num_epochs": args.num_epochs,
                "full_fine_tune": True,
                "memory_efficient": True,
                "finalized_early": True,
            }, f, indent=2)
        print(f"\nFinalized: {best_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    train_data, val_data = load_defense_data(args.dataset)

    # Precompute-only mode
    if args.precompute_only:
        print("\n" + "="*60)
        print("PRECOMPUTE-ONLY MODE")
        print("="*60)
        print("Will compute reference losses and exit.")
        print("You can then run multiple training jobs with different alphas.")
        print("="*60 + "\n")
        
        defense = DuoLearnMemoryEfficient(
            model_name=args.model,
            device=device,
            ref_model_name=args.ref_model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            finetune_ref=args.finetune_ref,
            dataset_name=args.dataset,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
        )

        # Run only phase 1
        ref_model_path = None
        if args.finetune_ref:
            ref_model_path = defense._finetune_reference(device)

        cache_path = defense._compute_and_cache_ref_losses(train_data, ref_model_path)
        
        print("\n" + "="*60)
        print("PRECOMPUTE COMPLETE")
        print("="*60)
        print(f"Cached losses: {cache_path}")
        print(f"\nYou can now run training with different alphas.")
        print(f"NOTE: --model must match --ref-model ({args.ref_model})")
        print(f"\nExample commands:")
        print(f"  python {__file__} --model {args.ref_model} --dataset {args.dataset} "
              f"--ref-model {args.ref_model} {'--finetune-ref ' if args.finetune_ref else ''}"
              f"--alpha 0.2 --gpu 0")
        print(f"  python {__file__} --model {args.ref_model} --dataset {args.dataset} "
              f"--ref-model {args.ref_model} {'--finetune-ref ' if args.finetune_ref else ''}"
              f"--alpha 0.5 --gpu 1")
        print(f"  python {__file__} --model {args.ref_model} --dataset {args.dataset} "
              f"--ref-model {args.ref_model} {'--finetune-ref ' if args.finetune_ref else ''}"
              f"--alpha 0.8 --gpu 2")
        print("="*60)
        return

    timestamp = datetime.now().strftime("%m%d-%H%M")
    out_dir = (
        f"{OUTPUT_DIR}/{args.model}/{args.dataset}/duolearn_memeff"
        f"_a{args.alpha}_ref{args.ref_model}_{timestamp}"
    )

    # Run memory-efficient DuoLearn
    defense = DuoLearnMemoryEfficient(
        model_name=args.model,
        device=device,
        alpha=args.alpha,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        ref_model_name=args.ref_model,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        finetune_ref=args.finetune_ref,
        dataset_name=args.dataset,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
    )

    defense.train(train_data, val_data, out_dir)
    print(f"\nDefended model saved to: {out_dir}/best")


if __name__ == "__main__":
    main()