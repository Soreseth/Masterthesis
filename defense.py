"""
defense.py — MIA Defense: Continual Fine-tuning with DP-SGD (Opacus) and DuoLearn.

Simulates a continual fine-tuning scenario with privacy-preserving defenses:
  - parameterlab validation split → new members (fine-tuning data)
  - parameterlab test split → non-members (held out for MIA evaluation)

After defended fine-tuning, MIA attacks can be run on the defended model to
measure whether the defense reduces attack AUC (validation vs test).

Three defense algorithms:
  1. DP-SGD (Abadi et al., CCS 2016) via Opacus
     - Per-example gradient clipping + calibrated Gaussian noise
     - Formal (ε, δ)-differential privacy guarantee
  2. DuoLearn (Tran et al., ACL'25 Findings)
     - Token-level calibration against a frozen reference model
     - Dual loss: learn on hard tokens, unlearn memorized tokens
     - L_dual = L_CE(Th) - α·L_CE(Tm)
  3. DP-LoRA (LoRA + fastDP with Ghost Clipping)
     - LoRA adapters (rank 64, α=16, dropout 0.1) on attention projections
     - fastDP PrivacyEngine with automatic clipping + MixOpt (Ghost Clipping)
     - DP on trainable LoRA parameters only (~100x fewer params)

Usage:
    python defense.py --method dpsgd --model pythia-2.8b --dataset Pile-CC --epsilon 8.0
    python defense.py --method duolearn --model pythia-2.8b --dataset Pile-CC --alpha 0.8
    python defense.py --method dplora --model pythia-2.8b --dataset Pile-CC --epsilon 8.0 --lora-rank 64
"""

import os
import json
import argparse
from datetime import datetime
from typing import Optional
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from tqdm import tqdm
from fastDP import PrivacyEngine
from peft import LoraConfig, get_peft_model, PeftModel

HF_DIR = "/lustre/selvaah3/hf_home"
DATASET_DIR = "/lustre/selvaah3/hf_home/datasets/parameterlab"
OUTPUT_DIR = "/lustre/selvaah3/projects/Masterthesis/defended_models"
MODEL_MAX_LENGTH = 2048

DATASETS = [
    "Pile-CC", "arxiv", 
    "FreeLaw", "Github", "HackerNews",
    "OpenWebText2",
    "USPTO_Backgrounds",
    "wiki"
]

# Data loading
def load_defense_data(dataset: str):
    """Load parameterlab Pile data for continual fine-tuning defense evaluation.

    Continual fine-tuning scenario:
      - validation split → new members (fine-tuning data)
      - test split → non-members (never seen, for MIA evaluation)

    After defended fine-tuning, MIA attacks are run on validation (members)
    vs test (non-members) to measure whether the defense reduces attack AUC.

    Uses the full split sizes as-is (varies per dataset subset).

    Args:
        dataset: Pile subset name (e.g. 'Pile-CC', 'arxiv')
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {DATASETS}")

    data_dir = f"{DATASET_DIR}/scaling_mia_the_pile_00_{dataset}/data"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset not found at {data_dir}")

    ds = load_dataset("parquet", data_dir=data_dir)

    # validation = new members for fine-tuning
    train_ds = ds["validation"]
    # test = non-members (held out for MIA evaluation)
    eval_ds = ds["test"]

    train_ds = train_ds.filter(lambda x: len(x["text"]) > 100)
    eval_ds = eval_ds.filter(lambda x: len(x["text"]) > 100)

    print(f"Defense data ({dataset}):")
    print(f"  Fine-tuning (validation/members): {len(train_ds)}")
    print(f"  Held-out (test/non-members):       {len(eval_ds)}")
    return train_ds, eval_ds


def tokenize_and_chunk(dataset, tokenizer, max_length=MODEL_MAX_LENGTH):
    """Tokenize and chunk documents into fixed-length segments.

    Long documents are split into multiple max_length chunks so no text is
    discarded. The last chunk is kept even if shorter than max_length.
    """
    def chunk_fn(examples):
        all_input_ids = []
        all_attention_mask = []
        for text in examples["text"]:
            tokens = tokenizer(text, truncation=False, padding=False)
            ids = tokens["input_ids"]
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


def tokenize_and_chunk_padded(dataset, tokenizer, max_length=MODEL_MAX_LENGTH):
    """Tokenize, chunk, and pad to max_length (required for DP-SGD fixed batch shapes).

    Long documents are split into multiple max_length chunks. Each chunk is
    padded to max_length. Labels mask padding tokens with -100.
    """
    def chunk_fn(examples):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        for text in examples["text"]:
            tokens = tokenizer(text, truncation=False, padding=False)
            ids = tokens["input_ids"]
            for i in range(0, len(ids), max_length):
                chunk = ids[i:i + max_length]
                pad_len = max_length - len(chunk)
                padded_ids = chunk + [tokenizer.pad_token_id] * pad_len
                mask = [1] * len(chunk) + [0] * pad_len
                labels = chunk + [-100] * pad_len
                all_input_ids.append(padded_ids)
                all_attention_mask.append(mask)
                all_labels.append(labels)
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }

    return dataset.map(
        chunk_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )


# Model loading
def load_pythia_model(model_name: str, device):
    """Load a Pythia model and tokenizer from local HF cache.

    Args:
        model_name: e.g. 'pythia-2.8b', 'pythia-1b'
        device: torch device
    """
    path = f"{HF_DIR}/models/EleutherAI__{model_name}"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")

    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        return_dict=True,
        torch_dtype=torch.float16,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


# Defense 1: DP-SGD via Opacus
# class DPSGD:
#     """DP-SGD defense for continual fine-tuning (Abadi et al., CCS 2016).

#     Full-model DP-SGD via fastDP with automatic clipping and MixOpt.
#     Currently deactivated from CLI — use DPLoRA instead.

#     Args:
#         model: Pre-trained model to defend
#         tokenizer: Corresponding tokenizer
#         device: torch device
#         epsilon: Privacy budget ε
#         delta: Privacy parameter δ
#         learning_rate: Optimizer learning rate
#         batch_size: Per-device batch size
#         grad_accumulation_steps: Gradient accumulation steps
#         num_epochs: Number of fine-tuning epochs
#         max_length: Max token length per sample
#     """

#     def __init__(self, model, tokenizer, device, epsilon=8.0, delta=1e-5,
#                  learning_rate=5e-5, batch_size=8,
#                  grad_accumulation_steps=4, num_epochs=4,
#                  max_length=MODEL_MAX_LENGTH):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.device = device
#         self.epsilon = epsilon
#         self.delta = delta
#         self.learning_rate = learning_rate
#         self.batch_size = batch_size
#         self.grad_accumulation_steps = grad_accumulation_steps
#         self.num_epochs = num_epochs
#         self.max_length = max_length

#     def train(self, train_dataset, val_dataset, output_dir: str):
#         """Run DP-SGD continual fine-tuning.

#         Args:
#             train_dataset: Validation split — new members for continual fine-tuning
#             val_dataset: Test split — non-members held out for MIA evaluation
#             output_dir: Where to save the defended model
#         """
#         train_tok = tokenize_and_chunk_padded(train_dataset, self.tokenizer, self.max_length)
#         val_tok = tokenize_and_chunk_padded(val_dataset, self.tokenizer, self.max_length)

#         train_tok.set_format("torch")
#         val_tok.set_format("torch")

#         data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
#         train_loader = DataLoader(
#             train_tok, batch_size=self.batch_size, shuffle=True,
#             collate_fn=data_collator
#         )
#         val_loader = DataLoader(
#             val_tok, batch_size=self.batch_size, shuffle=False,
#             collate_fn=data_collator
#         )

#         optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

#         # fastDP PrivacyEngine with Ghost Clipping (MixOpt)
#         privacy_engine = PrivacyEngine(
#             self.model,
#             batch_size=self.batch_size * self.grad_accumulation_steps,
#             sample_size=len(train_tok),
#             epochs=self.num_epochs,
#             target_epsilon=self.epsilon,
#             target_delta=self.delta,
#             clipping_fn='automatic',
#             clipping_mode='MixOpt',
#             origin_params=None,
#             clipping_style='all-layer',
#         )
#         privacy_engine.attach(optimizer)

#         print(f"\n{'='*60}")
#         print(f"DP-SGD Defense (fastDP)")
#         print(f"{'='*60}")
#         print(f"Target (ε, δ) = ({self.epsilon}, {self.delta})")
#         print(f"Clipping: automatic, MixOpt (Ghost Clipping), all-layer")
#         print(f"LR: {self.learning_rate}, Epochs: {self.num_epochs}")
#         print(f"{'='*60}\n")

#         best_val_loss = float("inf")
#         os.makedirs(output_dir, exist_ok=True)

#         for epoch in range(self.num_epochs):
#             self.model.train()
#             train_loss_sum = 0.0
#             train_steps = 0

#             for batch in tqdm(train_loader, desc=f"DP-SGD Epoch {epoch+1}/{self.num_epochs}"):
#                 batch = {k: v.to(self.device) for k, v in batch.items()}
#                 outputs = self.model(**batch)
#                 loss = outputs.loss

#                 loss.backward()

#                 if (train_steps + 1) % self.grad_accumulation_steps == 0:
#                     optimizer.step()
#                     optimizer.zero_grad()

#                 train_loss_sum += loss.item()
#                 train_steps += 1

#             avg_train = train_loss_sum / max(train_steps, 1)

#             self.model.eval()
#             val_loss_sum = 0.0
#             val_steps = 0
#             with torch.no_grad():
#                 for batch in val_loader:
#                     batch = {k: v.to(self.device) for k, v in batch.items()}
#                     val_loss_sum += self.model(**batch).loss.item()
#                     val_steps += 1

#             avg_val = val_loss_sum / max(val_steps, 1)
#             eps = privacy_engine.get_epsilon(self.delta)
#             print(f"Epoch {epoch+1}: train={avg_train:.4f}, val={avg_val:.4f}, ε={eps:.2f}")

#             # Save after every epoch
#             self.model.save_pretrained(f"{output_dir}/epoch-{epoch+1}")
#             self.tokenizer.save_pretrained(f"{output_dir}/epoch-{epoch+1}")

#             if avg_val < best_val_loss:
#                 best_val_loss = avg_val
#                 self.model.save_pretrained(f"{output_dir}/best")
#                 self.tokenizer.save_pretrained(f"{output_dir}/best")

#         final_eps = privacy_engine.get_epsilon(self.delta)
#         print(f"\nDP-SGD complete. Final (ε={final_eps:.2f}, δ={self.delta})")
#         print(f"Best val loss: {best_val_loss:.4f}")

#         with open(f"{output_dir}/dpsgd_config.json", "w") as f:
#             json.dump({
#                 "epsilon_achieved": final_eps,
#                 "delta": self.delta,
#                 "learning_rate": self.learning_rate,
#                 "num_epochs": self.num_epochs,
#                 "best_val_loss": best_val_loss,
#             }, f, indent=2)

#         return self.model


# Defense 2: DuoLearn (Tran et al., ACL'25)
class _DuoLearnTrainer(Trainer):
    """Internal HF Trainer subclass for DuoLearn's dual loss.

    Token-level calibration using a frozen reference model:
      s(ti) = loss_ref(ti) - loss_target(ti)
    Hard tokens (high s) → gradient descent (learning)
    Memorized tokens (low s) → gradient ascent (unlearning)
    """

    def __init__(self, *args, ref_model, alpha, top_k, bottom_k, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.alpha = alpha
        self.top_k = top_k
        self.bottom_k = bottom_k

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        if not model.training:
            outputs = model(**inputs)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        outputs = model(**inputs)
        logits = outputs.logits

        with torch.no_grad():
            ref_logits = self.ref_model(**inputs).logits

        labels = inputs["labels"]

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_ref_logits = ref_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Filter padding tokens
        mask = shift_labels != -100
        valid_logits = shift_logits[mask]
        valid_ref_logits = shift_ref_logits[mask]
        valid_labels = shift_labels[mask]
        num_tokens = valid_labels.numel()

        if num_tokens == 0:
            zero = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return (zero, outputs) if return_outputs else zero

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        token_loss = loss_fn(valid_logits, valid_labels)
        ref_token_loss = loss_fn(valid_ref_logits, valid_labels)

        # Calibrated score: s(ti) = ref_loss - model_loss
        calibrated_score = ref_token_loss - token_loss

        # Hard tokens: top-k by calibrated score (learn these)
        num_hard = max(1, int(self.top_k * num_tokens))
        _, hard_idx = torch.topk(calibrated_score, num_hard)
        hard_loss = token_loss[hard_idx].mean()

        # Memorized tokens: bottom-k by calibrated score (unlearn these)
        num_mem = max(1, int(self.bottom_k * num_tokens))
        _, mem_idx = torch.topk(calibrated_score, num_mem, largest=False)
        mem_loss = token_loss[mem_idx].mean()

        # L_dual = L_CE(hard) - α·L_CE(memorized)
        dual_loss = hard_loss - self.alpha * mem_loss

        return (dual_loss, outputs) if return_outputs else dual_loss


class DuoLearn:
    """
    Token-level calibration against a frozen reference model.
    Dual loss: learn on hard tokens, unlearn memorized tokens.

    Args:
        model: Pre-trained target model to defend
        tokenizer: Corresponding tokenizer
        device: torch device
        alpha: Unlearning weight
        top_k: Fraction of hard tokens for learning
        bottom_k: Fraction of memorized tokens for unlearning
        ref_model_name: Reference model for calibration (e.g. 'pythia-1b')
        learning_rate: Optimizer learning rate
        batch_size: Per-device batch size
        grad_accumulation_steps: Gradient accumulation steps
        num_epochs: Number of fine-tuning epochs
        max_length: Max token length per sample
        eval_steps: Evaluation frequency in steps
        lora_rank: LoRA rank (ScaleUP: 64)
        lora_alpha: LoRA alpha scaling (ScaleUP: 16)
        lora_dropout: LoRA dropout rate (ScaleUP: 0.1)
        target_modules: Which modules to apply LoRA to
    """

    def __init__(self, model, tokenizer, device, alpha=0.8, top_k=0.6,
                 bottom_k=0.2, ref_model_name="pythia-1b", learning_rate=1.75e-6,
                 batch_size=8, grad_accumulation_steps=4, num_epochs=10,
                 max_length=1024, eval_steps=100, lora_rank=64, lora_alpha=16,
                 lora_dropout=0.1, target_modules=None):
        self.tokenizer = tokenizer
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
        self.eval_steps = eval_steps
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["query_key_value"]

        # Apply LoRA adapters (ScaleUP: r=64, α=16, dropout=0.1)
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.target_modules,
        )
        model.enable_input_require_grads()
        self.model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"LoRA: {trainable:,} trainable / {total:,} total params "
              f"({100*trainable/total:.2f}%)")

    def train(self, train_dataset, val_dataset, output_dir: str):
        """Run DuoLearn continual fine-tuning.

        Args:
            train_dataset: Validation split — new members for continual fine-tuning
            val_dataset: Test split — non-members held out for MIA evaluation
            output_dir: Where to save the defended model
        """
        # Load frozen reference model
        ref_model, _ = load_pythia_model(self.ref_model_name, self.device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

        train_tok = tokenize_and_chunk(train_dataset, self.tokenizer, self.max_length)
        val_tok = tokenize_and_chunk(val_dataset, self.tokenizer, self.max_length)

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accumulation_steps,
            per_device_eval_batch_size=self.batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            learning_rate=self.learning_rate,
            load_best_model_at_end=True,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        print(f"\n{'='*60}")
        print(f"DuoLearn Defense (ACL'25 Findings) + LoRA")
        print(f"{'='*60}")
        print(f"Alpha (unlearning weight): {self.alpha}")
        print(f"Top-k (hard tokens): {self.top_k*100:.0f}%")
        print(f"Bottom-k (memorized): {self.bottom_k*100:.0f}%")
        print(f"Reference model: {self.ref_model_name}")
        print(f"LoRA rank={self.lora_rank}, α={self.lora_alpha}, "
              f"dropout={self.lora_dropout}")
        print(f"LR: {self.learning_rate}, Epochs: {self.num_epochs}")
        print(f"{'='*60}\n")

        trainer = _DuoLearnTrainer(
            model=self.model,
            ref_model=ref_model,
            alpha=self.alpha,
            top_k=self.top_k,
            bottom_k=self.bottom_k,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model(f"{output_dir}/best")
        self.tokenizer.save_pretrained(f"{output_dir}/best")

        with open(f"{output_dir}/duolearn_config.json", "w") as f:
            json.dump({
                "alpha": self.alpha,
                "top_k": self.top_k,
                "bottom_k": self.bottom_k,
                "ref_model": self.ref_model_name,
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs,
            }, f, indent=2)

        del ref_model
        torch.cuda.empty_cache()

        return self.model


# Defense 3: DP-LoRA (LoRA + fastDP with Ghost Clipping)
class DPLoRA:
    """DP-LoRA defense using fastDP with Ghost Clipping (MixOpt).

    Injects LoRA adapters into attention projections, then applies
    fastDP's PrivacyEngine with automatic clipping and MixOpt mode
    (Ghost Clipping) for memory-efficient per-sample gradients.

    Hyperparameters follow the ScaleUP paper (continual fine-tuning scenario):
      batch_size=8, epochs=4, block_size=1024, lr=2e-4,
      LoRA r=64 α=16 dropout=0.1.

    Args:
        model: Pre-trained model to defend
        tokenizer: Corresponding tokenizer
        device: torch device
        epsilon: Privacy budget ε (use float('inf') for non-DP LoRA baseline)
        delta: Privacy parameter δ (default: 1/n_train_samples)
        learning_rate: Optimizer learning rate (ScaleUP: 2e-4)
        batch_size: Per-device batch size (ScaleUP: 8)
        grad_accumulation_steps: Gradient accumulation steps
        num_epochs: Number of fine-tuning epochs (ScaleUP: 4)
        max_length: Block size / max token length (ScaleUP: 1024)
        lora_rank: LoRA rank (ScaleUP: 64)
        lora_alpha: LoRA alpha scaling (ScaleUP: 16)
        lora_dropout: LoRA dropout rate (ScaleUP: 0.1)
        target_modules: Which modules to apply LoRA to
    """

    def __init__(self, model, tokenizer, device, epsilon=10.0, delta=None,
                 learning_rate=2e-4, batch_size=8,
                 grad_accumulation_steps=4, num_epochs=10,
                 max_length=1024, lora_rank=64, lora_alpha=16,
                 lora_dropout=0.1, target_modules=None):
        self.tokenizer = tokenizer
        self.device = device
        self.epsilon = epsilon
        self.delta = delta  # set to 1/n from training data size if None
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.grad_accumulation_steps = grad_accumulation_steps
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["query_key_value"]

        # Apply LoRA adapters
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.target_modules,
        )
        model.enable_input_require_grads()
        self.model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"LoRA: {trainable:,} trainable / {total:,} total params "
              f"({100*trainable/total:.2f}%)")

    def train(self, train_dataset, val_dataset, output_dir: str):
        """Run DP-LoRA continual fine-tuning with fastDP Ghost Clipping.

        Args:
            train_dataset: Validation split — new members for continual fine-tuning
            val_dataset: Test split — non-members held out for MIA evaluation
            output_dir: Where to save the defended model
        """
        train_tok = tokenize_and_chunk_padded(train_dataset, self.tokenizer, self.max_length)
        val_tok = tokenize_and_chunk_padded(val_dataset, self.tokenizer, self.max_length)

        train_tok.set_format("torch")
        val_tok.set_format("torch")

        # δ = 1/n if not explicitly set
        if self.delta is None:
            self.delta = 1.0 / len(train_tok)

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        train_loader = DataLoader(
            train_tok, batch_size=self.batch_size, shuffle=True,
            collate_fn=data_collator
        )
        val_loader = DataLoader(
            val_tok, batch_size=self.batch_size, shuffle=False,
            collate_fn=data_collator
        )

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate
        )

        is_dp = self.epsilon != float("inf")

        if is_dp:
            # fastDP PrivacyEngine with Ghost Clipping (MixOpt)
            privacy_engine = PrivacyEngine(
                self.model,
                batch_size=self.batch_size * self.grad_accumulation_steps,
                sample_size=len(train_tok),
                epochs=self.num_epochs,
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                clipping_fn='automatic',
                clipping_mode='MixOpt',
                origin_params=None,
                clipping_style='all-layer',
            )
            privacy_engine.attach(optimizer)

        print(f"\n{'='*60}")
        print(f"DP-LoRA Defense (PEFT + fastDP Ghost Clipping)")
        print(f"{'='*60}")
        if is_dp:
            print(f"Target (ε, δ) = ({self.epsilon}, {self.delta})")
            print(f"Clipping: automatic, MixOpt (Ghost Clipping), all-layer")
        else:
            print(f"Non-DP baseline (ε=∞)")
        print(f"LoRA rank={self.lora_rank}, α={self.lora_alpha}, "
              f"dropout={self.lora_dropout}")
        print(f"Target modules: {self.target_modules}")
        print(f"LR: {self.learning_rate}, Epochs: {self.num_epochs}")
        print(f"{'='*60}\n")

        best_val_loss = float("inf")
        os.makedirs(output_dir, exist_ok=True)

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss_sum = 0.0
            train_steps = 0

            for batch in tqdm(train_loader, desc=f"DP-LoRA Epoch {epoch+1}/{self.num_epochs}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss

                loss.backward()

                if (train_steps + 1) % self.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss_sum += loss.item()
                train_steps += 1

            avg_train = train_loss_sum / max(train_steps, 1)

            # Validation
            self.model.eval()
            val_loss_sum = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    val_loss_sum += self.model(**batch).loss.item()
                    val_steps += 1

            avg_val = val_loss_sum / max(val_steps, 1)

            if is_dp:
                spent = privacy_engine.get_privacy_spent()
                eps = spent.get("eps_rdp", spent.get("eps_low", "?"))
                print(f"Epoch {epoch+1}: train={avg_train:.4f}, val={avg_val:.4f}, ε={eps}")
            else:
                print(f"Epoch {epoch+1}: train={avg_train:.4f}, val={avg_val:.4f}")

            # Save LoRA adapters after every epoch
            self.model.save_pretrained(f"{output_dir}/epoch-{epoch+1}")
            self.tokenizer.save_pretrained(f"{output_dir}/epoch-{epoch+1}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                self.model.save_pretrained(f"{output_dir}/best")
                self.tokenizer.save_pretrained(f"{output_dir}/best")

        if is_dp:
            spent = privacy_engine.get_privacy_spent()
            final_eps = spent.get("eps_rdp", spent.get("eps_low", "?"))
            print(f"\nDP-LoRA complete. Final ε={final_eps}, δ={self.delta}")
        else:
            final_eps = float("inf")
            print(f"\nLoRA baseline complete (no DP).")
        print(f"Best val loss: {best_val_loss:.4f}")

        with open(f"{output_dir}/dplora_config.json", "w") as f:
            json.dump({
                "epsilon_achieved": final_eps if final_eps != float("inf") else "inf",
                "delta": self.delta,
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "target_modules": self.target_modules,
                "best_val_loss": best_val_loss,
            }, f, indent=2)

        return self.model


# CLI
def main():
    parser = argparse.ArgumentParser(description="MIA Defense: DP-SGD or DuoLearn")
    parser.add_argument("--method", type=str, required=True,
                        choices=["duolearn", "dplora"],
                        help="Defense method")
    parser.add_argument("--model", type=str, default="pythia-2.8b",
                        help="Pythia model name (e.g. pythia-2.8b, pythia-1b)")
    parser.add_argument("--dataset", type=str, default="Pile-CC",
                        help="Pile subset from parameterlab (e.g. Pile-CC, arxiv)")
    parser.add_argument("--max-length", type=int, default=MODEL_MAX_LENGTH,
                        help="Max token length per sample (Pythia context: 2048)")
    parser.add_argument("--num-epochs", type=int, default=3,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: method-specific)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Per-device batch size")

    # DP specific (3 levels: inf=no privacy, 10=moderate, 1=strong)
    parser.add_argument("--epsilon", type=float, default=10.0,
                        help="DP: privacy budget ε (inf=no DP, 10=moderate, 1=strong)")
    parser.add_argument("--delta", type=float, default=None,
                        help="DP: privacy parameter δ (default: 1/n_train_samples)")

    # DuoLearn specific
    parser.add_argument("--alpha", type=float, default=0.8,
                        help="DuoLearn: unlearning weight")
    parser.add_argument("--top-k", type=float, default=0.6,
                        help="DuoLearn: hard token fraction")
    parser.add_argument("--bottom-k", type=float, default=0.2,
                        help="DuoLearn: memorized token fraction")
    parser.add_argument("--ref-model", type=str, default="pythia-1b",
                        help="DuoLearn: reference model for calibration")

    # DP-LoRA specific
    parser.add_argument("--lora-rank", type=int, default=64,
                        help="DP-LoRA: LoRA rank (default 64, Puerto et al.)")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="DP-LoRA: LoRA alpha scaling")
    parser.add_argument("--lora-dropout", type=float, default=0.1,
                        help="DP-LoRA: LoRA dropout rate")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index to use (e.g. 5 for NVIDIA GPU #5)")

    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load target model
    model, tokenizer = load_pythia_model(args.model, device)

    # Load defense data: validation=members (fine-tune), test=non-members (MIA eval)
    train_data, val_data = load_defense_data(args.dataset)

    # δ = 1/n (standard choice: probability of privacy failure scales with dataset)
    if args.delta is None:
        args.delta = 1.0 / len(train_data)
        print(f"δ = 1/n = {args.delta:.2e} (n={len(train_data)})")

    timestamp = datetime.now().strftime("%m%d-%H%M")
    base_out = f"{OUTPUT_DIR}/{args.model}/{args.dataset}"

    if args.method == "duolearn":
        defense = DuoLearn(
            model, tokenizer, device,
            alpha=args.alpha,
            top_k=args.top_k,
            bottom_k=args.bottom_k,
            ref_model_name=args.ref_model,
            learning_rate=args.lr or 1.75e-6,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            max_length=args.max_length,
        )
        out_dir = (
            f"{base_out}/duolearn"
            f"_a{defense.alpha}"
            f"_topk{defense.top_k}"
            f"_botk{defense.bottom_k}"
            f"_ref{defense.ref_model_name}"
            f"_{timestamp}"
        )
        defense.train(train_data, val_data, out_dir)

    elif args.method == "dplora":
        defense = DPLoRA(
            model, tokenizer, device,
            epsilon=args.epsilon,
            delta=args.delta,
            learning_rate=args.lr or 2e-4,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            max_length=args.max_length,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        eps_str = "inf" if args.epsilon == float("inf") else f"{defense.epsilon}"
        out_dir = (
            f"{base_out}/dplora"
            f"_eps{eps_str}"
            f"_r{defense.lora_rank}"
            f"_a{defense.lora_alpha}"
            f"_{timestamp}"
        )
        defense.train(train_data, val_data, out_dir)

    print(f"\nDefended model saved to: {out_dir}/best")

if __name__ == "__main__":
    main()
