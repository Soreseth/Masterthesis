"""
DPSGD Defense for Pythia-2.8B with DP-LoRA (fastDP Ghost Clipping).

Adapted from defense_dpsgd.py for Pythia-2.8B with block_size=1024.
Uses LoRA adapters + fastDP PrivacyEngine with MixOpt (Ghost Clipping).

Continual fine-tuning scenario:
  - parameterlab validation split -> new members (fine-tuning data)
  - parameterlab test split -> non-members (held out for MIA evaluation)

Hyperparameters (from ScaleUP paper):
  - LoRA: r=64, alpha=16, dropout=0.1, target_modules=query_key_value
  - Training: batch_size=8, grad_accum=4, epochs=4, lr=2e-5
  - Block size: 1024 tokens (chunked, every chunk kept)
  - DP: automatic clipping, MixOpt (Ghost Clipping)

Diagnostic additions (2026-04-18):
  - Log effective batch, sample_size, sample rate q before attach
  - Log privacy_engine.noise_multiplier after attach (confirm σ differs across ε)
  - Log per-step loss + total gradient L2 norm for the first --diag-steps batches
    (written to stdout and to {out_dir}/diag.jsonl)
  - optimizer.zero_grad() at the start of each epoch so that leftover
    microbatches from a non-divisible epoch-length do not leak into the next

Usage:
    python defense_dpsgd_pythia2.8b.py --dataset arxiv --epsilon 8.0
    python defense_dpsgd_pythia2.8b.py --dataset arxiv --epsilon inf  # non-DP baseline
    python defense_dpsgd_pythia2.8b.py --dataset arxiv --epsilon inf --lr 2e-4 --epochs 2 \
        --diag-steps 100  # LR sanity check
"""
import os
import json
import argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from tqdm import tqdm
from fastDP import PrivacyEngine
from peft import LoraConfig, get_peft_model, PeftModel

import os as _os
HF_DIR = _os.environ.get("HF_HOME", "./hf_cache")
DATASET_DIR = _os.environ.get("DATASET_DIR", _os.path.join(HF_DIR, "datasets", "parameterlab"))
OUTPUT_DIR = _os.path.join(
    _os.environ.get("DEFENDED_MODELS_ROOT", "./defended_models"),
    "dplora", "epochs_15",
)

# Hyperparameters (matching ScaleUP / existing defense.py DPLoRA)
BLOCK_SIZE = 1024
BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS = 10
LR = 1e-4
LORA_RANK = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
TARGET_MODULES = ["query_key_value"]

DATASETS = [
    "Pile-CC", "arxiv", "FreeLaw", "Github", "HackerNews",
    "OpenWebText2", "USPTO_Backgrounds", "wiki"
]


def load_defense_data(dataset):
    """Load parameterlab Pile data for continual fine-tuning defense."""
    data_dir = f"{DATASET_DIR}/scaling_mia_the_pile_00_{dataset}/data"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset not found at {data_dir}")

    ds = load_dataset("parquet", data_dir=data_dir)
    train_ds = ds["validation"]  # members for fine-tuning
    eval_ds = ds["test"]         # non-members for evaluation

    train_ds = train_ds.filter(lambda x: len(x["text"]) > 100)
    eval_ds = eval_ds.filter(lambda x: len(x["text"]) > 100)

    print(f"Defense data ({dataset}):")
    print(f"  Fine-tuning (validation/members): {len(train_ds)}")
    print(f"  Held-out (test/non-members):       {len(eval_ds)}")
    return train_ds, eval_ds


def tokenize_chunks(dataset, tokenizer, block_size=BLOCK_SIZE,
                    min_tail_tokens=None):
    """Tokenize each document, then split into block_size-token chunks
    (no truncation, every chunk kept). Tail chunks shorter than
    min_tail_tokens are dropped (default: block_size // 4) so we don't
    train on near-empty pads.

    Returns one row per (doc, chunk_offset) pair. The DP sample_size
    therefore counts CHUNKS, not docs -- neighbouring datasets in the DP
    sense differ by one chunk, not one document.
    """
    if min_tail_tokens is None:
        min_tail_tokens = max(1, block_size // 4)
    pad_id = tokenizer.pad_token_id

    def tok_and_chunk(examples):
        toks = tokenizer(examples["text"], truncation=False, padding=False)
        out_ids, out_attn, out_lbls = [], [], []
        for ids in toks["input_ids"]:
            for start in range(0, len(ids), block_size):
                chunk = ids[start:start + block_size]
                if len(chunk) < min_tail_tokens:
                    continue
                pad_len = block_size - len(chunk)
                if pad_len > 0:
                    chunk = chunk + [pad_id] * pad_len
                out_ids.append(chunk)
                out_attn.append([1 if t != pad_id else 0 for t in chunk])
                out_lbls.append([t if t != pad_id else -100 for t in chunk])
        return {"input_ids": out_ids, "attention_mask": out_attn, "labels": out_lbls}

    return dataset.map(
        tok_and_chunk, batched=True,
        remove_columns=dataset.column_names,
    )


def compute_total_grad_norm(model):
    """L2 norm over all trainable params with a populated .grad.
    Cast to float32 to avoid bf16 precision issues.
    Under fastDP 'automatic' clipping, this is post-normalization.
    Under non-DP (ε=∞), this is the raw gradient norm.
    """
    norms = []
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            norms.append(p.grad.detach().float().norm(2))
    if not norms:
        return float("nan")
    return torch.norm(torch.stack(norms)).item()


def dump_dp_engine_state(privacy_engine):
    """Print whatever fastDP exposes. Uses getattr so it won't crash if an
    attribute name differs across fastDP versions."""
    candidates = [
        "noise_multiplier", "sigma",
        "max_grad_norm", "clipping_fn", "clipping_mode", "clipping_style",
        "target_epsilon", "target_delta",
        "sample_size", "batch_size", "epochs", "steps",
        "sample_rate", "num_steps",
    ]
    print("fastDP PrivacyEngine exposed state:")
    for name in candidates:
        if hasattr(privacy_engine, name):
            val = getattr(privacy_engine, name)
            if callable(val):
                continue
            print(f"{name}: {val}")


def main():
    parser = argparse.ArgumentParser(description="DPSGD Defense for Pythia-2.8B")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--epsilon", type=float, default=8.0,
                        help="Privacy budget. Use 'inf' for non-DP baseline.")
    parser.add_argument("--delta", type=float, default=None,
                        help="Privacy delta (default: 1/n_train_samples)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Override the model save root. Default: {OUTPUT_DIR}. "
                             "Use a different root when changing block-size to avoid "
                             "mixing runs from incompatible configs in the same dir.")
    parser.add_argument("--diag-steps", type=int, default=50,
                        help="Log per-step loss and grad norm for the first N batches.")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to an existing LoRA adapter (e.g. '.../dpsgd/defense_dplora_epsinf_r64_a16_0420-1830/final'). "
                             "The adapter is loaded and training continues for --epochs more epochs. "
                             "WARNING for DP runs (epsilon != inf): resuming composes two Gaussian mechanisms, "
                             "so the effective epsilon is LARGER than args.epsilon. "
                             "Do NOT interpret the saved config's epsilon as the true privacy guarantee.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epsilon = args.epsilon
    is_dp = epsilon != float("inf")

    # Model name for saving
    eps_str = f"eps{epsilon}" if is_dp else "epsinf"
    timestamp = datetime.now().strftime("%m%d-%H%M")
    model_name = (f"defense_dplora_{eps_str}_r{args.lora_rank}_a{LORA_ALPHA}_"
                  f"bs{args.block_size}_{timestamp}")

    print(f"\n{'='*60}")
    print(f"  DPSGD Defense (DP-LoRA + fastDP Ghost Clipping)")
    print(f"  Model: Pythia-2.8B")
    print(f"  Dataset: {args.dataset}")
    print(f"  Block size: {args.block_size}")
    print(f"  Epsilon: {epsilon} {'(non-DP baseline)' if not is_dp else ''}")
    print(f"  LoRA: r={args.lora_rank}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"  Training: batch={args.batch_size}, grad_accum={GRAD_ACCUM}, epochs={args.epochs}, lr={args.lr}")
    print(f"{'='*60}\n")

    # Load model
    model_path = f"{HF_DIR}/models/EleutherAI__pythia-2.8b"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True,
        return_dict=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Apply LoRA
    model.enable_input_require_grads()
    if args.resume_from is not None:
        if not os.path.isdir(args.resume_from):
            raise FileNotFoundError(f"--resume-from path not found: {args.resume_from}")
        print(f"\n[RESUME] Loading existing LoRA adapter from {args.resume_from}")
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
        # Make sure LoRA params are trainable even when loading a previously-saved adapter.
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
        if is_dp:
            print("[RESUME][WARNING] Continuing DP training on top of an existing adapter. "
                  "The effective epsilon is composed with the prior run -- the --epsilon value "
                  "passed here is NOT the total privacy guarantee.")
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        )
        model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable:,} trainable / {total:,} total params ({100*trainable/total:.2f}%)")

    # Load and tokenize data -- N samples/doc (block_size chunks)
    train_ds, val_ds = load_defense_data(args.dataset)
    train_tok = tokenize_chunks(train_ds, tokenizer, args.block_size)
    val_tok = tokenize_chunks(val_ds, tokenizer, args.block_size)
    sample_descr = (
        f"block_size={args.block_size} "
        f"(avg ~{len(train_tok) / max(1, len(train_ds)):.1f} chunks/doc)"
    )

    train_tok.set_format("torch")
    val_tok.set_format("torch")

    print(f"Fine-tune samples: {len(train_tok)} ({sample_descr})")
    print(f"Eval samples: {len(val_tok)}")
    print(f"Total steps: {len(train_tok) // args.batch_size * args.epochs}")

    # Delta
    delta = args.delta if args.delta else 1.0 / len(train_tok)

    # DataLoaders
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(
        train_tok, batch_size=args.batch_size, shuffle=True,
        collate_fn=data_collator
    )
    val_loader = DataLoader(
        val_tok, batch_size=args.batch_size, shuffle=False,
        collate_fn=data_collator
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr
    )

    # ---- DIAGNOSTIC 4: pre-attach sanity check ----
    effective_batch = args.batch_size * GRAD_ACCUM
    sample_rate_q = effective_batch / len(train_tok)
    n_microbatches_per_epoch = len(train_loader)
    n_steps_per_epoch = n_microbatches_per_epoch // GRAD_ACCUM
    leftover_microbatches = n_microbatches_per_epoch % GRAD_ACCUM

    print(f"\n[DIAG] Pre-attach config:")
    print(f"  sample_size (n_train):        {len(train_tok)}")
    print(f"  physical batch_size:          {args.batch_size}")
    print(f"  grad_accum:                   {GRAD_ACCUM}")
    print(f"  effective batch_size:         {effective_batch}")
    print(f"  sample rate q = B/N:          {sample_rate_q:.6f}")
    print(f"  microbatches/epoch:           {n_microbatches_per_epoch}")
    print(f"  optimizer steps/epoch:        {n_steps_per_epoch}")
    print(f"  leftover microbatches/epoch:  {leftover_microbatches} "
          f"(discarded via zero_grad at epoch start)")
    print(f"  target_epsilon:               {epsilon}")
    print(f"  target_delta:                 {delta:.3e}")

    # Attach DP engine
    if is_dp:
        privacy_engine = PrivacyEngine(
            model,
            batch_size=args.batch_size * GRAD_ACCUM,
            sample_size=len(train_tok),
            epochs=args.epochs,
            target_epsilon=epsilon,
            target_delta=delta,
            clipping_fn='automatic',
            clipping_mode='MixOpt',
            origin_params=None,
            clipping_style='all-layer',
        )
        privacy_engine.attach(optimizer)
        print(f"\n[DIAG] DP-LoRA attached: target (ε, δ) = ({epsilon}, {delta:.2e})")
        dump_dp_engine_state(privacy_engine)
    else:
        privacy_engine = None
        print(f"\n[DIAG] Non-DP LoRA baseline (no noise, no clipping)")

    # Output directory
    out_dir = f"{args.output_dir}/{args.dataset}/dpsgd/{model_name}"
    os.makedirs(out_dir, exist_ok=True)

    # Save config
    config = {
        "method": "dplora",
        "model": "pythia-2.8b",
        "dataset": args.dataset,
        "epsilon": epsilon,
        "delta": delta,
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "grad_accum": GRAD_ACCUM,
        "effective_batch": effective_batch,
        "sample_rate_q": sample_rate_q,
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_rank": args.lora_rank,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "n_train_samples": len(train_tok),
        "n_val_samples": len(val_tok),
        "diag_steps": args.diag_steps,
        "resumed_from": args.resume_from,
    }
    if privacy_engine is not None:
        config["noise_multiplier"] = getattr(privacy_engine, "noise_multiplier", None)
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Diagnostic JSONL log (one line per logged step)
    diag_path = f"{out_dir}/diag.jsonl"
    diag_fh = open(diag_path, "w")

    # Training loop
    best_val_loss = float("inf")
    global_step = 0  # counts microbatches across all epochs
    for epoch in range(args.epochs):
        # ---- FIX: discard any leftover microbatch gradients from previous epoch ----
        optimizer.zero_grad()

        model.train()
        train_loss = 0
        train_steps = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            # ---- DIAGNOSTIC 2: per-step loss + grad norm, first N microbatches ----
            if global_step < args.diag_steps:
                grad_norm = compute_total_grad_norm(model)
                rec = {
                    "epoch": epoch + 1,
                    "epoch_step": train_steps + 1,
                    "global_step": global_step + 1,
                    "loss": float(loss.item()),
                    "total_grad_norm": grad_norm,
                    "is_dp": bool(is_dp),
                }
                diag_fh.write(json.dumps(rec) + "\n")
                diag_fh.flush()
                # Also print the first few to stdout
                if global_step < 10 or global_step % 10 == 0:
                    print(f"[DIAG step {global_step+1:3d}] "
                          f"loss={rec['loss']:.4f}  grad_norm={grad_norm:.4f}")

            if (train_steps + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()
            train_steps += 1
            global_step += 1

        avg_train_loss = train_loss / train_steps

        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                val_steps += 1

        avg_val_loss = val_loss / val_steps
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(f"{out_dir}/best")
            tokenizer.save_pretrained(f"{out_dir}/best")
            print(f"  Saved best model (val_loss={best_val_loss:.4f})")

    diag_fh.close()

    # Save final model
    model.save_pretrained(f"{out_dir}/final")
    tokenizer.save_pretrained(f"{out_dir}/final")

    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Saved to: {out_dir}")
    print(f"  Diagnostics: {diag_path}")
    if is_dp:
        print(f"  Privacy guarantee: ({epsilon}, {delta:.2e})-DP")


if __name__ == "__main__":
    main()