from scores import inference, raw_values, min_k, min_k_plus_plus, get_conditional_ll
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from preprocess import create_chunks, save_dataset
from datasets import load_from_disk
from tqdm import tqdm 
import numpy as np

model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-1b",
        cache_dir="models/EleutherAI__pythia-1b",
        local_files_only=False,
        return_dict=True,
        device_map="auto",
        dtype=torch.float16
    )

model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-1b",
    cache_dir="models/EleutherAI__pythia-1b",
    local_files_only=False,
)

if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss, token_log_probs, logits, input_ids = raw_values(sentence="Hello how are you ?", tokenizer=tokenizer, model=model)
print(loss, token_log_probs, np.shape(logits), input_ids)
print(f"Log-Likelihood: {get_conditional_ll(prefix_text="This is London.", target_text="Dürfte ich einen Döner kaufen ?", model=model, tokenizer=tokenizer, device=device)}")
print(f"MIN K%: {min_k(token_probs=token_log_probs, ratio=0.3)}")
print(f"MIN K++%: {min_k_plus_plus(input_ids=input_ids, logits=logits, ratio=0.3)}")