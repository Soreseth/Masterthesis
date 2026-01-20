from tqdm import tqdm
import random
import datasets
from datasets import load_dataset, concatenate_datasets, load_from_disk
import numpy as np
import os 
import pickle
import torch
import json
from collections import defaultdict
import math
from nltk.tokenize import sent_tokenize
from scores import raw_values, CIMIA
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set offline mode to prevent internet access
# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
MODEL_MAX_LENGTH = 2048
MIN_CHARS = 100
MIA_SCORE_SAVING_DIR = "/lustre/selvaah3/projects/Masterthesis/output_mia/pythia-2.8b"
HF_DIR = "/lustre/selvaah3/hf_home"
DIVISOR = 1000

seed = random.seed() 
rng = np.random.default_rng(seed=seed)

class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle PyTorch Tensors
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        
        # Handle Numpy Arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle Numpy Scalars
        if isinstance(obj, np.number):
            return obj.item()
            
        return super().default(obj)

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.nn import CrossEntropyLoss
import gc

def compute_batch_scores(tokenizer, model, device, target_texts, prefix_text=None, batch_size=4):
    """
    Computes Log-Likelihoods for a batch of targets against a single prefix.
    strictly enforces separate truncation for prefix and targets.
    """
    scores = []
    
    # 1. Prepare Prefix (Once per batch loop)
    # We tokenize the prefix ONLY ONCE to save CPU time
    if prefix_text:
        prefix_tokens = tokenizer(
            prefix_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512, 
            add_special_tokens=False
        )
        prefix_ids = prefix_tokens.input_ids.to(device) # Shape: [1, Seq_Len]
        prefix_len = prefix_ids.shape[1]
    else:
        prefix_ids = None
        prefix_len = 0

    # 2. Process Targets in Batches
    for i in range(0, len(target_texts), batch_size):
        batch_targets = target_texts[i:i + batch_size]
        
        # Tokenize Batch of Targets (Strict Max 512 + Padding)
        target_tokens = tokenizer(
            batch_targets,
            return_tensors="pt",
            padding=True,           # Pad to longest in batch
            truncation=True,        # Strict truncation
            max_length=512,         # Strict max length for target
            add_special_tokens=False
        ).to(device)
        
        target_ids = target_tokens.input_ids
        target_mask = target_tokens.attention_mask
        
        # 3. Concatenate Prefix + Targets
        if prefix_ids is not None:
            # Expand prefix to match batch size: [1, L] -> [Batch, L]
            current_batch_size = target_ids.shape[0]
            batch_prefix_ids = prefix_ids.expand(current_batch_size, -1)
            batch_prefix_mask = torch.ones_like(batch_prefix_ids)
            
            # Concat: [Prefix, Target]
            input_ids = torch.cat((batch_prefix_ids, target_ids), dim=1)
            attention_mask = torch.cat((batch_prefix_mask, target_mask), dim=1)
        else:
            input_ids = target_ids
            attention_mask = target_mask

        # 4. Model Forward Pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            
            # Shift for Causal LM Loss
            # Logits: predict next token (remove last)
            shift_logits = logits[..., :-1, :].contiguous()
            # Labels: are the next tokens (remove first)
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Loss per token
            loss_fct = CrossEntropyLoss(reduction='none')
            # Flatten batch for loss calculation
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size())
            
            # Mask Padding (so pads don't count towards score)
            shift_mask = attention_mask[..., 1:].contiguous()
            loss = loss * shift_mask
            
            # Average Loss per Sequence
            seq_lens = shift_mask.sum(dim=1)
            # return negative loss
            batch_scores = - (loss.sum(dim=1) / torch.clamp(seq_lens, min=1))
            
            scores.extend(batch_scores.cpu().numpy())
            
        # Cleanup memory
        del input_ids, attention_mask, logits, loss
    
    return np.array(scores)

def topPref(candidate_prefixes, validation_texts, validation_labels, tokenizer, model, device, num_prefix:int, batch_size=8):
    """ 
    Fast TopPref using Batched Inference.
    """
    ranked_results = []

    print(f"Start search for optimal prefix under {len(candidate_prefixes)} candidates ...")
    unconditional_scores = compute_batch_scores(
        tokenizer, model, device, validation_texts, prefix_text=None, batch_size=batch_size
    )

    # ---------------------------------------------------------
    # 2. Search Loop
    # ---------------------------------------------------------
    print("Testing prefixes...")
    for prefix in tqdm(candidate_prefixes, desc="Testing Prefixes"):
        
        # Compute Conditional Scores (Batched)
        # Process ALL validation texts against THIS prefix in batches
        conditional_scores = compute_batch_scores(
            tokenizer, model, device, validation_texts, prefix_text=prefix, batch_size=batch_size
        )
        
        # Vectorized ReCaLL Calculation
        recall_scores = conditional_scores / (unconditional_scores + 1e-10)
        
        # AUC Calculation
        try:
            current_auc = roc_auc_score(validation_labels, recall_scores)
        except ValueError:
            current_auc = 0.5 
        
        ranked_results.append((current_auc, prefix))

    # Sort
    ranked_results.sort(key=lambda x: x[0], reverse=True)

    # Output
    print("-" * 40)
    if ranked_results:
        best_score, best_p = ranked_results[0]
        print(f"WINNER PREFIX: '{best_p[:50]}...'")
        print(f"PERFECT SCORE: {best_score:.4f}")
        
    return ranked_results[:num_prefix]

def safe_pre_encode_shots(text_list, tokenizer, max_shot_len:int):
    """
    Encodes each shot individually and returns a list of tensors.
    """
    encoded_shots = []
    for text in text_list:
        # Encode individually
        enc = tokenizer.encode(
            text, 
            add_special_tokens=False, 
            return_tensors='pt', 
            truncation=True, 
            max_length=max_shot_len
        )
        encoded_shots.append(enc) # Append distinct tensor
    return encoded_shots

def create_chunks(text, tokenizer, max_length):
    """
    Create chunks for paragraph scales (512, 1024, 2048).
    For sentence scale (avg. 43) we use the nltk tokenizer
    """
    if max_length != 43:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        chunks = [tokenizer.decode(tokens[i:i+max_length], skip_special_tokens=True) for i in range(0, len(tokens), max_length)]
    else:
        chunks = sent_tokenize(text)
    return chunks

def mia_dataset(dataset_path: str):
    """
    Load datasets and set train datapoints as members and validation/test datapoints as non-members. 
    """
    # Dataset from MIA-Scaling
    data = load_dataset("parquet", data_files={"train": f"{dataset_path}/data/train-*.parquet", "validation": f"{dataset_path}/data/validation-*.parquet", "test": f"{dataset_path}/data/test-*.parquet"})

    # Filter outliers (too small text)
    data["train"] = data["train"].filter(lambda x: len(x["text"]) > MIN_CHARS)
    data["validation"] = data["validation"].filter(lambda x: len(x["text"]) > MIN_CHARS)
    data["test"] = data["test"].filter(lambda x: len(x["text"]) > MIN_CHARS)

    non_members = concatenate_datasets([data["validation"], data["test"]])
    
    # Ensure balanced dataset and similar lengths
    doc_lengths = [len(text['text']) for text in non_members]
    if not doc_lengths:
        min_len, max_len = 0, 0
    else:
        min_len = min(doc_lengths)
        max_len = max(doc_lengths)

    non_members = datasets.Dataset.from_list(non_members)
    members = data['train'].filter(lambda x: min_len <= len(x["text"]) and len(x["text"]) <= max_len).shuffle(seed=seed).select(range(len(non_members)))

    return members, non_members

def save_dataset():
    dataset_path = "/lustre/selvaah3/hf_home/datasets/parameterlab"

    for dataset_name in os.listdir(dataset_path):

        output_directory = f"{MIA_SCORE_SAVING_DIR}/{dataset_name.split('scaling_mia_the_pile_00_', 1)[1]}"
        total_dataset_path = dataset_path + "/" + dataset_name
        members, non_members = mia_dataset(total_dataset_path)

        if not os.path.exists(os.path.join(output_directory, "non_members")):
            non_members.save_to_disk(os.path.join(output_directory, "non_members"))
            members.save_to_disk(os.path.join(output_directory, "members"))

        # Sentence-, Paragraph- and Document-Level
        for max_length in [43, 512, 1024, 2048]:
            if not os.path.exists(f"{output_directory}/paragraph_{max_length}"):
                os.makedirs(f"{output_directory}/paragraph_{max_length}")
            else: 
                print("Output directory already existing... ")
            
            if not os.path.exists(f"{output_directory}/document_{max_length}"):
                os.makedirs(f"{output_directory}/document_{max_length}")
            else: 
                print("Output directory already existing... ")

        # Collection level
        if not os.path.exists(f"{output_directory}/collection"):
            os.makedirs(f"{output_directory}/collection")
        else: 
            print("Output directory already existing... ")
                
def build_freq_dist(save_path: str, dataset_path:str, base_tokenizer):
    """
    Load allenai/c4-en datasets and grab all samples to determine token frequency. Then save the list  
    """

    # Dataset from MIA-Scaling
    data = load_dataset("json", data_files={"train": f"{dataset_path}/c4-train*.json.gz", "validation": "{dataset_path}/c4-validation*.json.gz"}, streaming=True)

    freq_dist = [0] * len(base_tokenizer)
    for sample in data["train"].iter(batch_size=100000):
        outputs = base_tokenizer(sample['text'], max_length=2048, truncation=True)

        for input_ids in outputs["input_ids"]:
            for token_id in input_ids:
                if token_id < len(freq_dist):
                    freq_dist[token_id] += 1

    print("Frequency Distribution: Finished with train set")
    for sample in data["validation"].iter(batch_size=100000):
        outputs = base_tokenizer(sample['text'], max_length=2048, truncation=True)

        for input_ids in outputs["input_ids"]:
            for token_id in input_ids:
                if token_id < len(freq_dist):
                    freq_dist[token_id] += 1

    print("Frequency Distribution: Finished with validation set")
    print("Saving the frequency distribution to freq_dist.pkl ...")
    # Change "/" to '/' inside the split()
    with open(f"{save_path}/{type(base_tokenizer).__name__}_{dataset_path.split('/')[:-1]}_freq_dist.pkl", "wb") as f:
        pickle.dump(freq_dist, f)
        
def collect_calibration_signals(non_member_path: str, save_path: str, token_level: int, model, tokenizer, device):
    
    non_member_dataset = load_from_disk(dataset_path=non_member_path)
    num_signals = min(int(len(non_member_dataset['text'])*0.5), 1000)
    
    if os.path.exists(f"{save_path}/calibration_signals.json"):
        print("File already exists.")
    else:
        calibration_data = defaultdict(list)
        non_member_dataset = non_member_dataset['text'][:num_signals]

        mia = CIMIA(
            target_model=model, 
            target_tokenizer=tokenizer, 
            device=device, 
            max_len=token_level,
            calibration_signal = None  # Leave empty for calibration phase
        )

        # Loop over single texts
        for i, text in tqdm(enumerate(non_member_dataset), total=len(non_member_dataset), desc="Collecting calibration signals from non-member samples..."):
            res = raw_values(text, model, tokenizer, device)
            
            # Raw Mode
            raw_signals = mia.predict(
                input_ids=res['input_ids'].squeeze(0), 
                token_log_probs=res['token_log_probs'], 
                loss=res['loss'].item(), 
                raw_signals=True
            )
            
            for key, value in raw_signals.items():
                calibration_data[key].append(value)

        # Save
        output_file = f"{save_path}/calibration_signals.json"
        with open(output_file, "w") as f:
            json.dump(calibration_data, f, cls=TensorEncoder)

        print(f"Sample saved in {output_file}")
    return num_signals