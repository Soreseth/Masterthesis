from tqdm import tqdm
import random
import datasets
from datasets import load_dataset, concatenate_datasets
import numpy as np
import os 
import pickle
import torch
import json

# Set offline mode to prevent internet access
# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
MIN_CHARS = 100
MIA_SCORE_SAVING_DIR = "/lustre/selvaah3/projects/myproj/output_mia/pythia-2.8b"
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
        
        # Handle Numpy Scalars (float32, int64, etc) -> CRITICAL FIX
        if isinstance(obj, np.number):
            return obj.item()
            
        return super().default(obj)

def safe_truncate(text, max_words=100):
    if isinstance(text, list):
        text = " ".join(text)
    
    words = text.split()
    return " ".join(words[:max_words])

def create_chunks(text, tokenizer, max_length):
    """Create chunks for paragraph scales (512, 1024, 2048)."""
    tokens = tokenizer.encode(text, add_special_tokens=True)
    chunks = [tokenizer.decode(tokens[i:i+max_length], skip_special_tokens=True) for i in range(0, len(tokens), max_length)]
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
    if not doc_lengths: # Safety check if empty
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

        for max_length in [512, 1024, 2048]:

            if not os.path.exists(f"{output_directory}/token_{max_length}"):
                os.makedirs(f"{output_directory}/token_{max_length}")
        
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