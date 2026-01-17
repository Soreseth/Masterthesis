from scores import inference, RelativeLikelihood, DCPDD, OfflineRobustMIA, NoisyNeighbour, CIMIA, TagTab
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
import time
from preprocess import create_chunks, save_dataset, build_freq_dist, safe_pre_encode_shots, TensorEncoder, collect_calibration_signals
from datasets import load_from_disk
from tqdm import tqdm
import traceback
import argparse
import math
import spacy
import numpy as np
import pickle

os.environ["HF_HUB_OFFLINE"] = "1"
MIN_CHARS = 100
MODEL_MAX_LENGTH = 2048
MIA_SCORE_SAVING_DIR = "output_mia"
HF_DIR = "/lustre/selvaah3/hf_home"
os.environ['HF_HOME'] = HF_DIR

def get_mapped_value(x):
    mapping = {
        43: 10,
        512: 5,
        1024: 2,
        2048: 1
    }
    
    return mapping.get(x, 1)

def main(pythia_model: str, max_length: int, miaset: str, dataset: str, dataset_range: list[int]):
    
    model = AutoModelForCausalLM.from_pretrained(
        f"{HF_DIR}/models/EleutherAI__{pythia_model}",
        local_files_only=True,
        return_dict=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        f"{HF_DIR}/models/EleutherAI__{pythia_model}",
        local_files_only=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    if pythia_model == "pythia-2.8b":
        reference_model_name = "pythia-1.4b"
    elif pythia_model == "pythia-6.9b":
        reference_model_name = "pythia-1.4b"
    else:
        reference_model_name = "pythia-1.4b" 

    reference_model = AutoModelForCausalLM.from_pretrained(
        f"{HF_DIR}/models/EleutherAI__{reference_model_name}",
        local_files_only=False,
        return_dict=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
        
    reference_model.eval()
    
    reference_tokenizer = AutoTokenizer.from_pretrained(
        f"{HF_DIR}/models/EleutherAI__{reference_model_name}",
        local_files_only=False,
    )

    if reference_tokenizer.pad_token is None:
        reference_tokenizer.pad_token = reference_tokenizer.eos_token
    reference_tokenizer.pad_token_id = reference_tokenizer.eos_token_id 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_directory = f"{MIA_SCORE_SAVING_DIR}/{pythia_model}/{dataset}"
    if not os.path.exists(f"{output_directory}/members") or not os.path.exists(f"{output_directory}/non_members"):
        return f"{output_directory} files not found."
    
    calibration_signal_path = f"{output_directory}/paragraph_{max_length}/calibration_signals.json"
    lock_path = calibration_signal_path + ".lock"

    if not os.path.exists(calibration_signal_path):
        try:
            os.makedirs(lock_path, exist_ok=False)
            print(f"Job claimed calibration for {dataset}")
            num_signals = collect_calibration_signals(
                non_member_path=f"{output_directory}/non_members", 
                save_path=f"{output_directory}/paragraph_{max_length}", 
                token_level=max_length, model=model, tokenizer=tokenizer, device=device
            )
            os.rmdir(lock_path)
        except FileExistsError:
            print("Waiting for another job to finish calibration...")
            while not os.path.exists(calibration_signal_path):
                time.sleep(10)
            time.sleep(5)

    # 2. Load the signals (Now guaranteed to exist and be complete)
    with open(calibration_signal_path, "r") as f:
        calibration_signal = json.load(f)

    # 3. Calculate num_signals for indexing
    non_member_dataset = load_from_disk(f"{output_directory}/non_members")
    num_signals = min(int(len(non_member_dataset) * 0.5), 1000)
    
    rel_attacks = RelativeLikelihood(base_model=model, base_tokenizer=tokenizer, device=device)
    
    freq_dict_path="/lustre/selvaah3/projects/Masterthesis/GPTNeoXTokenizerFast_realnewslike_freq_dist.pkl"
    with open(freq_dict_path, "rb") as f:
        freq_dict = np.array(pickle.load(f), dtype=np.float32)
        
    dcpdd = DCPDD(freq_dict, device=device)
    rmia_attack = OfflineRobustMIA(target_model=model, target_tokenizer=tokenizer, reference_model=reference_model, reference_tokenizer=reference_tokenizer, a = 1, device=device)
    noisyneighbour_attack = NoisyNeighbour(model=model, sigma=10**(-1), device=device, batch_size = get_mapped_value(max_length)) # Depending on the max length, we choose the optimal batch size. 512 => 12, 1024 => 6, 2048 => 3
    tag_tab = TagTab(target_model=model, target_tokenizer=tokenizer, k = 5, device=device, nlp = spacy.load("en_core_web_sm"), min_size=3 if max_length == 43 else 7) 
    cimia_attack = CIMIA(target_model=model, target_tokenizer=tokenizer, device=device, max_len=max_length, calibration_signal = calibration_signal)
    
    completed_configs = set()
    # if os.path.exists("output_mia/completed_log.txt"):
    #     with open("output_mia/completed_log.txt", "r") as f:
    #         completed_configs = set(line.strip() for line in f)

    dataset_name = f"scaling_mia_the_pile_00_{dataset}"

    try:
        members = load_from_disk(dataset_path=f"{output_directory}/members")
        non_members = load_from_disk(dataset_path=f"{output_directory}/non_members")
        
        rng = np.random.default_rng(seed=42) 

        # Negative / Non-Member Prefix
        rand_idx_non = rng.integers(low=0, high=len(non_members), size=1 if max_length in [43, 512, 1024] else 2) # 7 Shots mentioned in the ConRecall Paper by Wang et al., but we only use 1 shots depending on the scale, because we have only 2048 context window
        non_members_shots = non_members.select(rand_idx_non)["text"]
        global_non_member_prefix = safe_pre_encode_shots(text_list=non_members_shots, tokenizer=tokenizer, max_shot_len=min(max_length-1, 1023))

        # Member Prefix
        rand_idx_mem = rng.integers(low=0, high=len(members), size=1 if max_length in [43, 512, 1024] else 2) # 7 Shots mentioned in the ConRecall Paper by Wang et al., but we only use 1 or 2 shots depending on the scale, because we have only 2048 context window. Most optimal method int(MODEL_MAX_LENGTH//max_length)-1
        members_shots = members.select(rand_idx_mem)["text"]
        global_member_prefix = safe_pre_encode_shots(text_list=members_shots, tokenizer=tokenizer, max_shot_len=min(max_length-1, 1023))
        
        if miaset == "member":
            # config_signature = f"DONE: member_{dataset_name} | Token: {max_length}\n"
            # if config_signature in completed_configs:
            #     return f"Skipping {dataset_name} {max_length} (Already Logged as Done)"
            
            collection_members = {}
            data_points_members = []
            for idx in tqdm(range(dataset_range[0], dataset_range[1]), desc="Members"):
                text = members['text'][idx]
                all_chunks = create_chunks(text, tokenizer, max_length)
                collection_members[f'Document_{idx}'] = []
                for chunk in all_chunks:
                    # Super short chunks are too noisy, so we filter them out
                    if len(chunk) > 25:
                        res = inference(
                            text=chunk, 
                            model=model, 
                            tokenizer=tokenizer, 
                            negative_prefix=global_non_member_prefix,
                            member_prefix=global_member_prefix,
                            non_member_prefix=global_non_member_prefix,
                            device=device, 
                            rel_attacks=rel_attacks, 
                            dcpdd=dcpdd, 
                            offline_rmia=rmia_attack, 
                            noisy_attack=noisyneighbour_attack, 
                            tagtab_attack=tag_tab, 
                            cimia_attack=cimia_attack
                        )
                        data_points_members.append({'pred': res, 'label': 1}) 
                        collection_members[f'Document_{idx}'].append(res)
                    
                with open((f"{output_directory}/paragraph_{max_length}/mia_members_{dataset_range[0]}_{dataset_range[1]}.jsonl"), "a") as f:
                    for dp in data_points_members:
                        f.write(json.dumps(dp, cls=TensorEncoder) + "\n")
 
                data_points_members = []
            
                with open(f"{output_directory}/document_{max_length}/mia_members_{dataset_range[0]}_{dataset_range[1]}.jsonl", "a") as f:
                    for doc_id, scores in collection_members.items():
                        # Save the ID and the LIST of scores
                        entry = {'id': doc_id, 'pred': scores, 'label': 1} 
                        f.write(json.dumps(entry, cls=TensorEncoder) + "\n")
                
                collection_members = {}
                
            # completion_log_path = "output_mia/completed_log.txt"
            # with open(completion_log_path, "a") as log_file:
            #     log_file.write(config_signature)

        if miaset == "nonmember":
            # config_signature = f"DONE: non_member_{dataset_name} | Token: {max_length}"
            # if config_signature in completed_configs:
            #     return f"Skipping {dataset_name} {max_length} (Already Logged as Done)"
            
            collection_nonmembers = {}
            data_points_nonmembers = []
            # We shift the index by the number of calibration signals we need for CIMIR, such that we prevent Data Leakage !
            for idx in tqdm(range(num_signals+dataset_range[0], min(dataset_range[1]+num_signals, len(non_members['text']))), desc="Non-Members"):
                text = non_members['text'][idx]
                all_chunks = create_chunks(text, tokenizer, max_length)
                
                collection_nonmembers[f'Document_{idx}'] = []
                for chunk in all_chunks:
                    # Super short chunks are too noisy, so we filter them out
                    if len(chunk) > 25:
                        res = inference(
                            text=chunk, 
                            model=model, 
                            tokenizer=tokenizer, 
                            negative_prefix=global_non_member_prefix,
                            member_prefix=global_member_prefix,
                            non_member_prefix=global_non_member_prefix,
                            device=device, 
                            rel_attacks=rel_attacks, 
                            dcpdd=dcpdd, 
                            offline_rmia=rmia_attack, 
                            noisy_attack=noisyneighbour_attack, 
                            tagtab_attack=tag_tab, 
                            cimia_attack=cimia_attack
                        )
                        data_points_nonmembers.append({'pred': res, 'label': 0})
                        collection_nonmembers[f'Document_{idx}'].append(res)
                    
                with open((f"{output_directory}/paragraph_{max_length}/mia_nonmembers_{num_signals+dataset_range[0]}_{min(dataset_range[1]+num_signals, len(non_members['text']))}.jsonl"), "a") as f:
                    for dp in data_points_nonmembers:
                        f.write(json.dumps(dp, cls=TensorEncoder) + "\n")
            
                data_points_nonmembers = []
                
                with open(f"{output_directory}/document_{max_length}/mia_nonmembers_{num_signals+dataset_range[0]}_{min(dataset_range[1]+num_signals, len(non_members['text']))}.jsonl", "a") as f:
                    for doc_id, scores in collection_nonmembers.items():
                        # Save the ID and the LIST of scores
                        entry = {'id': doc_id, 'pred': scores, 'label': 0} 
                        f.write(json.dumps(entry, cls=TensorEncoder) + "\n")
                
                collection_nonmembers = {}
                    
            # completion_log_path = "output_mia/completed_log.txt"
            # with open(completion_log_path, "a") as log_file:
            #     log_file.write(config_signature + "\n")
            
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with custom config")
    parser.add_argument("--pythia_model", type=str, default="pythia-2.8b", help="Model name")
    parser.add_argument("--miaset", type=str, default="nonmember", help="Dataset split")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--dataset", type=str, default="YoutubeSubtitles", help="dataset")
    parser.add_argument("--range", nargs=2, type=int, default=[0,0], help="List of two integers for dataset range")
    args = parser.parse_args()

    main(
        pythia_model=args.pythia_model, 
        max_length=args.max_length, 
        miaset=args.miaset,
        dataset=args.dataset,
        dataset_range=args.range,
    )