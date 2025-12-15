from scores import inference, RelativeLikelihoodAttacks, BaselineAttacks, NeighbourhoodComparisonAttack, MaxRenyiAttack, DCPDDAttack, OfflineRobustMIA
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from preprocess import create_chunks, save_dataset, build_freq_dist, safe_truncate, TensorEncoder
from datasets import load_from_disk
from tqdm import tqdm 
import traceback
import argparse

os.environ["HF_HUB_OFFLINE"] = "1"
MIN_CHARS = 100
MIA_SCORE_SAVING_DIR = "output_mia"
HF_DIR = "/lustre/selvaah3/hf_home"



def main(pythia_model: str, max_length: int, miaset: str, datasets: list):
    BATCH_SIZE = 16 #max(1, (max_length/2048)-1)
    model = AutoModelForCausalLM.from_pretrained(
        f"{HF_DIR}/models/EleutherAI__{pythia_model}",
        local_files_only=True,
        return_dict=True,
        device_map="auto"
    )
    
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        f"{HF_DIR}/models/EleutherAI__{pythia_model}",
        local_files_only=True
    )
    
    reference_model = AutoModelForCausalLM.from_pretrained(
        "/lustre/selvaah3/hf_home/models/EleutherAI__pythia-1.4b",
        local_files_only=False,
        return_dict=True,
        device_map="auto"
    )
        
    reference_tokenizer = AutoTokenizer.from_pretrained(
        "/lustre/selvaah3/hf_home/models/EleutherAI__pythia-1.4b",
        local_files_only=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save_path="output_mia"
    # dataset_path="/lustre/selvaah3/hf_home/datasets/allenai__c4_en/en"
    # if not os.path.exists(f"big1_{save_path}/{type(tokenizer).__name__}_{dataset_path.split('/')[:-1]}_freq_dist.pkl"):
    #     build_freq_dist(save_path=save_path, base_tokenizer=tokenizer, dataset_path=dataset_path)
    # Change "/" to '/' inside the split()

    # save_dataset()
    dataset_path = "/lustre/selvaah3/hf_home/datasets/parameterlab"

    rel_attacks = RelativeLikelihoodAttacks(base_model=model, base_tokenizer=tokenizer, device=device)
    neighbour_attacks = NeighbourhoodComparisonAttack(target_model=model, target_tokenizer=tokenizer, search_model_name="/lustre/selvaah3/hf_home/models/FacebookAI__roberta-base", search_cache_dir="/lustre/selvaah3/hf_home/models/FacebookAI__roberta-base", device=device)
    dcpdd = DCPDDAttack(freq_dict_path="/lustre/selvaah3/projects/Masterthesis/output_mia/pythia-2.8b/GPTNeoXTokenizerFast_realnewslike_freq_dist.pkl")
    rmia_attack = OfflineRobustMIA(target_model=model, target_tokenizer=tokenizer, reference_model=reference_model, reference_tokenizer=reference_tokenizer, a = 1, device=device)

    completed_configs = set()
    if os.path.exists("output_mia/completed_log.txt"):
        with open("output_mia/completed_log.txt", "r") as f:
            completed_configs = set(line.strip() for line in f)

    # Document-Level in chunk sizes of 512, 1024, 2048
    for dataset_name in os.listdir(dataset_path):
        if dataset_name in os.listdir(dataset_path):
            # for max_length in [1024]: # [512, 1024, 2048]
            # Construct the log string to check

            try:
                if 'scaling_mia_the_pile_00_' not in dataset_name:
                    continue
                
                print(f"Processing {dataset_name} ...")
                # distinct check: is ANY item in blocklist inside dataset_name?
                if any(substring in dataset_name for substring in ['Pile-CC', 'OpenSubtitles', 'OpenWebText2', "StackExchange "]):
                    print(f"Skipping {dataset_name}")
                    continue
                    
                output_directory = f"{MIA_SCORE_SAVING_DIR}/{pythia_model}/{dataset_name.split('scaling_mia_the_pile_00_', 1)[1]}"
                
                # Check if paths exist before loading
                if not os.path.exists(f"{output_directory}/members") or not os.path.exists(f"{output_directory}/non_members"):
                    print(f"Skipping {output_directory}, files not found.")
                    continue

                members = load_from_disk(dataset_path=f"{output_directory}/members")
                non_members = load_from_disk(dataset_path=f"{output_directory}/non_members")

                if miaset == "member":
                    config_signature = f"DONE: member_{dataset_name} | Token: {max_length}"
                    if config_signature in completed_configs:
                        print(f"Skipping {dataset_name} {max_length} (Already Logged as Done)")
                        continue

                    # calculate mia scores for member set
                    data_points_members = []
                    for text in tqdm(members['text'], desc="Members"):
                        all_chunks = create_chunks(text, tokenizer, max_length)
                        doc_features = []
                        for i in range(0, len(all_chunks), BATCH_SIZE):
                            chunk = all_chunks[i: i+BATCH_SIZE]
                            batch_features = inference(chunk_batch=chunk, model=model, tokenizer=tokenizer, negative_prefix=safe_truncate(non_members["text"][:3], 40), member_prefix=safe_truncate(members["text"][:3], 40), non_member_prefix=safe_truncate(non_members["text"][:3], 40), rel_attacks=rel_attacks, dcpdd=dcpdd, device=device, neighbour_attacks=neighbour_attacks, offline_rmia=rmia_attack) 
                
                    # save mia scores for member set
                    with open((f"{output_directory}/token_{max_length}/mia_members.jsonl"), "w") as f:
                        for dp in data_points_members:
                            f.write(json.dumps(dp, cls=TensorEncoder) + "\n")

                    completion_log_path = "output_mia/completed_log.txt"
                    
                    with open(completion_log_path, "a") as log_file:
                        log_file.write(f"DONE: member_{dataset_name} | Token: {max_length}\n")
                    
                    torch.cuda.empty_cache()

                if miaset == "nonmember":
                    
                    config_signature = f"DONE: non_member_{dataset_name} | Token: {max_length}"
                    if config_signature in completed_configs:
                        print(f"Skipping {dataset_name} {max_length} (Already Logged as Done)")
                        continue
                    # calculate mia scores for non-member set
                    data_points_nonmembers = []
                    for text in tqdm(non_members['text'], desc="Non-Members"):
                        all_chunks = create_chunks(text, tokenizer, max_length)
                        doc_features = []
                        for i in range(0, len(all_chunks), BATCH_SIZE):
                            chunk = all_chunks[i: i+BATCH_SIZE]
                            batch_features = inference(chunk_batch=chunk, model=model, tokenizer=tokenizer, negative_prefix=safe_truncate(members["text"][:3], 40),member_prefix=safe_truncate(members["text"][:3], 40), non_member_prefix=safe_truncate(non_members["text"][:3], 40), rel_attacks=rel_attacks, offline_rmia=rmia_attack, neighbour_attacks=neighbour_attacks, dcpdd=dcpdd, device=device),
                            for res in batch_features:
                                doc_features.append({'pred': res, 'label': 0})
                        data_points_nonmembers.append(doc_features)

                    # save mia scores for non-member set
                    with open((f"{output_directory}/token_{max_length}/mia_nonmembers.jsonl"), "w") as f:
                        for dp in data_points_nonmembers:
                            f.write(json.dumps(dp, cls=TensorEncoder) + "\n")
                
                    completion_log_path = "output_mia/completed_log.txt"
                    
                    with open(completion_log_path, "a") as log_file:
                        log_file.write(f"DONE: non_member_{dataset_name} | Token: {max_length}\n")

                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                traceback.print_exc()
        else:
            print(f"{dataset_name} doesn't exist")
if __name__ == "__main__":
    # 1. Initialize the parser
    parser = argparse.ArgumentParser(description="Run experiment with custom config")

    # 2. Define the arguments you want to change from the bash script
    parser.add_argument("--pythia_model", type=str, default="pythia-2.8b", help="Model name")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--miaset", type=str, default="nonmember", help="Dataset split")
    parser.add_argument("--datasets", nargs='+', default=["YoutubeSubtitles"], help="List of datasets")
    # 3. Parse the arguments
    args = parser.parse_args()

    # 4. Pass the parsed arguments into your main function
    main(
        pythia_model=args.pythia_model, 
        max_length=args.max_length, 
        miaset=args.miaset,
        datasets=args.datasets
    )