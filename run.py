from scores import inference
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from preprocess import create_chunks, save_dataset
from datasets import load_from_disk
from tqdm import tqdm 


MIN_CHARS = 100
MIA_SCORE_SAVING_DIR = "/lustre/selvaah3/projects/myproj/output_mia/pythia-2.8b"
HF_DIR = "/lustre/selvaah3/hf_home"

if __name__ == "__main__":

    save_dataset()

    model = AutoModelForCausalLM.from_pretrained(
        f"{HF_DIR}/models/EleutherAI__pythia-2.8b",
        local_files_only=True,
        return_dict=True,
        device_map="auto",
        dtype=torch.float16
    )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        f"{HF_DIR}/models/EleutherAI__pythia-2.8b",
        local_files_only=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = "/lustre/selvaah3/hf_home/datasets/parameterlab"

    # Document-Level in chunk sizes of 512, 1024, 2048
    for dataset_name in os.listdir(dataset_path):
        for max_length in [512, 1024, 2048]:
            try:
                # String parsing fix to be safe
                if 'scaling_mia_the_pile_00_' not in dataset_name:
                    continue
                    
                output_directory = f"{MIA_SCORE_SAVING_DIR}/{dataset_name.split('scaling_mia_the_pile_00_', 1)[1]}"
                
                # Check if paths exist before loading
                if not os.path.exists(f"{output_directory}/members") or not os.path.exists(f"{output_directory}/non_members"):
                    print(f"Skipping {output_directory}, files not found.")
                    continue

                members = load_from_disk(dataset_path=f"{output_directory}/members")
                non_members = load_from_disk(dataset_path=f"{output_directory}/non_members")
                
                # calculate mia scores for member set
                data_points_members = []
                for text in tqdm(members['text'], desc="Members"):
                    chunks = create_chunks(text, tokenizer, max_length)
                    doc_features = []
                    for chunk in chunks:
                        mia_features = inference(sentence=chunk, model=model, tokenizer=tokenizer)
                        doc_features.append({'pred': mia_features, 'label': 1})
                    data_points_members.append(doc_features)
                
                # save mia scores for member set
                with open((f"{output_directory}/token_{max_length}/mia_members.jsonl"), "w") as f:
                    for dp in data_points_members:
                        f.write(json.dumps(dp) + "\n")

                # calculate mia scores for non-member set
                data_points_nonmembers = []
                for text in tqdm(non_members['text'], desc="Non-Members"):
                    chunks = create_chunks(text, tokenizer, max_length)
                    doc_features = []
                    for chunk in chunks:
                        if len(chunk) > MIN_CHARS:
                            mia_features = inference(sentence=chunk, model=model, tokenizer=tokenizer)
                            doc_features.append({'pred': mia_features, 'label': 0})
                    data_points_nonmembers.append(doc_features)

                # save mia scores for non-member set
                with open((f"{output_directory}/token_{max_length}/mia_nonmembers.jsonl"), "w") as f:
                    for dp in data_points_nonmembers:
                        f.write(json.dumps(dp) + "\n")

                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")

    for dataset_name in os.listdir(dataset_path):
        try:
            # String parsing fix to be safe
            if 'scaling_mia_the_pile_00_' not in dataset_name:
                continue
                
            output_directory = f"{MIA_SCORE_SAVING_DIR}/{dataset_name.split('scaling_mia_the_pile_00_', 1)[1]}"
            
            # Check if paths exist before loading
            if not os.path.exists(f"{output_directory}/members") or not os.path.exists(f"{output_directory}/non_members"):
                print(f"Skipping {output_directory}, files not found.")
                continue

            members = load_from_disk(dataset_path=f"{output_directory}/members")
            non_members = load_from_disk(dataset_path=f"{output_directory}/non_members")
            
            # calculate mia scores for member set
            data_points_members = []
            for text in tqdm(members['text'], desc="Members"):
                chunks = create_chunks(text, tokenizer, max_length)
                doc_features = []
                for chunk in chunks:
                    mia_features = inference(sentence=chunk, model=model, tokenizer=tokenizer)
                    doc_features.append({'pred': mia_features, 'label': 1})
                data_points_members.append(doc_features)
            
            # save mia scores for member set
            with open((f"{output_directory}/token_{max_length}/mia_members.jsonl"), "w") as f:
                for dp in data_points_members:
                    f.write(json.dumps(dp) + "\n")

            # calculate mia scores for non-member set
            data_points_nonmembers = []
            for text in tqdm(non_members['text'], desc="Non-Members"):
                chunks = create_chunks(text, tokenizer, max_length)
                doc_features = []
                for chunk in chunks:
                    if len(chunk) > MIN_CHARS:
                        mia_features = inference(sentence=chunk, model=model, tokenizer=tokenizer)
                        doc_features.append({'pred': mia_features, 'label': 0})
                data_points_nonmembers.append(doc_features)

            # save mia scores for non-member set
            with open((f"{output_directory}/token_{max_length}/mia_nonmembers.jsonl"), "w") as f:
                for dp in data_points_nonmembers:
                    f.write(json.dumps(dp) + "\n")

            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")