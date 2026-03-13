from scores import inference, RelativeLikelihood, DCPDD, NoisyNeighbour, CAMIA, TagTab
from reference_scores import RefLossDiff, TokenLevelInfoRMIA, WBC
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
import time
from preprocess import create_chunks, safe_pre_encode_shots, TensorEncoder
from datasets import load_from_disk
from tqdm import tqdm
import traceback
import argparse
import spacy
import numpy as np
import pickle

os.environ["HF_HUB_OFFLINE"] = "1"
MIN_CHARS = 100
MODEL_MAX_LENGTH = 2048
MIA_SCORE_SAVING_DIR = "output_mia"
HF_DIR = "/lustre/selvaah3/hf_home"
os.environ['HF_HOME'] = HF_DIR

# Reference model configurations for TokenLevelInfoRMIA and WBC
# (model_size, step_suffix, directory_name)
REF_CONFIGS = [
    ("70m",  "step1",       "EleutherAI__pythia-70m_step1"),
    ("70m",  "step512",     "EleutherAI__pythia-70m_step512"),
    ("70m",  "step1000",    "EleutherAI__pythia-70m_step1000"),
    ("70m",  "step3000",    "EleutherAI__pythia-70m_step3000"),
    ("70m",  "step5000",    "EleutherAI__pythia-70m_step5000"),
    ("70m",  "step10000",   "EleutherAI__pythia-70m_step10000"),
    ("70m",  "step100000",  "EleutherAI__pythia-70m_step100000"),
    ("70m",  "final",       "EleutherAI__pythia-70m"),
    ("160m", "step1",       "EleutherAI__pythia-160m_step1"),
    ("160m", "step512",     "EleutherAI__pythia-160m_step512"),
    ("160m", "step1000",    "EleutherAI__pythia-160m_step1000"),
    ("160m", "step3000",    "EleutherAI__pythia-160m_step3000"),
    ("160m", "step5000",    "EleutherAI__pythia-160m_step5000"),
    ("160m", "step10000",   "EleutherAI__pythia-160m_step10000"),
    ("160m", "step100000",  "EleutherAI__pythia-160m_step100000"),
    ("160m", "final",       "EleutherAI__pythia-160m"),
    ("410m", "step1",       "EleutherAI__pythia-410m_step1"),
    ("410m", "step512",     "EleutherAI__pythia-410m_step512"),
    ("410m", "step1000",    "EleutherAI__pythia-410m_step1000"),
    ("410m", "step3000",    "EleutherAI__pythia-410m_step3000"),
    ("410m", "step5000",    "EleutherAI__pythia-410m_step5000"),
    ("410m", "step10000",   "EleutherAI__pythia-410m_step10000"),
    ("410m", "step100000",  "EleutherAI__pythia-410m_step100000"),
    ("410m", "final",       "EleutherAI__pythia-410m"),
    ("1b",   "step1",       "EleutherAI__pythia-1b_step1"),
    ("1b",   "step512",     "EleutherAI__pythia-1b_step512"),
    ("1b",   "step1000",    "EleutherAI__pythia-1b_step1000"),
    ("1b",   "step3000",    "EleutherAI__pythia-1b_step3000"),
    ("1b",   "step5000",    "EleutherAI__pythia-1b_step5000"),
    ("1b",   "step10000",   "EleutherAI__pythia-1b_step10000"),
    ("1b",   "step100000",  "EleutherAI__pythia-1b_step100000"),
    ("1b",   "final",       "EleutherAI__pythia-1b"),
]

# WBC window sizes to evaluate individually
WBC_WINDOW_SIZES = [1, 2, 4, 10, 20, 40]


def get_mapped_value(x):
    mapping = {
        43: 10,
        512: 5,
        1024: 2,
        2048: 1
    }
    return mapping.get(x, 1)


def load_ref_model(dir_name, device):
    """Load a reference model and tokenizer from local directory."""
    path = f"{HF_DIR}/models/{dir_name}"
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping")
        return None, None

    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, return_dict=True,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def main(pythia_model: str, max_length: int, miaset: str, dataset: str,
         dataset_range: list[int], member_shot_indices: list[int] = None,
         nonmember_shot_indices: list[int] = None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load target model (non-deduped, fully trained) from local directory
    model = AutoModelForCausalLM.from_pretrained(
        f"{HF_DIR}/models/EleutherAI__{pythia_model}",
        local_files_only=True, return_dict=True,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        f"{HF_DIR}/models/EleutherAI__{pythia_model}",
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    output_directory = f"{MIA_SCORE_SAVING_DIR}/{pythia_model}/{dataset}"
    if not os.path.exists(f"{output_directory}/members") or not os.path.exists(f"{output_directory}/non_members"):
        return f"{output_directory} files not found."

    # ─── NoisyNeighbour ───
    # batch_size must divide max_neighbours=30, so use gcd-friendly sizes
    # For 30 neighbours: batch_size must divide 10, 20, 30
    nn_batch_size = get_mapped_value(max_length)
    # Ensure batch_size divides 10 (smallest checkpoint)
    while 10 % nn_batch_size != 0:
        nn_batch_size -= 1
        if nn_batch_size < 1:
            nn_batch_size = 1
            break
    noisyneighbour_attack = NoisyNeighbour(
        model=model, device=device, batch_size=nn_batch_size
    )

    non_members = load_from_disk(dataset_path=f"{output_directory}/non_members")

    rel_attacks = RelativeLikelihood(base_model=model, base_tokenizer=tokenizer, device=device)

    # ─── DC-PDD (always with Laplace smoothing) ───
    freq_dict_path = "/lustre/selvaah3/projects/Masterthesis/GPTNeoXTokenizerFast_realnewslike_freq_dist.pkl"
    with open(freq_dict_path, "rb") as f:
        freq_dict = np.array(pickle.load(f), dtype=np.float32)

    dcpdd = DCPDD(
        freq_dict, device=device,
        a=0.01,  # default (predict_multi will use multiple a values)
        apply_smoothing=True
    )

    # ─── TagTab ───
    tag_tab = TagTab(
        target_model=model, target_tokenizer=tokenizer,
        top_k=10, nlp=spacy.load("en_core_web_sm"), device=device,
        entropy_map=None,
        min_sentence_len=3 if max_length == 43 else 7,
        max_sentence_len=40
    )

    # ─── CAMIA ───
    camia_attack = CAMIA(
        target_model=model, target_tokenizer=tokenizer,
        device=device, max_len=max_length, calibration_signal={}
    )

    dataset_name = f"scaling_mia_the_pile_00_{dataset}"

    try:
        members = load_from_disk(dataset_path=f"{output_directory}/members")

        # Use shot indices from slurm.py (guarantees all jobs use the same shots)
        if member_shot_indices is None or nonmember_shot_indices is None:
            raise ValueError("member_shot_indices and nonmember_shot_indices must be provided")

        rand_idx_mem = np.array(member_shot_indices)
        rand_idx_non = np.array(nonmember_shot_indices)
        excluded_member_indices = set(rand_idx_mem.tolist())
        excluded_nonmember_indices = set(rand_idx_non.tolist())

        # Calculate target text length reserve (95th percentile of dataset)
        sample_size = min(100, len(members), len(non_members['text']))
        sample_texts = list(members['text'][:sample_size]) + list(non_members['text'][:sample_size])
        sample_lengths = [len(tokenizer.encode(text, add_special_tokens=True)) for text in sample_texts]
        target_reserve = int(np.percentile(sample_lengths, 95))

        # Member Prefix
        members_shots = members.select(rand_idx_mem)["text"]
        global_member_prefix = safe_pre_encode_shots(
            text_list=members_shots, tokenizer=tokenizer,
            max_shot_len=min(max_length-1, 1023), reserve_for_target=target_reserve
        )

        # Non-member prefix
        non_members_shots = [non_members['text'][idx] for idx in rand_idx_non]
        global_non_member_prefix = safe_pre_encode_shots(
            text_list=non_members_shots, tokenizer=tokenizer,
            max_shot_len=min(max_length-1, 1023), reserve_for_target=target_reserve
        )

        # Determine dataset source and indices
        if miaset == "member":
            label_val = 1
            data_source = members
            idx_range = range(dataset_range[0], dataset_range[1])
            excluded = excluded_member_indices
            desc = "Members"
        else:
            label_val = 0
            data_source = non_members
            idx_range = range(dataset_range[0], min(dataset_range[1], len(non_members['text'])))
            excluded = excluded_nonmember_indices
            desc = "Non-Members"

        # ════════════════════════════════════════════════════════════════════
        # Output paths
        # ════════════════════════════════════════════════════════════════════
        if miaset == "member":
            para_path = f"{output_directory}/paragraph_{max_length}/mia_members_{dataset_range[0]}_{dataset_range[1]}.jsonl"
            doc_path = f"{output_directory}/document_{max_length}/mia_members_{dataset_range[0]}_{dataset_range[1]}.jsonl"
        else:
            end = min(dataset_range[1], len(non_members['text']))
            para_path = f"{output_directory}/paragraph_{max_length}/mia_nonmembers_{dataset_range[0]}_{end}.jsonl"
            doc_path = f"{output_directory}/document_{max_length}/mia_nonmembers_{dataset_range[0]}_{end}.jsonl"

        os.makedirs(os.path.dirname(para_path), exist_ok=True)
        os.makedirs(os.path.dirname(doc_path), exist_ok=True)

        # ════════════════════════════════════════════════════════════════════
        # Mini-batch processing: Pass 1 → Pass 2 → Save per batch
        # Avoids OOM from accumulating target_cache across all documents
        # ════════════════════════════════════════════════════════════════════
        MINI_BATCH_DOCS = 100  # documents per mini-batch
        total_chunks = 0
        total_docs = 0

        # Build list of valid indices (excluding shot indices)
        valid_indices = [idx for idx in idx_range if idx not in excluded]

        for mb_start in range(0, len(valid_indices), MINI_BATCH_DOCS):
            mb_indices = valid_indices[mb_start:mb_start + MINI_BATCH_DOCS]
            mb_num = mb_start // MINI_BATCH_DOCS + 1
            mb_total = (len(valid_indices) + MINI_BATCH_DOCS - 1) // MINI_BATCH_DOCS

            # ── Pass 1: Base attacks + build target cache ──
            chunk_texts = []
            chunk_results = []
            chunk_doc_ids = []
            target_cache = []

            for idx in tqdm(mb_indices, desc=f"{desc} batch {mb_num}/{mb_total}"):
                text = data_source['text'][idx]
                all_chunks = create_chunks(text, tokenizer, max_length)
                for chunk in all_chunks:
                    if len(chunk) <= 25:
                        continue
                    res, raw_data = inference(
                        text=chunk, model=model, tokenizer=tokenizer,
                        negative_prefix=global_non_member_prefix,
                        member_prefix=global_member_prefix,
                        non_member_prefix=global_non_member_prefix,
                        device=device, rel_attacks=rel_attacks, dcpdd=dcpdd,
                        noisy_attack=noisyneighbour_attack,
                        tagtab_attack=tag_tab, camia_attack=camia_attack,
                    )
                    if res is not None:
                        chunk_texts.append(chunk)
                        chunk_results.append(res)
                        chunk_doc_ids.append(idx)
                        target_cache.append({
                            'loss': raw_data['loss'].item(),
                            'logits': raw_data['logits'].cpu(),
                            'labels': raw_data['input_ids'].cpu(),
                            'per_token_losses': torch.nn.functional.cross_entropy(
                                raw_data['logits'].squeeze(0),
                                raw_data['input_ids'].squeeze(0),
                                reduction='none'
                            ).cpu().numpy(),
                        })

            if not chunk_texts:
                continue

            print(f"  Batch {mb_num}: Pass 1 done — {len(chunk_texts)} chunks from {len(set(chunk_doc_ids))} docs")

            # ── Pass 2: Reference-model attacks ──
            for size, step, dir_name in tqdm(REF_CONFIGS, desc=f"Ref models (batch {mb_num})"):
                ref_m, ref_t = load_ref_model(dir_name, device)
                if ref_m is None:
                    continue
                label = f"{size}_{step}"

                rld = RefLossDiff(
                    target_model=model, target_tokenizer=tokenizer,
                    ref_model=ref_m, ref_tokenizer=ref_t, device=device
                )
                tl_rmia = TokenLevelInfoRMIA(
                    target_model=model, target_tokenizer=tokenizer,
                    reference_models=[ref_m], reference_tokenizers=[ref_t],
                    temperature=2.0, aggregation='mean', device=device
                )
                wbc = WBC(
                    target_model=model, target_tokenizer=tokenizer,
                    ref_model=ref_m, ref_tokenizer=ref_t, device=device,
                    window_sizes=WBC_WINDOW_SIZES
                )

                for i, chunk in enumerate(chunk_texts):
                    res = chunk_results[i]
                    tc = target_cache[i]

                    tc_logits = tc['logits'].to(device) if tc['logits'] is not None else None
                    tc_labels = tc['labels'].to(device) if tc['labels'] is not None else None

                    try:
                        score = rld.predict(chunk, target_loss=tc['loss'])
                        res[f'ref_loss_diff_{label}'] = score if not np.isnan(score) else 0.0
                    except Exception:
                        res[f'ref_loss_diff_{label}'] = 0.0

                    try:
                        tl_scores = tl_rmia.predict_multi(
                            chunk,
                            temperatures=[0.5, 1.0, 2.0, 5.0],
                            aggregations=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0],
                            ref_labels=[label],
                            target_logits=tc_logits,
                            target_labels=tc_labels
                        )
                        res.update(tl_scores)
                    except Exception as e:
                        print(f"TokenLevelInfoRMIA {label} failed: {e}")

                    try:
                        wbc_scores = wbc.predict_per_window(
                            chunk, label=label, target_losses=tc['per_token_losses']
                        )
                        res.update(wbc_scores)
                    except Exception as e:
                        print(f"WBC {label} failed: {e}")

                    del tc_logits, tc_labels

                del rld, tl_rmia, wbc, ref_m, ref_t
                torch.cuda.empty_cache()

            print(f"  Batch {mb_num}: Pass 2 done — {len(REF_CONFIGS)} ref configs")

            # ── Save this mini-batch (append) ──
            collection = {}
            with open(para_path, "a") as f:
                for i, res in enumerate(chunk_results):
                    doc_id = f'Document_{chunk_doc_ids[i]}'
                    f.write(json.dumps({'pred': res, 'label': label_val}, cls=TensorEncoder) + "\n")
                    if doc_id not in collection:
                        collection[doc_id] = []
                    collection[doc_id].append(res)

            with open(doc_path, "a") as f:
                for doc_id, scores in collection.items():
                    f.write(json.dumps({'id': doc_id, 'pred': scores, 'label': label_val}, cls=TensorEncoder) + "\n")

            total_chunks += len(chunk_texts)
            total_docs += len(set(chunk_doc_ids))

            # Free memory before next mini-batch
            del chunk_texts, chunk_results, chunk_doc_ids, target_cache, collection
            import gc; gc.collect()

        # Free NoisyNeighbour
        del noisyneighbour_attack
        torch.cuda.empty_cache()

        print(f"All done: {total_chunks} chunks from {total_docs} documents")

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
    parser.add_argument("--member_shot_indices", type=str, required=True, help="Comma-separated member shot indices (from slurm.py)")
    parser.add_argument("--nonmember_shot_indices", type=str, required=True, help="Comma-separated non-member shot indices (from slurm.py)")
    args = parser.parse_args()

    main(
        pythia_model=args.pythia_model,
        max_length=args.max_length,
        miaset=args.miaset,
        dataset=args.dataset,
        dataset_range=args.range,
        member_shot_indices=[int(x) for x in args.member_shot_indices.split(",")],
        nonmember_shot_indices=[int(x) for x in args.nonmember_shot_indices.split(",")],
    )
