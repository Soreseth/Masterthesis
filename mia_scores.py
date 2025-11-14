import torch
import numpy as np
from tqdm import tqdm
import zlib

def raw_values(sentence, model, tokenizer, gpu):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(gpu)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    # Grab cross-entropy loss and raw scores
    loss, logits = outputs[:2]

    # Calculate probabilites
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]

    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)

    return loss, all_prob, loss.item()

def perplexity(loss):
    """
    exp(loss)
    """    

    return torch.exp(loss).item()


def k_min_probs(all_prob, k=0.05):
    """
    Sum over k% lowest probability of tokens and then average them

    Outputs average log-probabilites
    """

    sorted_prob = sorted(all_prob)
    num_values = max(1, int(len(sorted_prob)*k))
    k_min = sorted_prob[:num_values]
    if sum(k_min)==0 or len(k_min)==0:
        return 0
    return sum(k_min)/len(k_min)

# TO FIX
def zlib_ratio(sentence):
    # Encode the string into bytes (using UTF-8)
    original_bytes = sentence.encode("utf-8")
    
    # Compress the bytes
    compressed_bytes = zlib.compress(original_bytes)
    
    return len(compressed_bytes) / len(original_bytes)

# TODO: Neighborhood based scores