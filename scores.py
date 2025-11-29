import torch
import numpy as np
import zlib
import math
import random 
import torch.nn.functional as F
import os
import numpy as np
import tqdm
import transformers
from src.config import MODEL_MAX_LENGTH

MODEL_MAX_LENGTH = 2048

def fix_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def raw_values(sentence, model, tokenizer):
    """
    Used to calculate the cross-entropy and probabilities of tokens for a given sentence and model 
    """
    # Ensure input is on the correct device
    encodings = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=MODEL_MAX_LENGTH)
    # if model.device.type == "cuda": v.to(model.device)
    encodings = {k: v.to(model.device) for k, v in encodings.items()}
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings['input_ids'])

    # Average Cross-entropy loss over sentence. Taking the negative is the likelihood.
    loss = outputs.loss

    # raw, unnormalized scores for every word in its vocabulary for every position in the sentence. [number of positions x size of vocubulary] matrix
    logits = outputs.logits

    # turn raw scores at each position to log probabilities
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)
    
    token_log_probs = []
    
    input_ids = encodings['input_ids'][0][1:].unsqueeze(-1)

    # Get the probabilities of each word in the generated sentence by looking in the log_probs
    token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
    return loss, token_log_probs, logits, encodings['input_ids']

def perplexity(loss):
    return torch.exp(loss).item()

def lowercase_perplexity(lowercase_ppl_val, original_ppl_val):
    return -(np.log(lowercase_ppl_val)/np.log(original_ppl_val)).item()

def entropy(logits):
    logits = logits[:, :-1]
    neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
    return -neg_entropy.sum(-1).mean().item()

def min_k(token_probs, ratio=0.05):
    """
    the average of log-likelihood of the k% tokens with lowest probabilities
    """
    k_length = max(1, int(len(token_probs)*ratio))
    sorted_prob = np.sort(token_probs.cpu())[:k_length]
    topk = sorted_prob[:k_length]
    if sum(topk)==0 or k_length==0:
        return 0
    return np.mean(topk).item()

def min_k_plus_plus(logits, input_ids, ratio=0.05):
    """
    a standardized version of Min-K% over the model's vocabulary
    """
    # Number representation of sentence
    input_ids = input_ids[0][1:].unsqueeze(-1)

    # turn raw scores at each position to probabilities
    probs = F.softmax(logits[0, :-1], dim=-1)

    # log probabilities
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)

    # token probabilities
    token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)

    # mean and variance
    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

    mink_plus = (token_log_probs - mu) / sigma.sqrt()
    k_length = int(len(mink_plus) * ratio)
    topk = np.sort(mink_plus.cpu())[:k_length]
    return np.mean(topk)
    
def zlib_entropy(sentence):
    """
    the ratio of the target loss and the zlib compression score of the target
    """
    return len(zlib.compress(bytes(sentence, 'utf-8')))

def ranks(logits, input_ids):
    """
    the average rank of the predicted token at each step
    """
    labels = input_ids[:, 1:]
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    ranks, _ = matches[:, -1], matches[:, -2]
    ranks = ranks.float() + 1
    return ranks.float().mean().item()

def get_conditional_ll(prefix_text: str, target_text: list, model, tokenizer, device):
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    prefix_encodings = tokenizer(prefix_text, return_tensors="pt")
    target_encodings = tokenizer(target_text, return_tensors="pt")

    prefix_ids = prefix_encodings.input_ids.to(device)
    target_ids = target_encodings.input_ids.to(device)

    # Concat the encodings of the prefix with the target text
    concat_ids = torch.cat(
        (prefix_ids, target_ids), dim=1
    )

    if concat_ids.shape[1] > model.config.max_position_embeddings:
        excess = concat_ids.shape[1] - model.config.max_position_embeddings
        concat_ids = concat_ids[:, excess:]
        labels = concat_ids.clone()
        labels[:, : prefix_ids.size(1) - excess] = -100
    else:
        labels = concat_ids.clone()
        labels[:, : prefix_ids.size(1)] = -100

    concat_ids = concat_ids.contiguous()
    labels = labels.contiguous()

    with torch.no_grad():
        outputs = model(concat_ids, labels=labels)
    loss, _ = outputs[:2]
    return -loss.item()

def process_prefix(self, prefix: list, avg_length: int) -> list:
    max_length = (
        self.base_model.config.max_position_embeddings
        if "mpt" not in self.base_model_name
        else self.base_model.config.max_seq_len
    )
    token_counts = [
        len(self.base_tokenizer.encode(shot, truncation=True)) for shot in prefix
    ]
    target_token_count = avg_length
    total_tokens = sum(token_counts) + target_token_count
    if total_tokens <= max_length:
        return prefix
    # Determine the maximum number of shots that can fit within the max_length
    max_shots = 0
    cumulative_tokens = target_token_count
    for count in token_counts:
        if cumulative_tokens + count <= max_length:
            max_shots += 1
            cumulative_tokens += count
        else:
            break
    # Truncate the prefix to include only the maximum number of shots
    truncated_prefix = prefix[-max_shots:]
    return truncated_prefix

def detect(self, text: str, negative_prefix: list) -> float:
    if len(text) == 0:
        return 0.0
    with torch.no_grad():
        first_device = next(self.base_model.parameters()).device
        tokenized = self.base_tokenizer(
            text, truncation=True, return_tensors="pt"
        ).to(first_device)

        # get unconditional log likelihood
        labels = tokenized.input_ids
        ll = -self.base_model(**tokenized, labels=labels).loss.item()

        # get conditional log likelihood with prefix
        ll_negative = self.get_conditional_ll(text, negative_prefix)

        return ll_negative / ll
    

def inference(sentence, model, tokenizer):
    fix_seed(0)
    loss, token_log_probs, logits, input_ids = raw_values(sentence=sentence, model=model, tokenizer=tokenizer)
    loss_lower, token_log_probs_lower, logits_lower, input_ids_lower = raw_values(sentence=sentence.lower(), model=model, tokenizer=tokenizer)
    
    pred = {
        'ppl': perplexity(loss=loss), 
        'ppl/lowercase_ppl': lowercase_perplexity(lowercase_ppl_val=perplexity(loss=loss_lower), original_ppl_val=perplexity(loss=loss)),
        'ppl/zlib': np.log(perplexity(loss=loss))/zlib_entropy(sentence=sentence),
    }

    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:  
        pred[f"Min_{ratio*100}% Prob"] = min_k(token_probs=token_log_probs, ratio=ratio)
        pred[f"Min_++{ratio*100}% Prob"] = min_k_plus_plus(logits=logits, input_ids=input_ids, ratio=ratio)

    return pred