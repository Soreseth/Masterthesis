import torch
import numpy as np
import zlib
import math
import random 
from torch.nn.functional import F
def fix_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def raw_values(sentence, model, tokenizer):
    """
    Used to calculate the cross-entropy and probabilities of tokens for a given sentence and model 
    """
    # Ensure input is on the correct device
    encodings = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=2048)
    if model.device.type == "cuda":
        encodings = {k: v.cuda() for k, v in encodings.items()}
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings['input_ids'])

    # Average Cross-entropy loss over sentence
    loss = outputs.loss

    # raw, unnormalized scores for every word in its vocabulary for every position in the sentence. [number of positions x size of vocubulary] matrix
    logits = outputs.logits

    # turn raw scores at each position to log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    token_log_probs = []
    
    input_ids_processed = encodings['input_ids'][0][1:]

    # Get the probabilities of each word in the generated sentence by looking in the log_probs
    for i, token_id in enumerate(input_ids_processed):
        probability = log_probs[0, i, token_id].item()
        token_log_probs.append(probability)

    return loss, token_log_probs, logits, encodings['input_ids']

def perplexity(loss):
    return torch.exp(loss).item()

def lowercase_perplexity(lowercase_ppl_val, original_ppl_val):
    return -(np.log(lowercase_ppl_val)/np.log(original_ppl_val)).item()


def min_k(token_probs, ratio=0.05):
    sorted_prob = sorted(token_probs)
    k_length = max(1, int(len(sorted_prob)*ratio))
    topk = sorted_prob[:k_length]
    if sum(topk)==0 or k_length==0:
        return 0
    return np.mean(topk).item()

def min_k_plus_plus(logits, input_ids, ratio=0.05):

    input_ids = input_ids[0][1:].unsqueeze(-1)
    probs = F.softmax(logits[0, :-1], dim=-1)
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

    ## mink++
    mink_plus = (token_log_probs - mu) / sigma.sqrt()
    k_length = int(len(mink_plus) * ratio)
    topk = np.sort(mink_plus.cpu())[:k_length]
    return np.mean(topk)
    
def zlib_entropy(sentence):
    return len(zlib.compress(bytes(sentence, 'utf-8')))

def inference(sentence, model, tokenizer):
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