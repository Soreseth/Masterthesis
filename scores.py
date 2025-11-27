import torch
import numpy as np
import zlib
import math
import random 

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

    # BUG FIX: When return_dict=True, you cannot slice outputs[:2]. 
    # You must access attributes directly.
    loss = outputs.loss
    logits = outputs.logits

    # Calculate probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    
    # Process tokens (Slow serial loop - kept as requested)
    input_ids_processed = encodings['input_ids'][0][1:]

    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)

    log_likelihood = -loss.item()

    return loss, all_prob, log_likelihood

def conditional_log_likelihood(input_text, target_text, model, tokenizer, device):
    input_encodings = tokenizer(input_text, return_tensors="pt")
    target_encodings = tokenizer(target_text, return_tensors="pt")
    concat_ids = torch.cat((input_encodings.input_ids.to(device), target_encodings.input_ids.to(device)), dim=1)
    labels = concat_ids.clone()
    
    labels[:, : input_encodings.input_ids.size(1)] = -100
    with torch.no_grad():
        outputs = model(concat_ids, labels=labels)

    loss = outputs.loss
    logits = outputs.logits

    # Calculate probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    
    # Process tokens (Slow serial loop - kept as requested)
    input_ids_processed = encodings['input_ids'][0][1:]

    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)

    log_likelihood = -loss.item()

    return loss, all_prob, log_likelihood
    
    

def perplexity(loss):
    return torch.exp(loss).item()

def lowercase_perplexity(lowercase_ppl_val, original_ppl_val):
    return -(np.log(lowercase_ppl_val)/np.log(original_ppl_val)).item()

def k_min_probs(all_prob, ratio=0.05):
    sorted_prob = sorted(all_prob)
    num_values = max(1, int(len(sorted_prob)*ratio))
    k_min = sorted_prob[:num_values]
    if sum(k_min)==0 or num_values==0:
        return 0
    return -sum(k_min)/num_values

def zlib_entropy(sentence):
    return len(zlib.compress(bytes(sentence, 'utf-8')))

def inference(sentence, model, tokenizer):
    loss, all_prob, scalar_loss = raw_values(sentence=sentence, model=model, tokenizer=tokenizer)
    loss_lower, all_prob_lower, scalar_loss_lower = raw_values(sentence=sentence.lower(), model=model, tokenizer=tokenizer)
    
    pred = {
        'ppl': perplexity(loss=loss), 
        'ppl/lowercase_ppl': lowercase_perplexity(lowercase_ppl_val=perplexity(loss=loss_lower), original_ppl_val=perplexity(loss=loss)),
        'ppl/zlib': np.log(perplexity(loss=loss))/zlib_entropy(sentence=sentence)
    }

    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:  
        pred[f"Min_{ratio*100}% Prob"] = k_min_probs(all_prob=all_prob, ratio=ratio)

    return pred