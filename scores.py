import torch
import numpy as np
import zlib
import math
import random 
import torch.nn.functional as F
import os
import numpy as np
from heapq import nlargest
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaForMaskedLM, RobertaTokenizer
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

    if len(topk) == 0:
        return 0.0
    
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

    if len(topk) == 0:
        return 0.0
    
    return np.mean(topk).item()
    
def zlib_entropy(sentence):
    """
    the ratio of the target loss and the zlib compression score of the target
    """
    return len(zlib.compress(bytes(sentence, 'utf-8')))

def ranks(logits, input_ids):
    """
    the average rank of the predicted token at each step
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    
    # Calculate ranks on aligned tensors
    matches = (shift_logits.argsort(-1, descending=True) == shift_labels.unsqueeze(-1)).nonzero()
    
    # Extract rank indices
    ranks_indices = matches[:, -1]
    
    # Convert to 1-based ranking and float
    ranks_float = ranks_indices.float() + 1
    return ranks_float.mean().item()


######### ReCalll and ConRecall attacks #########
def get_conditional_ll(prefix_text: str, target_text: list, model, tokenizer, device):
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    prefix_encodings = tokenizer("".join(prefix_text), return_tensors="pt", truncation=True, max_length=MODEL_MAX_LENGTH)
    target_encodings = tokenizer(target_text, return_tensors="pt", truncation=True, max_length=MODEL_MAX_LENGTH)

    prefix_ids = prefix_encodings.input_ids.to(device)
    target_ids = target_encodings.input_ids.to(device)

    # Concat the encodings of the prefix with the target text
    concat_ids = torch.cat(
        (prefix_ids, target_ids), dim=1
    )

    if concat_ids.shape[1] > model.config.max_position_embeddings:
        excess = concat_ids.shape[1] - model.config.max_position_embeddings
        concat_ids = concat_ids[:, excess:]

        # Set labels of prefix to -100
        labels = concat_ids.clone()
        labels[:, : prefix_ids.size(1) - excess] = -100
    else:
        labels = concat_ids.clone()
        labels[:, : prefix_ids.size(1)] = -100

    with torch.no_grad():
        outputs = model(concat_ids, labels=labels)
    loss, _ = outputs[:2]
    return -loss.item()

def process_prefix(prefix: list, target_length: int, model, tokenizer) -> list:

    token_counts = [
        len(tokenizer.encode(shot, truncation=True, max_length=MODEL_MAX_LENGTH)) for shot in prefix
    ]

    target_token_count = target_length
    total_tokens = sum(token_counts) + target_token_count
    if total_tokens <= model.config.max_position_embeddings:
        return prefix
    
    # Determine the maximum number of shots that can fit within the max_length
    max_shots = 0
    cumulative_tokens = target_token_count
    for count in token_counts:
        if cumulative_tokens + count <= model.config.max_position_embeddings:
            max_shots += 1
            cumulative_tokens += count
        else:
            break
    # Truncate the prefix to include only the maximum number of shots
    truncated_prefix = prefix[-max_shots:]
    return truncated_prefix

def recall(negative_prefix: list, target_text: str, model, tokenizer, device) -> float:
    
    if len(target_text) == 0:
        return 0.0
    
    with torch.no_grad():
        tokenized = tokenizer(
            target_text, truncation=True, return_tensors="pt"
        ).to(device)

        seq_len = tokenized.input_ids.shape[1]
        joint_prefix = process_prefix(prefix=negative_prefix, target_length=seq_len, model=model, tokenizer=tokenizer)
        # get unconditional log likelihood
        labels = tokenized.input_ids
        ll = -model(**tokenized, labels=labels).loss.item()

        # get conditional log likelihood with prefix
        ll_negative = get_conditional_ll(prefix_text=joint_prefix, target_text=target_text, model=model, tokenizer=tokenizer, device=device)

        return ll_negative / ll
    
def conrecall(target_text:str, member_prefix: list, non_member_prefix:list, model, tokenizer, device) -> dict:
    
    scores = {}
    if len(target_text) == 0:
        return {}
    
    with torch.no_grad():
        tokenized = tokenizer(
            target_text, truncation=True, return_tensors="pt"
        ).to(device)

        seq_len = tokenized.input_ids.shape[1]
        joint_member_prefix = process_prefix(prefix=member_prefix, target_length=seq_len, model=model, tokenizer=tokenizer)
        joint_non_member_prefix = process_prefix(prefix=non_member_prefix, target_length=seq_len, model=model, tokenizer=tokenizer)

        # get unconditional log likelihood
        labels = tokenized.input_ids
        ll = -model(**tokenized, labels=labels).loss.item()

        # get conditional log likelihood with prefix
        ll_member = get_conditional_ll(prefix=joint_member_prefix, target_text=target_text, model=model, tokenizer=tokenizer, device=device)
        ll_nonmember = get_conditional_ll(prefix=joint_non_member_prefix, target_text=target_text, model=model, tokenizer=tokenizer, device=device)

        for gamma in [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            scores[f"con-recall_{gamma}"] = (ll_nonmember - gamma * ll_member) / ll

        return scores

device = torch.device('mps:0')
attack_model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-1b",
        cache_dir="models/EleutherAI__pythia-1b",
        local_files_only=False,
        return_dict=True,
        device_map="auto",
        dtype=torch.float16
    )

attack_model = attack_model.to(device)

attack_tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-1b",
    cache_dir="models/EleutherAI__pythia-1b",
    local_files_only=False,
)

attack_tokenizer.pad_token = attack_tokenizer.eos_token

search_tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir="models/roberta_base")
search_model = RobertaForMaskedLM.from_pretrained('roberta-base', cache_dir="models/roberta_base")

print(search_model)

search_model = search_model.to(device)
token_dropout = torch.nn.Dropout(p=0.7)

def generate_neighbours_alt(text, num_word_changes=1):
    text_tokenized = search_tokenizer(text, padding = True, truncation = True, max_length = 2048, return_tensors='pt').input_ids.to(device)
    original_text = search_tokenizer.batch_decode(text_tokenized)[0]

    candidate_scores = dict()
    replacements = dict()

    for target_token_index in list(range(len(text_tokenized[0,:])))[1:]:

        target_token = text_tokenized[0,target_token_index]
        embeds = search_model.roberta.embeddings(text_tokenized)
            
        embeds = torch.cat((embeds[:,:target_token_index,:], token_dropout(embeds[:,target_token_index,:]).unsqueeze(dim=0), embeds[:,target_token_index+1:,:]), dim=1)
        
        token_probs = torch.softmax(search_model(inputs_embeds=embeds).logits, dim=2)

        original_prob = token_probs[0,target_token_index, target_token]

        top_probabilities, top_candidates = torch.topk(token_probs[:,target_token_index,:], 6, dim=1)

        for cand, prob in zip(top_candidates[0], top_probabilities[0]):
            if not cand == target_token:
                if original_prob.item() == 1:
                    print("probability is one!")
                    replacements[(target_token_index, cand)] = prob.item()/(1-0.9)
                else:
                    replacements[(target_token_index, cand)] = prob.item()/(1-original_prob.item())
    
    #highest_scored_texts = max(candidate_scores.iteritems(), key=operator.itemgetter(1))[:100]
    highest_scored_texts = nlargest(100, candidate_scores, key = candidate_scores.get)

    replacement_keys = nlargest(50, replacements, key=replacements.get)
    replacements_new = dict()
    for rk in replacement_keys:
        replacements_new[rk] = replacements[rk]
    
    replacements = replacements_new

    highest_scored = nlargest(100, replacements, key=replacements.get)

    texts = []
    for single in highest_scored:
        alt = text_tokenized
        target_token_index, cand = single
        alt = torch.cat((alt[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to(device), alt[:,target_token_index+1:]), dim=1)
        alt_text = search_tokenizer.batch_decode(alt)[0]
        texts.append((alt_text, replacements[single]))

    return texts

def get_logprob(text):
    text_tokenized = attack_tokenizer(text, padding = True, truncation = True, max_length = 2048, return_tensors='pt').input_ids.to(device)
    logprob = - attack_model(text_tokenized, labels=text_tokenized).loss.item()

    return logprob

def neighborhood_attack(text):
    attack_model.eval()
    search_model.eval()

    neighbor_loss = 0
    
    with torch.no_grad():
        start = time.time()
        neighbours = generate_neighbours_alt(text)
        end = time.time()

        print(len(neighbours))
        for n_tuple in neighbours:
            # n_tuple ist (Text, SwapScore)
            n_text = n_tuple[0].replace(" [SEP]", " ").replace("[CLS] ", " ")
            
            # Berechne den Loss für den Nachbarn
            neighbor_loss -= get_logprob(n_text) 

        tok_orig = search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to(device)
        orig_dec = search_tokenizer.batch_decode(tok_orig)[0].replace(" [SEP]", " ").replace("[CLS] ", " ")
        original_loss = -get_logprob(orig_dec)
        
        return original_loss-(neighbor_loss/len(neighbours))

print(neighborhood_attack("The capital of France is Paris"))    


def inference(sentence, model, tokenizer, negative_prefix, member_prefix, non_member_prefix, device):
    
    pred = {}
    fix_seed(0)

    loss, token_log_probs, logits, input_ids = raw_values(sentence=sentence, model=model, tokenizer=tokenizer)
    loss_lower, token_log_probs_lower, logits_lower, input_ids_lower = raw_values(sentence=sentence.lower(), model=model, tokenizer=tokenizer)
    
    pred = {
        'ppl': perplexity(loss=loss), 
        'ppl/lowercase_ppl': lowercase_perplexity(lowercase_ppl_val=perplexity(loss=loss_lower), original_ppl_val=perplexity(loss=loss)),
        'ppl/zlib': np.log(perplexity(loss=loss))/zlib_entropy(sentence=sentence),
        'ranks': ranks(logits=logits, input_ids=input_ids),
        'recall': recall(negative_prefix=negative_prefix, target_text=sentence, model=model, tokenizer=tokenizer, device=device),
        'conrecall': conrecall(target_text=sentence, member_prefix=member_prefix, non_member_prefix=non_member_prefix, model=model, tokenizer=tokenizer, device=device)
    }  

    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:  
        pred[f"Min_{ratio*100}% Prob"] = min_k(token_probs=token_log_probs, ratio=ratio)
        pred[f"Min_++{ratio*100}% Prob"] = min_k_plus_plus(logits=logits, input_ids=input_ids, ratio=ratio)

    # CLEANUP
    del loss, token_log_probs, logits, input_ids, loss_lower
    torch.cuda.empty_cache()
    return pred