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
from sklearn.metrics import auc

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

    token_probs = F.softmax(logits[0, :-1], dim=-1)
    # turn raw scores at each position to log probabilities
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)
    
    token_log_probs = []
    
    input_ids = encodings['input_ids'][0][1:].unsqueeze(-1)

    # Get the probabilities of each word in the generated sentence by looking in the log_probs
    token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
    return loss, token_probs, token_log_probs, logits, encodings['input_ids']


def perplexity(loss):
    return torch.exp(loss).item()

def lowercase_perplexity(lowercase_ppl_val, original_ppl_val):
    return -(np.log(lowercase_ppl_val)/np.log(original_ppl_val)).item()

def zlib_entropy(sentence):
    """
    the ratio of the target loss and the zlib compression score of the target
    """
    return len(zlib.compress(bytes(sentence, 'utf-8')))

class BaselineAttacks:
    def __init__(self, logits, input_ids, token_log_probs):
        self.logits = logits
        self.input_ids = input_ids
        self.token_log_probs = token_log_probs

    def min_k(self, ratio=0.05):
        """
        the average of log-likelihood of the k% tokens with lowest probabilities
        """
        k_length = max(1, int(len(self.token_log_probs)*ratio))
        sorted_prob = np.sort(self.token_log_probs.cpu())[:k_length]
        topk = sorted_prob[:k_length]

        # if len(topk) == 0:
        #     return 0.0
        
        return np.nanmean(topk).item()

    def min_k_plus_plus(self, ratio=0.05):
        """
        a standardized version of Min-K% over the model's vocabulary
        """
        # Number representation of sentence
        input_ids = self.input_ids[0][1:].unsqueeze(-1)

        # turn raw scores at each position to probabilities
        probs = F.softmax(self.logits[0, :-1], dim=-1)

        # log probabilities
        log_probs = F.log_softmax(self.logits[0, :-1], dim=-1)

        # token probabilities
        token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)

        # mean and variance
        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

        mink_plus = (token_log_probs - mu) / sigma.sqrt()
        k_length = int(len(mink_plus) * ratio)
        topk = np.sort(mink_plus.cpu())[:k_length]

        # if len(topk) == 0:
        #     return 0.0
        
        return np.nanmean(topk).item()

    def ranks(self):
        """
        the average rank of the predicted token at each step
        """
        shift_logits = self.logits[:, :-1, :]
        shift_labels = self.input_ids[:, 1:]
        
        # Calculate ranks on aligned tensors
        matches = (shift_logits.argsort(-1, descending=True) == shift_labels.unsqueeze(-1)).nonzero()
        
        # Extract rank indices
        ranks_indices = matches[:, -1]
        
        # Convert to 1-based ranking and float
        ranks_float = ranks_indices.float() + 1
        return ranks_float.mean().item()


# Adapted from https://github.com/ryuryukke/mint/blob/main/methods/recall/recall.py
class RelativeLikelihoodAttacks:
    def __init__(self, base_model_name: str, cache_dir: str, device):
        self.base_model_name = base_model_name
        self.cache_dir = cache_dir
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name, cache_dir=self.cache_dir, device_map="auto"
        )
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, cache_dir=self.cache_dir
        )
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id

        self.device = device
        self.base_model = self.base_model.to(self.device)

    def get_conditional_ll(self, prefix_text: str, target_text: list):
        
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            
        prefix_encodings = self.base_tokenizer("".join(prefix_text), return_tensors="pt", truncation=True, max_length=MODEL_MAX_LENGTH)
        target_encodings = self.base_tokenizer(target_text, return_tensors="pt", truncation=True, max_length=MODEL_MAX_LENGTH)

        prefix_ids = prefix_encodings.input_ids.to(self.device)
        target_ids = target_encodings.input_ids.to(self.device)

        # Concat the encodings of the prefix with the target text
        concat_ids = torch.cat(
            (prefix_ids, target_ids), dim=1
        )

        if concat_ids.shape[1] > self.base_model.config.max_position_embeddings:
            excess = concat_ids.shape[1] - self.base_model.config.max_position_embeddings
            concat_ids = concat_ids[:, excess:]

            # Set labels of prefix to -100
            labels = concat_ids.clone()
            labels[:, : prefix_ids.size(1) - excess] = -100
        else:
            labels = concat_ids.clone()
            labels[:, : prefix_ids.size(1)] = -100

        with torch.no_grad():
            outputs = self.base_model(concat_ids, labels=labels)
        loss, _ = outputs[:2]
        return -loss.item()

    def process_prefix(self, prefix: list, target_length: int) -> list:

        token_counts = [
            len(self.base_tokenizer.encode(shot, truncation=True, max_length=MODEL_MAX_LENGTH)) for shot in prefix
        ]

        target_token_count = target_length
        total_tokens = sum(token_counts) + target_token_count
        if total_tokens <= self.base_model.config.max_position_embeddings:
            return prefix
        
        # Determine the maximum number of shots that can fit within the max_length
        max_shots = 0
        cumulative_tokens = target_token_count
        for count in token_counts:
            if cumulative_tokens + count <= self.base_model.config.max_position_embeddings:
                max_shots += 1
                cumulative_tokens += count
            else:
                break
        # Truncate the prefix to include only the maximum number of shots
        truncated_prefix = prefix[-max_shots:]
        return truncated_prefix

    def recall(self, negative_prefix: list, target_text: str) -> float:
        
        if len(target_text) == 0:
            return 0.0
        
        with torch.no_grad():
            tokenized = self.base_tokenizer(
                target_text, truncation=True, return_tensors="pt"
            ).to(self.device)

            seq_len = tokenized.input_ids.shape[1]
            joint_prefix = self.process_prefix(prefix=negative_prefix, target_length=seq_len)
            # get unconditional log likelihood
            labels = tokenized.input_ids
            ll = -self.base_model(**tokenized, labels=labels).loss.item()

            # get conditional log likelihood with prefix
            ll_negative = self.get_conditional_ll(prefix_text=joint_prefix, target_text=target_text)

            return ll_negative / ll
        
    def conrecall(self, target_text:str, member_prefix: list, non_member_prefix:list) -> dict:
        
        scores = {}
        if len(target_text) == 0:
            return {}
        
        with torch.no_grad():
            tokenized = self.base_tokenizer(
                target_text, truncation=True, return_tensors="pt"
            ).to(self.device)

            seq_len = tokenized.input_ids.shape[1]
            joint_member_prefix = self.process_prefix(self, prefix=member_prefix, target_length=seq_len)
            joint_non_member_prefix = self.process_prefix(self, prefix=non_member_prefix, target_length=seq_len)

            # get unconditional log likelihood
            labels = tokenized.input_ids
            ll = -self.base_model(**tokenized, labels=labels).loss.item()

            # get conditional log likelihood with prefix
            ll_member = self.get_conditional_ll(self, prefix=joint_member_prefix, target_text=target_text)
            ll_nonmember = self.get_conditional_ll(self, prefix=joint_non_member_prefix, target_text=target_text)

            for gamma in [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                scores[f"con-recall_{gamma}"] = (ll_nonmember - gamma * ll_member) / ll

            return scores

# Mostly copied from https://github.com/mireshghallah/neighborhood-curvature-mia/tree/main
class NeighbourhoodComparisonAttack:
    def __init__(self, target_model_name: str, target_cache_dir: str, search_model_name: str, search_cache_dir: str, device):
        self.target_model_name = target_model_name
        self.target_cache_dir = target_cache_dir
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.target_model_name,
            cache_dir=self.target_cache_dir,
            local_files_only=False,
            return_dict=True,
            device_map="auto",
            dtype=torch.float16,
        ),
        
        self.target_tokenizer = AutoTokenizer.from_pretrained(
            self.target_model_name,
            cache_dir=self.target_cache_dir,
            local_files_only=False,
        ),

        self.target_tokenizer.pad_token = self.target_tokenizer.eos_token

        self.search_model_name = search_model_name
        self.search_cache_dir = search_cache_dir
        self.search_model = RobertaForMaskedLM.from_pretrained(self.search_model_name, cache_dir=self.search_cache_dir, local_files_only=False,device_map="auto", dtype=torch.float16)
        self.search_tokenizer = RobertaTokenizer.from_pretrained(self.search_model_name, cache_dir=self.search_cache_dir, local_files_only=False)
    
        self.device = device
        self.target_model = self.target_model.to(self.device)
        self.search_model = self.search_model.to(self.device)


    def generate_neighbours_alt(self, text, num_word_changes=1):
        text_tokenized = self.search_tokenizer(text, padding = True, truncation = True, max_length = MODEL_MAX_LENGTH, return_tensors='pt').input_ids.to(self.device)
        original_text = self.search_tokenizer.batch_decode(text_tokenized)[0]

        candidate_scores = dict()
        replacements = dict()
        token_dropout = torch.nn.Dropout(p=0.7)
        
        for target_token_index in list(range(len(text_tokenized[0,:])))[1:]:

            target_token = text_tokenized[0,target_token_index]
            embeds = self.search_model.roberta.embeddings(text_tokenized)
                
            embeds = torch.cat((embeds[:,:target_token_index,:], token_dropout(embeds[:,target_token_index,:]).unsqueeze(dim=0), embeds[:,target_token_index+1:,:]), dim=1)
            
            token_probs = torch.softmax(self.search_model(inputs_embeds=embeds).logits, dim=2)

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
            alt = torch.cat((alt[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to(self.device), alt[:,target_token_index+1:]), dim=1)
            alt_text = self.search_tokenizer.batch_decode(alt)[0]
            texts.append((alt_text, replacements[single]))

        return texts

    def get_logprob(self, text):
        text_tokenized = self.target_tokenizer(text, padding = True, truncation = True, max_length = MODEL_MAX_LENGTH, return_tensors='pt').input_ids.to(self.device)
        logprob = - self.target_model(text_tokenized, labels=text_tokenized).loss.item()

        return logprob

    def neighbourhood(self, text):
        self.target_model.eval()
        self.search_model.eval()

        neighbor_loss = 0
        
        with torch.no_grad():
            neighbours = self.generate_neighbours_alt(text)

            for n_tuple in neighbours:
                # n_tuple ist (Text, SwapScore)
                n_text = n_tuple[0].replace(" [SEP]", " ").replace("[CLS] ", " ")
                
                # Berechne den Loss für den Nachbarn
                neighbor_loss -= self.get_logprob(n_text) 

        tok_orig = self.search_tokenizer(text, padding = True, truncation = True, max_length = MODEL_MAX_LENGTH, return_tensors='pt').input_ids.to(self.device)
        orig_dec = self.search_tokenizer.batch_decode(tok_orig)[0].replace(" [SEP]", " ").replace("[CLS] ", " ")
        original_loss = -self.get_logprob(orig_dec)
        
        # There must be a threshold (gamma) here, but by aggregating the scores we "learn" the optimal thresholds anyways. 
        # The output would be usually a boolean and not float. Here is the code if we would use the threshold:
        # return (original_loss-(neighbor_loss/len(neighbours))) < gamma # Returns true if the difference would be smaller than gamma, false if bigger

        return original_loss-(neighbor_loss/len(neighbours))

class OfflineRobustMIA:
    def __init__(self, target_model_name: str, target_cache_dir: str, reference_model_names_list: list[str], reference_cache_dir_list: list[str], a: float, device):
        self.target_model_name = target_model_name
        self.target_cache_dir = target_cache_dir

        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.target_model_name,
            cache_dir=self.target_cache_dir,
            local_files_only=False,
            return_dict=True,
            device_map="auto",
            dtype=torch.float16,
        ),
        
        self.target_tokenizer = AutoTokenizer.from_pretrained(
            self.target_model_name,
            cache_dir=self.target_cache_dir,
            local_files_only=False,
        ),

        self.reference_model_names_list = reference_model_names_list
        self.reference_cache_dir_list = reference_cache_dir_list
        self.a = a
        self.device = device

    def robustmia(self, text):

        reference_likelihood = 0

        for i, model_names in enumerate(self.reference_model_names_list):
            reference_model = AutoModelForCausalLM.from_pretrained(
                model_names,
                cache_dir=self.reference_cache_dir_list[i],
                local_files_only=False,
                return_dict=True,
                device_map="auto",
                dtype=torch.float16,
            ),
            
            reference_tokenizer = AutoTokenizer.from_pretrained(
                model_names,
                cache_dir=self.reference_cache_dir_list[i],
                local_files_only=False,
            ),
            
            text_tokenized = reference_tokenizer(text, padding = True, truncation = True, max_length = MODEL_MAX_LENGTH, return_tensors='pt').input_ids.to(self.device)
            logprob = - reference_model(text_tokenized, labels=text_tokenized).loss.item()
            reference_likelihood += logprob

        average_pr_out = reference_likelihood/len(self.reference_model_names_list)

        # In OFFLINE mode: Estimate Pr(x)_IN with the Pr(x)_OUT. Pr(x)_OUT is the likelihood for a given sample x and model f_theta, 
        # where the model f_theta wasn't trained on the sample x. The parameter a must be optimized with the AUC score in mind. The usual 
        # procedure is copied from the original paper:
        # We choose two existing models and then, select one as the temporary target model and subject it to attacks from the other model 
        # using varying values of a. Finally, we select the one that yields the highest AUC as the optimal a. In the case of having only one 
        # reference model, we simulate an attack against the reference model and use the original target model as the reference model for 
        # the simulated attack to obtain the best a. Based on the result of our experiments, this optimal a remains roughly consistent 
        # across random selections of reference models. 

        pr_in = 1/2((1+self.a)*average_pr_out+(1-self.a))

        text_tokenized = self.target_tokenizer(text, padding = True, truncation = True, max_length = MODEL_MAX_LENGTH, return_tensors='pt').input_ids.to(self.device)
        target_logprob = - self.target_model(text_tokenized, labels=text_tokenized).loss.item()

        return target_logprob/pr_in
    
# Mostly copied from https://github.com/LIONS-EPFL/VL-MIA/blob/main/metric_util.py
class MaxRenyiAttack:
    def __init__(self, token_probs, token_log_probs, input_ids):
        self.token_probs =token_probs
        self.token_log_probs = token_log_probs
        self.input_ids_processed = input_ids[0][1:].unsqueeze(-1)

    def build_freq_dist():
        

    def calculateEntropy(self):
        entropies = []
        modified_entropies = []
        max_prob = []
        gap_prob = []
        renyi_05 = []
        renyi_2 = []
        modified_entropies_alpha05 = []
        modified_entropies_alpha2 = []
        epsilon = 1e-10

        input_ids_processed = self.input_ids_processed[1:]  # Exclude the first token for processing
        for i, token_id in enumerate(input_ids_processed):
            token_probs = self.token_probs[i, :]  # Get the probability distribution for the i-th token
            token_probs = token_probs.clone().detach().to(dtype=torch.float64)
            token_log_probs = self.token_log_probs[i, :]  # Log probabilities for entropy
            token_log_probs = token_log_probs.clone().detach().to(dtype=torch.float64)

            entropy = -(token_probs * token_log_probs).sum().item()  # Calculate entropy
            entropies.append(entropy)

            token_probs_safe = torch.clamp(token_probs, min=epsilon, max=1-epsilon)

            alpha = 0.5
            renyi_05_ = (1 / (1 - alpha)) * torch.log(torch.sum(torch.pow(token_probs_safe, alpha))).item()
            renyi_05.append(renyi_05_)
            alpha = 2
            renyi_2_ = (1 / (1 - alpha)) * torch.log(torch.sum(torch.pow(token_probs_safe, alpha))).item()
            renyi_2.append(renyi_2_)

            max_p = token_log_probs.max().item()
            second_p = token_log_probs[token_log_probs != token_log_probs.max()].max().item()
            gap_p = max_p - second_p
            gap_prob.append(gap_p)
            max_prob.append(max_p)

            # Modified entropy
            p_y = token_probs_safe[token_id].item()
            modified_entropy = -(1 - p_y) * torch.log(torch.tensor(p_y)) - (token_probs * torch.log(1 - token_probs_safe)).sum().item() + p_y * torch.log(torch.tensor(1 - p_y)).item()
            modified_entropies.append(modified_entropy)

            token_probs_remaining = torch.cat((token_probs_safe[:token_id], token_probs_safe[token_id+1:]))
            
            for alpha in [0.5,2]:
                entropy = - (1 / abs(1 - alpha)) * (
                    (1-p_y)* p_y**(abs(1-alpha))\
                        - (1-p_y)
                        + torch.sum(token_probs_remaining * torch.pow(1-token_probs_remaining, abs(1-alpha))) \
                        - torch.sum(token_probs_remaining)
                        ).item() 
                if alpha==0.5:
                    modified_entropies_alpha05.append(entropy)
                if alpha==2:
                    modified_entropies_alpha2.append(entropy)

        # loss = torch.tensor(loss)

        return {
            "entropies": entropies,
            "modified_entropies": modified_entropies,
            "max_prob": max_prob,
            "gap_prob": gap_prob,
            "renyi_05": renyi_05,
            "renyi_2": renyi_2,
            "mod_renyi_05" : modified_entropies_alpha05,
            "mod_renyi_2" : modified_entropies_alpha2
        }

class DCPDDAttack:
    def __init__(self, base_model_name: str, cache_dir: str):
        self.base_model_name = base_model_name
        self.cache_dir = cache_dir
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name, cache_dir=self.cache_dir, device_map="auto"
        )
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, cache_dir=self.cache_dir
        )
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_tokenizer.model_max_length = MODEL_MAX_LENGTH
                

        # using the defaults of the original implementation
        self.file_num = 15
        self.max_tok = 1024
        self.a = 1e-10

    def cal_ppl(self, text: str) -> tuple:
        first_device = next(self.base_model.parameters()).device
        input_ids = self.base_tokenizer.encode(text, truncation=True)
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        input_ids = input_ids.to(first_device)
        with torch.no_grad():
            output = self.base_model(input_ids, labels=input_ids)
        logit = output[1]
        # Apply softmax to the logits to get probabilities
        prob = torch.nn.functional.log_softmax(logit, dim=-1)[0][:-1]
        input_ids = input_ids[0][1:]
        probs = prob[torch.arange(len(prob)).to(first_device), input_ids].tolist()
        input_ids = input_ids.cpu().numpy()
        return probs, input_ids

    def detect(self, text: str, freq_dist: dict) -> float:
        if len(text) == 0:
            return 0.0
        # compute the probability distribution
        probs, input_ids = self.cal_ppl(text)
        # compute the prediction score by calibrating the probability with the token frequency distribution
        probs = np.exp(probs)
        indexes = []
        current_ids = []
        for i, input_id in enumerate(input_ids):
            if input_id not in current_ids:
                indexes.append(i)
                current_ids.append(input_id)

        x_pro = probs[indexes]
        x_fre = np.array(freq_dist)[input_ids[indexes].tolist()]
        # To avoid zero-division:
        epsilon = 1e-10
        x_fre = np.where(x_fre == 0, epsilon, x_fre)
        ce = x_pro * np.log(1 / x_fre)
        ce[ce > self.a] = self.a
        return -np.mean(ce)

def inference(sentence, model, tokenizer, negative_prefix, member_prefix, non_member_prefix, device):
    
    pred = {}
    fix_seed(42)

    loss, token_probs, token_log_probs, logits, input_ids = raw_values(sentence=sentence, model=model, tokenizer=tokenizer)
    loss_lower, token_log_probs_lower, logits_lower, input_ids_lower = raw_values(sentence=sentence.lower(), model=model, tokenizer=tokenizer)
    
    rel_attacks = RelativeLikelihoodAttacks(base_model_name="EleutherAI/pythia-1b",cache_dir="models/EleutherAI__pythia-1b", device=device)
    base_attacks = BaselineAttacks(logits=logits, input_ids=input_ids, token_log_probs=token_log_probs)
    neighbour_attacks = NeighbourhoodComparisonAttack(target_model_name="EleutherAI/pythia-1b", target_cache_dir="models/EleutherAI__pythia-1b", search_model_name="roberta-base", search_cache_dir="models/roberta_base", device=device)
    max_renyi = MaxRenyiAttack(token_probs=token_probs, token_log_probs=token_log_probs, input_ids=input_ids)
    dcpdd = DCPDDAttack(base_model_name="EleutherAI/pythia-1b",cache_dir="models/EleutherAI__pythia-1b")
    # Need to add RMIA !


    pred = {
        'max_renyi': max_renyi.calculateEntropy(),
        'ppl': perplexity(loss=loss), 
        'ppl/lowercase_ppl': lowercase_perplexity(lowercase_ppl_val=perplexity(loss=loss_lower), original_ppl_val=perplexity(loss=loss)),
        'ppl/zlib': np.log(perplexity(loss=loss))/zlib_entropy(sentence=sentence),
        'ranks': base_attacks.ranks(logits=logits, input_ids=input_ids),
        'neighbourhood': neighbour_attacks.neighbourhood(text=sentence),
        'recall': rel_attacks.recall(negative_prefix=negative_prefix, target_text=sentence, model=model, tokenizer=tokenizer, device=device),
        'conrecall': rel_attacks.conrecall(target_text=sentence, member_prefix=member_prefix, non_member_prefix=non_member_prefix, model=model, tokenizer=tokenizer, device=device),
        'dcpdd': dcpdd.detect(text=sentence, freq_dist=0.1) # Need to look into freq_dist
    }  

    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:  
        pred[f"Min_{ratio*100}% Prob"] = base_attacks.min_k(ratio=ratio)
        pred[f"Min_++{ratio*100}% Prob"] = base_attacks.min_k_plus_plus(ratio=ratio)

    # CLEANUP
    del loss, token_log_probs, logits, input_ids, loss_lower
    torch.cuda.empty_cache()
    return pred