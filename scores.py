import torch
import numpy as np
import zlib
import math
import random 
import torch.nn.functional as F
import os
import numpy as np
from numpy import nanmean
from heapq import nlargest
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaForMaskedLM, RobertaTokenizer
from src.config import MODEL_MAX_LENGTH
from sklearn.metrics import auc
import pickle
import traceback
# os.environ["HF_HUB_OFFLINE"] = "1"
MODEL_MAX_LENGTH = 2048

def fix_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def raw_values(sentences, model, tokenizer, device):
    """
    Used to calculate the cross-entropy and probabilities of tokens for a given sentence and model 
    """
    encodings = tokenizer(
        sentences, 
        return_tensors='pt', 
        truncation=True, 
        max_length=MODEL_MAX_LENGTH, 
        padding=True
    ).to(device)

    vocab_size = model.config.vocab_size
    
    # Check if max ID is out of bounds
    if encodings['input_ids'].max() >= vocab_size:
        
        # 1. Define the Safe Token (UNK is best, fallback to EOS)
        safe_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id
        
        # 2. Create a mask of all invalid positions (True where ID is too big)
        invalid_mask = encodings['input_ids'] >= vocab_size
        encodings['input_ids'][invalid_mask] = safe_token_id

    with torch.no_grad():
        outputs = model(**encodings)

    results = []

    # Average Cross-entropy loss over sentence. Taking the negative is the likelihood.
    loss = outputs.loss

    # raw, unnormalized scores for every word in its vocabulary for every position in the sentence. [number of positions x size of vocubulary] matrix
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = encodings.input_ids[..., 1:].contiguous()
    shift_attention = encodings.attention_mask[..., 1:].contiguous()

    batch_size = shift_labels.size(0)
    for i in range(batch_size):
        valid_indices = shift_attention[i] == 1
    
        sample_ids = shift_labels[i][valid_indices]
        sample_logits = shift_logits[i][valid_indices]

        # turn raw scores at each position to log probabilities
        log_probs = F.log_softmax(sample_logits, dim=-1)

        probs = F.softmax(sample_logits, dim=-1)
    

        # Get the probabilities of each word in the generated sentence by looking in the log_probs
        token_log_probs = log_probs.gather(dim=-1, index=sample_ids.unsqueeze(-1)).squeeze(-1)
        token_probs = probs.gather(dim=-1, index=sample_ids.unsqueeze(-1)).squeeze(-1)

        loss = -token_log_probs.nanmean()
        results.append({
                "loss": loss,
                "token_probs": token_probs,
                "token_log_probs": token_log_probs,
                "logits": sample_logits.unsqueeze(0),
                "input_ids": sample_ids.unsqueeze(0).unsqueeze(0),
                "full_token_probs": probs,
                "full_log_probs": log_probs
            })
    
    del shift_logits
    del encodings
    del outputs
    torch.cuda.empty_cache()

    return results
def perplexity(loss):
    return torch.exp(loss).item()

def zlib_entropy(sentence):
    """
    the ratio of the target loss and the zlib compression score of the target
    """
    return len(zlib.compress(bytes(sentence, 'utf-8')))

class BaselineAttacks:
    def __init__(self, logits, input_ids, token_log_probs):
        self.logits = logits
        if input_ids.dim() == 3 and input_ids.shape[1] == 1:
            self.input_ids = input_ids.squeeze(1)
        else:
            self.input_ids = input_ids
        self.token_log_probs = token_log_probs

    def min_k(self, ratio=0.05):
        k_length = max(1, int(len(self.token_log_probs)*ratio))
        sorted_prob = np.sort(self.token_log_probs.cpu())[:k_length]
        topk = sorted_prob[:k_length]
        return nanmean(topk).item()

    def min_k_plus_plus(self, ratio=0.05):
        # Input IDs are already aligned to the logits
        # input_ids = self.input_ids[0].unsqueeze(-1) # [Seq_Len, 1]

        # Logits are already aligned
        probs = F.softmax(self.logits[0], dim=-1) # [Seq_Len, Vocab]
        log_probs = F.log_softmax(self.logits[0], dim=-1) # [Seq_Len, Vocab]

        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

        mink_plus = (self.token_log_probs - mu) / (sigma.sqrt() + 1e-10)
        k_length = max(1, int(len(mink_plus) * ratio))
        topk = np.sort(mink_plus.cpu())[:k_length]
        return nanmean(topk).item()

    def ranks(self):
        # logits[i] is the prediction for labels[i]
        
        logits = self.logits # [1, Seq_Len, Vocab]
        labels = self.input_ids # [1, Seq_Len]
        
        # Calculate ranks
        sorted_idxs = logits.argsort(-1, descending=True)
        
        # Find where the sorted indices match the actual labels
        ranks = (sorted_idxs == labels.unsqueeze(-1)).nonzero(as_tuple=True)[-1]
        
        # Convert to 1-based ranking and float
        ranks_float = ranks.float() + 1
        return ranks_float.nanmean().item()


# Adapted from https://github.com/ryuryukke/mint/blob/main/methods/recall/recall.py
class RelativeLikelihoodAttacks:
    def __init__(self, base_model, base_tokenizer, device):
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id

        self.device = device
        # self.base_model = self.base_model.to(self.device)

    def _sanitize(self, input_ids):
        vocab_size = self.base_model.config.vocab_size
        if input_ids.max() >= vocab_size:
            safe_id = self.base_tokenizer.unk_token_id if self.base_tokenizer.unk_token_id is not None else self.base_tokenizer.eos_token_id
            input_ids[input_ids >= vocab_size] = safe_id
        return input_ids
    
    def _get_batch_loss(self, prefixes, targets):
        """
        Calculates loss for a batch of (Prefix + Target) pairs.
        """
        self.base_tokenizer.padding_side = "right"
        
        # 1. Tokenize Prefix and Target separately to manage masking
        prefix_enc = self.base_tokenizer(prefixes, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        target_enc = self.base_tokenizer(targets, return_tensors='pt', padding=True, truncation=True, max_length=1024, add_special_tokens=False)
        
        # 2. Sanitize Inputs (Crucial for stability)
        prefix_ids = self._sanitize(prefix_enc['input_ids']).to(self.device)
        target_ids = self._sanitize(target_enc['input_ids']).to(self.device)
        
        # 3. Concatenate
        input_ids = torch.cat((prefix_ids, target_ids), dim=1)
        
        # 4. Build Masks
        prefix_mask = prefix_enc.attention_mask.to(self.device)
        target_mask = target_enc.attention_mask.to(self.device)
        attention_mask = torch.cat((prefix_mask, target_mask), dim=1)
        
        # 5. Create Labels (Mask out prefix and padding)
        labels = input_ids.clone()
        # Mask the prefix part
        labels[:, :prefix_ids.shape[1]] = -100 
        # Mask the padding
        labels[attention_mask == 0] = -100

        # 6. Forward Pass (Heavy GPU work happens here)
        with torch.no_grad():
            outputs = self.base_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # 7. Calculate Loss per Sample
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_losses = token_losses.view(shift_labels.size())
        
        # Sum valid losses and divide by number of non-masked tokens
        valid_tokens = (shift_labels != -100).sum(dim=1)
        # Avoid div by zero
        valid_tokens[valid_tokens == 0] = 1
        sample_losses = token_losses.sum(dim=1) / valid_tokens
        
        return sample_losses

    # def process_prefix(self, prefix: list, target_length: int) -> list:

    #     token_counts = [
    #         len(self.base_tokenizer.encode(shot, truncation=True, max_length=MODEL_MAX_LENGTH)) for shot in prefix
    #     ]

    #     target_token_count = target_length
    #     total_tokens = sum(token_counts) + target_token_count
    #     if total_tokens <= self.base_model.config.max_position_embeddings:
    #         return prefix
        
    #     # Determine the maximum number of shots that can fit within the max_length
    #     max_shots = 0
    #     cumulative_tokens = target_token_count
    #     for count in token_counts:
    #         if cumulative_tokens + count <= self.base_model.config.max_position_embeddings:
    #             max_shots += 1
    #             cumulative_tokens += count
    #         else:
    #             break
    #     # Truncate the prefix to include only the maximum number of shots
    #     truncated_prefix = prefix[-max_shots:]
    #     return truncated_prefix

    # def recall(self, negative_prefix: list, target_text: str) -> float:
        
    #     if len(target_text) == 0:
    #         return 0.0
        
    #     with torch.no_grad():
    #         tokenized = self.base_tokenizer(
    #             target_text, truncation=True, return_tensors="pt", max_length=MODEL_MAX_LENGTH
    #         ).to(self.device)

    #         vocab_size = self.base_model.config.vocab_size
                
    #         # Check if max ID is out of bounds
    #         if tokenized['input_ids'].max() >= vocab_size:
                
    #             # 1. Define the Safe Token (UNK is best, fallback to EOS)
    #             safe_token_id = tokenized.unk_token_id if tokenized.unk_token_id is not None else tokenizer.eos_token_id
                
    #             # 2. Create a mask of all invalid positions (True where ID is too big)
    #             invalid_mask = tokenized['input_ids'] >= vocab_size
    #             tokenized['input_ids'][invalid_mask] = safe_token_id

    #         seq_len = tokenized.input_ids.shape[1]
    #         joint_prefix = self.process_prefix(prefix=negative_prefix, target_length=seq_len)
    #         # get unconditional log likelihood
    #         labels = tokenized.input_ids
    #         ll = -self.base_model(**tokenized, labels=labels).loss.item()

    #         # get conditional log likelihood with prefix
    #         ll_negative = self.get_conditional_ll(prefix_text=joint_prefix, target_text=target_text)

    #         return ll_negative / ll
        
    # def conrecall(self, target_text:str, member_prefix: list, non_member_prefix:list) -> dict:
        
    #     scores = {}
    #     if len(target_text) == 0:
    #         return {}
        
    #     with torch.no_grad():
    #         tokenized = self.base_tokenizer(
    #             target_text, truncation=True, return_tensors="pt"
    #         ).to(self.device)

    #         seq_len = tokenized.input_ids.shape[1]
    #         joint_member_prefix = self.process_prefix(prefix=member_prefix, target_length=seq_len)
    #         joint_non_member_prefix = self.process_prefix(prefix=non_member_prefix, target_length=seq_len)

    #         # get unconditional log likelihood
    #         labels = tokenized.input_ids
    #         ll = -self.base_model(**tokenized, labels=labels).loss.item()

    #         # get conditional log likelihood with prefix
    #         ll_member = self.get_conditional_ll(prefix_text=joint_member_prefix, target_text=target_text)
    #         ll_nonmember = self.get_conditional_ll(prefix_text=joint_non_member_prefix, target_text=target_text)

    #         for gamma in [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    #             scores[f"con-recall_{gamma}"] = (ll_nonmember - gamma * ll_member) / ll

    #         return scores
    def batch_recall(self, chunk_batch, base_losses, negative_prefix_str):
        # Repeat the prefix for the whole batch
        prefixes = [negative_prefix_str] * len(chunk_batch)
        cond_losses = self._get_batch_loss(prefixes, chunk_batch)
        
        results = []
        for i in range(len(chunk_batch)):
            ll_negative = -cond_losses[i].item()
            ll_base = -base_losses[i]
            
            if ll_base == 0: 
                results.append(0.0)
            else:
                results.append(ll_negative / ll_base)
        return results

    def batch_conrecall(self, chunk_batch, base_losses, member_prefix_str, non_member_prefix_str):
        # 1. Batch Member Check
        prefixes_mem = [member_prefix_str] * len(chunk_batch)
        cond_loss_mem = self._get_batch_loss(prefixes_mem, chunk_batch)
        
        # 2. Batch Non-Member Check
        prefixes_non = [non_member_prefix_str] * len(chunk_batch)
        cond_loss_non = self._get_batch_loss(prefixes_non, chunk_batch)
        
        results = []
        for i in range(len(chunk_batch)):
            ll = -base_losses[i]
            ll_member = -cond_loss_mem[i].item()
            ll_nonmember = -cond_loss_non[i].item()
            
            scores = {}
            if ll == 0:
                results.append({})
                continue

            for gamma in [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                scores[f"con-recall_{gamma}"] = (ll_nonmember - gamma * ll_member) / ll
            results.append(scores)
        return results
    
# Mostly copied from https://github.com/mireshghallah/neighborhood-curvature-mia/tree/main
class NeighbourhoodComparisonAttack:
    def __init__(self, target_model, target_tokenizer, search_model_name: str, search_cache_dir: str, device):
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.target_tokenizer.pad_token = self.target_tokenizer.eos_token

        self.search_model_name = search_model_name
        self.search_cache_dir = search_cache_dir
        self.search_model = RobertaForMaskedLM.from_pretrained(
            self.search_model_name, 
            cache_dir=self.search_cache_dir, 
            local_files_only=True, 
            device_map="auto"
        )
        self.search_tokenizer = RobertaTokenizer.from_pretrained(self.search_model_name, cache_dir=self.search_cache_dir, local_files_only=True)
    
        self.device = device
        # self.target_model = self.target_model.to(self.device)
        # self.search_model = self.search_model.to(self.device)


    def generate_neighbours_alt(self, text, num_word_changes=1):
        text_tokenized = self.search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to(self.device)
        # original_text = self.search_tokenizer.batch_decode(text_tokenized)[0]

        vocab_size = self.base_model.config.vocab_size
            
        # Check if max ID is out of bounds
        if text_tokenized['input_ids'].max() >= vocab_size:
            
            # 1. Define the Safe Token (UNK is best, fallback to EOS)
            safe_token_id = text_tokenized.unk_token_id if text_tokenized.unk_token_id is not None else self.search_tokenizer.eos_token_id
            
            # 2. Create a mask of all invalid positions (True where ID is too big)
            invalid_mask = text_tokenized['input_ids'] >= vocab_size
            text_tokenized['input_ids'][invalid_mask] = safe_token_id

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
                        replacements[(target_token_index, cand)] = prob.item()/(1-0.9)
                    else:
                        replacements[(target_token_index, cand)] = prob.item()/(1-original_prob.item())
        
        #highest_scored_texts = max(candidate_scores.iteritems(), key=operator.itemgetter(1))[:100]
        # highest_scored_texts = nlargest(100, candidate_scores, key = candidate_scores.get)

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
                
                # Calculate loss for neighbours
                neighbor_loss -= self.get_logprob(n_text) 

        tok_orig = self.search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to(self.device)
        orig_dec = self.search_tokenizer.batch_decode(tok_orig)[0].replace(" [SEP]", " ").replace("[CLS] ", " ")
        original_loss = -self.get_logprob(orig_dec)
        
        # There must be a threshold (gamma) here, but by aggregating the scores we "learn" the optimal thresholds anyways. 
        # The output would be usually a boolean and not float. Here is the code if we would use the threshold:
        # return (original_loss-(neighbor_loss/len(neighbours))) < gamma # Returns true if the difference would be smaller than gamma, false if bigger

        return original_loss-(neighbor_loss/len(neighbours))

class OfflineRobustMIA:
    def __init__(self, target_model, target_tokenizer, reference_model_names_list: list[str], reference_cache_dir_list: list[str], a: float, device):
        self.target_model = target_model
        
        self.target_tokenizer = target_tokenizer

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
                device_map="auto"
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

        pr_in = 0.5 * ((1+self.a)*average_pr_out+(1-self.a))

        text_tokenized = self.target_tokenizer(text, padding = True, truncation = True, max_length = MODEL_MAX_LENGTH, return_tensors='pt').input_ids.to(self.device)
        target_logprob = - self.target_model(text_tokenized, labels=text_tokenized).loss.item()

        return target_logprob/pr_in
    
# Mostly copied from https://github.com/LIONS-EPFL/VL-MIA/blob/main/metric_util.py
class MaxRenyiAttack:
    def __init__(self, token_probs, token_log_probs, input_ids):
        self.token_probs =token_probs
        self.token_log_probs = token_log_probs
        self.input_ids_processed = input_ids.squeeze().unsqueeze(-1)

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

        for i, token_id_tensor in enumerate(self.input_ids_processed):
            token_id = token_id_tensor.item() 
            token_probs = self.token_probs[i, :] 
            token_probs = token_probs.clone().detach().to(dtype=torch.float32)

            token_log_probs = self.token_log_probs[i, :]  # Log probabilities for entropy
            token_log_probs = token_log_probs.clone().detach().to(dtype=torch.float32)

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

            # Use the integer token_id for slicing
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
            "entropies": np.nanmean(entropies),
            "modified_entropies": np.nanmean(modified_entropies),
            "max_prob": np.nanmean(max_prob),
            "gap_prob": np.nanmean(gap_prob),
            "renyi_05": np.nanmean(renyi_05),
            "renyi_2": np.nanmean(renyi_2),
            "mod_renyi_05": np.nanmean(modified_entropies_alpha05),
            "mod_renyi_2": np.nanmean(modified_entropies_alpha2),
            
            # Optional: The negative score if you want to match the other class's style
            "entropy_score": -np.nanmean(entropies) 
        }

class DCPDDAttack:
    def __init__(self, freq_dict_path: str):
        self.freq_dict_path = freq_dict_path
        self.a = 1e-10
        # Load the frequency dictionary ONCE during init to save disk I/O speed
        with open(self.freq_dict_path, "rb") as f:
            self.freq_dist = np.array(pickle.load(f))

    def detect(self, token_probs, input_ids) -> float:
        if isinstance(token_probs, torch.Tensor):
            probs = token_probs.detach().cpu().numpy().flatten()
        else:
            probs = token_probs.flatten()

        if isinstance(input_ids, torch.Tensor):
            # Flatten to 1D array of integers
            input_ids_np = input_ids.detach().cpu().numpy().flatten()
        else:
            input_ids_np = input_ids.flatten()

        # Logic to find unique token indices
        indexes = []
        current_ids = set() # Use set for O(1) lookup speed
        for i, input_id in enumerate(input_ids_np):
            if input_id not in current_ids:
                indexes.append(i)
                current_ids.add(input_id)
        
        x_pro = probs[indexes]

        # Prevent token ID  larger than our frequency dictionary
        valid_ids = input_ids_np[indexes]
        max_freq_id = len(self.freq_dist) - 1
        valid_ids = np.clip(valid_ids, 0, max_freq_id)
        
        x_fre = self.freq_dist[valid_ids]
        
        # To avoid zero-division:
        epsilon = 1e-10
        x_fre = np.where(x_fre == 0, epsilon, x_fre)
        
        ce = x_pro * np.log(1 / x_fre)
        ce[ce > self.a] = self.a
        return -float(np.nanmean(ce))

def inference(chunk_batch, model, tokenizer, negative_prefix, member_prefix, non_member_prefix, rel_attacks, dcpdd, device):
    preds = []
    
    # 1. Base Model Pass (Original Text)
    batch_data = raw_values(chunk_batch, model, tokenizer, device)
    
    # Extract base losses now to reuse in attacks
    base_losses = [d['loss'].item() for d in batch_data]
    
    # 2. Base Model Pass (Lowercase)
    lowercase_batch = [c.lower() for c in chunk_batch]
    batch_data_lower = raw_values(lowercase_batch, model, tokenizer, device)
    
    # 3. Rel Attacks
    recall_scores = rel_attacks.batch_recall(chunk_batch, base_losses, negative_prefix)
    conrecall_scores = rel_attacks.batch_conrecall(chunk_batch, base_losses, member_prefix, non_member_prefix)

    # 4. Assembly Loop (Lightweight CPU work)
    for i, sentence in enumerate(chunk_batch):
        data = batch_data[i]
        data_lower = batch_data_lower[i]
        
        base_attacks = BaselineAttacks(
            logits=data['logits'], 
            input_ids=data['input_ids'], 
            token_log_probs=data['token_log_probs']
        )
        max_renyi = MaxRenyiAttack(
            token_probs=data['full_token_probs'], 
            token_log_probs=data['full_log_probs'], 
            input_ids=data['input_ids']
        )
        
        pred = {
            'max_renyi': max_renyi.calculateEntropy(),
            'ppl': math.exp(base_losses[i]), 
            'ppl/lowercase_ppl': -(np.log(math.exp(base_losses[i])) / np.log(math.exp(data_lower['loss'].item()))),
            'ppl/zlib': base_losses[i] / zlib_entropy(sentence),
            'ranks': base_attacks.ranks(),
            # TODO Neighbours batch processing 
            'recall': recall_scores[i],
            'conrecall': conrecall_scores[i],
            'dcpdd': dcpdd.detect(token_probs=data['token_probs'], input_ids=data['input_ids'])
        }

        for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:  
            pred[f"Min_{ratio*100}% Prob"] = base_attacks.min_k(ratio=ratio)
            pred[f"Min_++{ratio*100}% Prob"] = base_attacks.min_k_plus_plus(ratio=ratio)
            
        preds.append(pred)
        
    return preds
