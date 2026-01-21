import torch
from torch import nanmean
import numpy as np
import zlib
import math
import random 
import torch.nn.functional as F
from heapq import nlargest
from collections import defaultdict
import time
from transformers import RobertaTokenizer, AutoTokenizer, LongformerForMaskedLM
import os
import pandas as pd
import re
import string
import json
from wordfreq import word_frequency
from typing import Any, Optional, List, Dict, Tuple, Sequence, Union

MODEL_MAX_LENGTH = 2048
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def fix_seed(seed: int = 0) -> None:
    """
                Set random seeds for reproducibility across torch, numpy and random.
    
                Parameters
                ----------
                seed : int, optional
                    Seed value to set for reproducibility (default is 0).
    
                Returns
                -------
                None
    
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def raw_values(sentence: str, model: torch.nn.Module, tokenizer: Any, device: torch.device) -> Dict[str, Any]:
    """
                Compute loss, per-token probabilities and related tensors for a single sentence.
    
                Parameters
                ----------
                sentence : str
                    Input sentence to evaluate.
                model : torch.nn.Module
                    Language model used to compute logits and loss (expects HF-like forward API).
                tokenizer : Any
                    Tokenizer providing encoding/decoding methods (HuggingFace tokenizer-like).
                device : torch.device
                    Device where tensors and model reside (e.g., torch.device('cuda:0') or 'cpu').
    
                Returns
                -------
                dict
                    Dictionary with keys:
                    - 'loss' (torch.Tensor): average cross-entropy loss for the sentence.
                    - 'token_probs' (torch.Tensor): per-token probabilities for the true labels.
                    - 'token_log_probs' (torch.Tensor): per-token log-probabilities for the true labels.
                    - 'logits' (torch.Tensor): model logits for the input (batched).
                    - 'input_ids' (torch.Tensor): target token ids aligned with logits.
                    - 'full_token_probs' (torch.Tensor): full vocabulary probabilities per position.
                    - 'full_log_probs' (torch.Tensor): full vocabulary log-probabilities per position.
    
    """
    encodings = tokenizer(
        sentence, 
        return_tensors='pt', 
        truncation=True, 
        max_length=MODEL_MAX_LENGTH, 
        padding=False 
    ).to(device)

    with torch.no_grad():
        outputs = model(**encodings)
    
    # Outputs are [1, seq_len, vocab]
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = encodings.input_ids[..., 1:].contiguous()
    
    # logits: [seq_len, vocab], labels: [seq_len]
    logits_sq = shift_logits.squeeze(0)
    labels_sq = shift_labels.squeeze(0)

    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fct(logits_sq, labels_sq)

    log_probs = F.log_softmax(logits_sq, dim=-1)
    probs = F.softmax(logits_sq, dim=-1)

    # Gather token probabilities
    # token_log_probs: [seq_len]
    token_log_probs = log_probs.gather(dim=-1, index=labels_sq.unsqueeze(-1)).squeeze(-1)
    token_probs = probs.gather(dim=-1, index=labels_sq.unsqueeze(-1)).squeeze(-1)

    # Clean up large tensors
    del shift_logits
    del encodings
    del outputs

    return {
        "loss": loss,
        "token_probs": token_probs,
        "token_log_probs": token_log_probs,
        "logits": logits_sq.unsqueeze(0), 
        "input_ids": labels_sq.unsqueeze(0).long(), 
        "full_token_probs": probs,
        "full_log_probs": log_probs
    }

def perplexity(loss: float) -> float:
    """
                Compute perplexity from a cross-entropy loss.
    
                Parameters
                ----------
                loss : float
                    Average cross-entropy loss.
    
                Returns
                -------
                float
                    Exponential of the loss (perplexity).
    
    """
    return math.exp(loss)

def zlib_entropy(sentence: str) -> int:
    """
                Approximate entropy of a string by measuring zlib-compressed length.
    
                Parameters
                ----------
                sentence : str
                    Input text.
    
                Returns
                -------
                int
                    Length in bytes of zlib.compress(sentence.encode('utf-8')).
    
    """
    return len(zlib.compress(bytes(sentence, 'utf-8')))

class Baseline:
    def __init__(self, logits: torch.Tensor, input_ids: torch.Tensor, token_log_probs: torch.Tensor) -> None:
        """
                    Store tensors required by baseline scoring helpers.
        
                    Parameters
                    ----------
                    logits : torch.Tensor
                        Model logits (usually shape [1, seq_len, vocab]).
                    input_ids : torch.Tensor
                        Target token ids aligned with logits.
                    token_log_probs : torch.Tensor
                        Log-probabilities of the target tokens (per position).
        
        """
        self.logits = logits
        self.token_log_probs = token_log_probs
        self.input_ids = input_ids
        
    def min_k(self, ratio: float = 0.05) -> float:
        """
                    Compute mean of the lowest-k token log-probabilities.
        
                    Parameters
                    ----------
                    ratio : float, optional
                        Fraction of tokens to include in the "k" set (default 0.05).
        
                    Returns
                    -------
                    float
                        Mean of the selected lowest log-probabilities.
        
        """
        k_length = max(1, int(len(self.token_log_probs)*ratio))
        topk, _ = torch.topk(self.token_log_probs, k_length, largest=False)
        return nanmean(topk).item()

    def min_k_plus_plus(self, ratio: float = 0.05) -> float:
        """
                    Variation of min_k that normalizes by local mean and variance (z-score like).
        
                    Parameters
                    ----------
                    ratio : float, optional
                        Fraction of tokens to include in the "k" set (default 0.05).
        
                    Returns
                    -------
                    float
                        Mean of the selected normalized scores.
        
        """
        # logits: [1, seq, vocab] -> [seq, vocab]
        probs = F.softmax(self.logits[0], dim=-1)
        log_probs = F.log_softmax(self.logits[0], dim=-1)

        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

        mink_plus = (self.token_log_probs - mu) / (sigma.sqrt() + 1e-10)
        k_length = max(1, int(len(mink_plus) * ratio))
        topk, _ = torch.topk(mink_plus, k_length, largest=False)
        return nanmean(topk).item()

    def ranks(self) -> float:
        """
                    Compute the average rank of the correct tokens in the model logits distribution.
        
                    Returns
                    -------
                    float
                        Average token rank (1 means the token is the top prediction).
        
        """
        logits = self.logits
        labels = self.input_ids if self.input_ids.dim() == 2 else self.input_ids.squeeze(0)
        
        target_logits = logits.gather(-1, labels.unsqueeze(-1))
        ranks = (logits > target_logits).sum(dim=-1).float() + 1

        return torch.nanmean(ranks).item()

class RelativeLikelihood:
    def __init__(self, base_model: Any, base_tokenizer: Any, device: torch.device) -> None:
        """
                    Helper for computing conditional likelihoods using a base language model.
        
                    Parameters
                    ----------
                    base_model : Any
                        Language model used for loss computations.
                    base_tokenizer : Any
                        Tokenizer corresponding to the base_model.
                    device : torch.device
                        Device to run computations on.
        
        """
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.device = device
        self.max_length = getattr(self.base_model.config, "max_position_embeddings", 2048)

    def _get_smart_chunks(self, input_ids: torch.Tensor, num_chunks: int) -> List[torch.Tensor]:
        """
                    Split input_ids into `num_chunks` trying to preserve sentence boundaries.
        
                    Parameters
                    ----------
                    input_ids : torch.Tensor
                        Tensor of shape [1, seq_len] containing token ids.
                    num_chunks : int
                        Desired number of chunks.
        
                    Returns
                    -------
                    list of torch.Tensor
                        List of chunk tensors (each shape [1, chunk_len]).
        
        """
        total_len = input_ids.shape[1]
        max_chunk_size = math.ceil(total_len / num_chunks)
        
        chunks = []
        start_idx = 0
        
        for i in range(num_chunks):
            # If this is the last chunk, take everything remaining
            if i == num_chunks - 1:
                chunks.append(input_ids[:, start_idx:])
                break
                
            # Define the hard limit for this chunk
            end_idx = min(start_idx + max_chunk_size, total_len)
            chunk_ids = input_ids[0, start_idx:end_idx]
            
            # Decode to find punctuation characters
            chunk_text = self.base_tokenizer.decode(chunk_ids, skip_special_tokens=True)
            
            # Search for the LAST sentence delimiter in this text
            # We prioritize newline, then period, question mark, exclamation
            # We restrict search to the second half of the chunk to avoid tiny chunks
            search_start_char = len(chunk_text) // 2
            match = None
            
            # Regex finds last occurrence of [.!?\n]
            # We iterate to find the one closest to the end
            for m in re.finditer(r'[.!?\n]', chunk_text):
                if m.end() > search_start_char:
                    match = m
            
            if match:
                # Found a punctuation!
                split_char_idx = match.end()
                valid_text = chunk_text[:split_char_idx]
                
                # Re-tokenize to get the exact token length of this valid sentence(s)
                # Note: This is crucial because chars != tokens
                valid_tokens = self.base_tokenizer(
                    valid_text, 
                    add_special_tokens=False
                )['input_ids']
                
                # The new split point relative to the full tensor
                split_point = start_idx + len(valid_tokens)
            else:
                # No punctuation found? Fallback to hard split
                split_point = end_idx
            
            # Create the chunk
            chunks.append(input_ids[:, start_idx:split_point])
            
            # Update start for next chunk
            start_idx = split_point
            
        return chunks

    def _get_single_loss(self, prefix_tensor: torch.Tensor, target: str) -> float:
        """
                    Compute the (average) loss for a single prefix + target combination.
        
                    The target text may be split into multiple smart chunks and averaged.
        
                    Parameters
                    ----------
                    prefix_tensor : torch.Tensor
                        Prefix tensor (can be batched) used as context before the target.
                    target : str
                        Target text to evaluate.
        
                    Returns
                    -------
                    float
                        Average loss across target chunks.
        
        """
        # Encode target
        target_enc = self.base_tokenizer(
            target, 
            return_tensors='pt', 
            truncation=True, 
            max_length=self.max_length,
            add_special_tokens=False
        )
        target_ids = target_enc['input_ids'].to(self.device)
        
        if target_ids.numel() == 0:
            return 0.0
        
        num_chunks = prefix_tensor.shape[0] if prefix_tensor.dim() > 1 else 1
        target_chunks = self._get_smart_chunks(target_ids, num_chunks)
        
        losses = []
        for i in range(len(target_chunks)):
            target_chunk = target_chunks[i]
            
            # Select appropriate prefix
            if prefix_tensor.dim() > 1:
                current_prefix = prefix_tensor[i % prefix_tensor.shape[0]].unsqueeze(0)
            else:
                current_prefix = prefix_tensor.unsqueeze(0)
            
            if target_chunk.numel() == 0:
                continue
            
            if current_prefix.numel() > 0:
                input_ids = torch.cat((current_prefix, target_chunk), dim=1)
                prefix_len = current_prefix.shape[1]
            else:
                input_ids = target_chunk
                prefix_len = 0  
            
            # Truncate if the input_ids are bigger than max length
            if input_ids.shape[1] > self.max_length:
                # Keep prefix intact, trim target from the end
                input_ids = input_ids[:, :self.max_length]

            labels = input_ids.clone()
            if prefix_len > 0:
                labels[:, :prefix_len] = -100 

            with torch.no_grad():
                outputs = self.base_model(input_ids, labels=labels)
                losses.append(outputs.loss.item())

        if not losses:
            return 0.0
            
        # Return the AVERAGE loss across chunks
        return sum(losses) / len(losses)

    def calc_recall(self, text: str, base_loss: float, negative_prefix_list: List[torch.Tensor]) -> float:
        """
                    Compute a recall-like metric comparing conditional loss with a set of negative prefixes.
        
                    Parameters
                    ----------
                    text : str
                        Target text.
                    base_loss : float
                        Loss of the target under the base model.
                    negative_prefix_list : list of torch.Tensor
                        List of negative prefix tensors to compute conditional loss.
        
                    Returns
                    -------
                    float
                        Ratio cond_loss / base_loss (or 0.0 when base_loss == 0).
        
        """
        if negative_prefix_list:
            # Concatenate and ensure it's on the correct device
            prefix_tensor = torch.cat(negative_prefix_list, dim=1).to(self.device)
        else:
            prefix_tensor = torch.tensor([[]], dtype=torch.long).to(self.device)
        
        # 1. Calculate Loss
        cond_loss = self._get_single_loss(prefix_tensor, text)
        
        if base_loss == 0:
            return 0.0
        return cond_loss / base_loss

    def calc_conrecall(self, text: str, base_loss: float, member_prefix_list: List[torch.Tensor], non_member_prefix_list: List[torch.Tensor]) -> Dict[str, float]:
        """
                    Compute conrecall metrics across several gamma thresholds comparing member and non-member prefixes.
        
                    Parameters
                    ----------
                    text : str
                        Target text.
                    base_loss : float
                        Loss of the target under the base model.
                    member_prefix_list : list of torch.Tensor
                        List of member prefix tensors.
                    non_member_prefix_list : list of torch.Tensor
                        List of non-member prefix tensors.
        
                    Returns
                    -------
                    dict
                        Dictionary mapping 'conrecall_gamma_*' to float scores.
        
        """
        # Member shots
        if member_prefix_list:
            mem_tensor = torch.cat(member_prefix_list, dim=1).to(self.device)
        else:
            mem_tensor = torch.tensor([[]], dtype=torch.long).to(self.device)

        # Non-Member Shots
        if non_member_prefix_list:
            non_tensor = torch.cat(non_member_prefix_list, dim=1).to(self.device)
        else:
            non_tensor = torch.tensor([[]], dtype=torch.long).to(self.device)
        
        cond_loss_mem = self._get_single_loss(mem_tensor, text)
        cond_loss_non = self._get_single_loss(non_tensor, text)
        
        ll = base_loss
        ll_member = -cond_loss_mem
        ll_nonmember = -cond_loss_non
        scores = {}
    
        for gamma in [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            val = 0.0
            if ll != 0:
                val = (ll_nonmember - gamma * ll_member) / ll
            scores[f"conrecall_gamma_{int(100*gamma)}"] = val
        return scores

class TagTab:
    def __init__(self, target_model: Any, target_tokenizer: Any, k: int, nlp: Any, device: torch.device, min_size: int = 7) -> None:
        """
                    Keyword/NER based scoring helper (TAG-TAB style).
        
                    Parameters
                    ----------
                    target_model : Any
                        Model used to compute per-token probabilities.
                    target_tokenizer : Any
                        Tokenizer corresponding to the model.
                    k : int
                        Number of low-entropy words to select per sentence.
                    nlp : Any
                        NLP pipeline (spaCy-like) used for sentence splitting and NER.
                    device : torch.device
                        Device used for model inference.
                    min_size : int, optional
                        Minimum token count for a sentence to be considered (default 7).
        
        """
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.k = k
        self.device = device
        self.min_size = min_size
        self.nlp = nlp
        self.punc_table = str.maketrans("", "", string.punctuation)
    
    def _normalize_token(self, w: str) -> str:
        """
                    Normalize a token by lowercasing, removing punctuation and whitespace.
        
                    Parameters
                    ----------
                    w : str
                        Input token text.
        
                    Returns
                    -------
                    str
                        Normalized token.
        
        """
        w = w.lower().translate(self.punc_table)
        w = re.sub(r"\s+", "", w)
        return w
    
    def get_tab_keywords(self, text: str) -> List[Tuple[np.ndarray, List[str]]]:
        """
                    Extract candidate keyword index sets and corresponding words per sentence.
        
                    Parameters
                    ----------
                    text : str
                        Input document text.
        
                    Returns
                    -------
                    list of (np.ndarray, list)
                        Each element is a tuple (indices_array, selected_words).
        
        """
        docs = self.nlp(text)
        result = []
        for sent_span in docs.sents:
            original_tokens = [t.text for t in sent_span]
            norm_tokens = [self._normalize_token(t.text) for t in sent_span]
            valid_indices = [i for i, t in enumerate(norm_tokens) if t]
            
            # Too small sentences are noisy, so we ignore them
            if len(valid_indices) < self.min_size:
                continue
            
            entropies = []
            for i in valid_indices:
                word = norm_tokens[i]
                p = word_frequency(word, 'en')
                if p > 0.0:
                    p_32 = np.float32(p)
                    ent = -p_32 * np.log2(p_32, dtype=np.float32)
                else:
                    ent = np.float32(0.0)
                entropies.append((ent, i))
            
            entropies.sort(key=lambda x: x[0])
            top_k = entropies[:self.k]
            final_token_indices = [x[1] for x in top_k]

            ner_indices = []
            for ent in sent_span.ents:
                for i in range(ent.start, ent.end):
                    if self._normalize_token(docs[i].text):
                        rel_idx = i - sent_span.start
                        ner_indices.append(rel_idx)
            
            # Combine low entropy words with NER words
            final_indices = np.union1d(np.unique(ner_indices), final_token_indices).astype(int)
            selected_words = [original_tokens[i] for i in final_indices]

            result.append((final_indices, selected_words))

        return result
    
    def get_tab_score(self, text: str, keywords: List[Tuple[np.ndarray, List[str]]]) -> float:
        """
                    Compute an average log-probability score for spans corresponding to the provided keywords.
        
                    Parameters
                    ----------
                    text : str
                        Input text.
                    keywords : list of (np.ndarray, list)
                        Output from get_tab_keywords indicating keyword spans and words.
        
                    Returns
                    -------
                    float
                        Mean log-probability across matched keyword spans (or 0.0 if none found).
        
        """
        inputs = self.target_tokenizer(text, padding = True, truncation = True, max_length = MODEL_MAX_LENGTH, return_tensors='pt').to(self.device)
        input_ids = inputs.input_ids
        
        with torch.no_grad():
            outputs = self.target_model(**inputs, labels=inputs.input_ids)
         
        shift_logits = outputs.logits[0, :-1, :].contiguous()
        shift_labels = inputs.input_ids[0, 1:].contiguous()
        
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
        
        seq_ids = input_ids[0]
        decoded_tokens = [self.target_tokenizer.decode([t]) for t in seq_ids]
        
        current_token_idx = 0
        flat_keywords = [item for sublist in keywords for item in sublist[1]]
        relevant_log_probs = []
        
        for keyword in flat_keywords:
            keyword_clean = keyword.strip().lower()
            if not keyword_clean: 
                continue
            
            temp_str = ""
            start_idx = current_token_idx
            found = False
            
            for i in range(current_token_idx, len(decoded_tokens)):
                temp_str += decoded_tokens[i]
                temp_str_clean = temp_str.lower().replace(" ", "").strip()
                
                if keyword_clean in temp_str_clean:
                    chunk_log_probs = token_log_probs[start_idx : i+1].cpu().numpy()
                    relevant_log_probs.extend(chunk_log_probs)
                    current_token_idx = i + 1
                    found = True
                    break
                
                # Optimization: Stop looking if temp string is way longer than keyword
                if len(temp_str_clean) > len(keyword_clean) + 20:
                    break
            
            if not found:
                continue

        if not relevant_log_probs:
            return 0.0
            
        return np.mean(relevant_log_probs, dtype=np.float32)
    
    def predict(self, text: str) -> float:
        """
                    Convenience wrapper that extracts keywords and returns the TAB score for the text.
        
                    Parameters
                    ----------
                    text : str
                        Input text.
        
                    Returns
                    -------
                    float
                        TAB score (0.0 when no keywords are found).
        
        """
        keyword_batches = self.get_tab_keywords(text)
        if not keyword_batches:
            return 0.0
        return self.get_tab_score(text, keyword_batches)

class NoisyNeighbour:
    def __init__(self, model: Any, sigma: float, device: torch.device, batch_size: int = 4) -> None:
        """
                    Neighbor-based attack that adds Gaussian noise in embedding space and measures loss change.
        
                    Parameters
                    ----------
                    model : Any
                        Target model supporting inputs_embeds in forward.
                    sigma : float
                        Standard deviation of Gaussian noise to add to embeddings.
                    device : torch.device
                        Device used for computation.
                    batch_size : int, optional
                        Number of noisy samples to compute per batch (default 4).
        
        """
        self.model = model
        self.sigma = sigma
        self.batch_size = batch_size
        self.device = device
        self.dtype = model.dtype 

    def predict(self, input_ids: torch.Tensor, base_loss: Optional[float] = None, num_of_neighbour: int = 48) -> Optional[float]:
        """
                    Compute the difference between original loss and average noisy-neighbour loss.
        
                    Parameters
                    ----------
                    input_ids : torch.Tensor
                        Token ids for the input sequence.
                    base_loss : float or None, optional
                        Precomputed loss for the original input. If None, it will be computed.
                    num_of_neighbour : int, optional
                        Total number of noisy neighbors to generate (should be divisible by batch_size).
        
                    Returns
                    -------
                    float or None
                        Original loss minus average neighbour loss (or None on input error).
        
        """
        if num_of_neighbour % self.batch_size != 0:
            print("ERROR: Number of Neighbours is not divisible by batch size")
            return None
        
        token_ids = input_ids.to(self.device)
        embedding_layer = self.model.get_input_embeddings()
        
        neighbour_loss = 0
        
        with torch.no_grad():
            if base_loss is not None:
                original_loss = base_loss
            else:
                original_loss = self.model(input_ids=token_ids, labels=token_ids).loss.item()
            
            # Embeddings holen
            input_embeddings = embedding_layer(token_ids).to(dtype=self.dtype, device=self.device)
            
            seq_len = input_embeddings.shape[1]
            hidden_dim = input_embeddings.shape[2]
            
            batch_labels = token_ids.repeat(self.batch_size, 1)
            num_batches = num_of_neighbour // self.batch_size
            
            for _ in range(num_batches):
                # Generate noise 
                noise = torch.randn(
                    (self.batch_size, seq_len, hidden_dim), 
                    device=self.device, 
                    dtype=self.dtype
                ) * self.sigma
                
                noisy_embeddings = torch.add(input_embeddings, noise, alpha=1)
                
                neighbour_loss += self.model(inputs_embeds=noisy_embeddings, labels=batch_labels).loss.item()

        avg_neighbor_loss = neighbour_loss / num_batches
        return original_loss - avg_neighbor_loss

class OfflineRobustMIA:
    def __init__(self, target_model: Any, target_tokenizer: Any, reference_models: List[Any], reference_tokenizers: List[Any], a_range: np.ndarray, gamma: float, device: torch.device) -> None:
        """
                    Initialize the offline robust MIA scorer with reference models and parameters.
        
                    Parameters
                    ----------
                    target_model : Any
                        Model under test.
                    target_tokenizer : Any
                        Tokenizer for the target model.
                    reference_models : list
                        List of reference models used to approximate background distribution.
                    reference_tokenizers : list
                        Corresponding tokenizers for the reference models.
                    a_range : np.ndarray
                        Array of 'a' parameter values to evaluate.
                    gamma : float
                        Threshold parameter for the LR test.
                    device : torch.device
                        Device for running computations.
        
        """
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.reference_models = reference_models
        self.reference_tokenizers = reference_tokenizers
        self.a_vals = a_range
        self.gamma = gamma     
        self.device = device
        self.population_ratios = None 

    def _get_prob(self, model: Any, tokenizer: Any, text: str) -> float:
        """
                    Compute model probability proxy exp(-loss) for a single text using a given model/tokenizer.
        
                    Parameters
                    ----------
                    model : Any
                        Model to evaluate.
                    tokenizer : Any
                        Tokenizer for the model.
                    text : str
                        Input text.
        
                    Returns
                    -------
                    float
                        exp(-loss) probability proxy.
        
        """
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs.input_ids).loss.item()
        return np.exp(-loss)

    def compute_ratios(self, text: str) -> np.ndarray:
        """
                    Compute vector of ratios required by the robust MIA decision rule for a text.
        
                    Parameters
                    ----------
                    text : str
                        Input text.
        
                    Returns
                    -------
                    np.ndarray
                        Array of ratios (one per a value).
        
        """
        # 1. Pr(x | theta_target)
        pr_target = self._get_prob(self.target_model, self.target_tokenizer, text)
        
        # 2. Pr(x)_OUT - Average over ALL reference models 
        ref_probs = []
        for model, tokenizer in zip(self.reference_models, self.reference_tokenizers):
            prob = self._get_prob(model, tokenizer, text)
            ref_probs.append(prob)
        
        # Take the mean of probabilities
        pr_out_avg = np.mean(ref_probs)
        
        # 3. Approximate Pr(x)
        pr_x_approx = 0.5 * ((1 + self.a_vals) * pr_out_avg + (1 - self.a_vals))
        
        return pr_target / pr_x_approx

    def precompute_population_statistics(self, population_samples: List[str]) -> None:
        """
                    Precompute ratio statistics for a population of background samples (run once).
        
                    Parameters
                    ----------
                    population_samples : list of str
                        Background samples used to form the null distribution.
        
                    Returns
                    -------
                    None
        
        """
        print(f"Pre-computing statistics for {len(population_samples)} population samples...")
        all_ratios = []
        for z_text in population_samples:
            # Returns array of shape (len(a_vals),)
            ratios_z = self.compute_ratios(z_text)
            all_ratios.append(ratios_z)
        
        # Stack into matrix: (num_samples, num_a_vals)
        self.population_ratios = np.stack(all_ratios)
        print("Pre-computation complete.")

    def predict(self, target_text: str) -> Dict[float, float]:
        """
                    Perform a fast prediction using precomputed population statistics.
        
                    Parameters
                    ----------
                    target_text : str
                        Text whose membership likelihood will be evaluated.
        
                    Returns
                    -------
                    dict
                        Mapping from a value (float) to a score (float) between 0 and 1.
        
        """
        if self.population_ratios is None:
            raise ValueError("You must call precompute_population_statistics first!")

        # Step 1: Compute ratios for target sample x (Array of ratios)
        ratios_x = self.compute_ratios(target_text)
        
        # Step 2 & 3: Pairwise Likelihood Ratio Test (Fully Vectorized)
        # Broadcasting: 
        # ratios_x shape: (num_a,) -> broadcast to (1, num_a)
        # population_ratios shape: (num_z, num_a)
        # Result shape: (num_z, num_a)
        
        lr_matrix = ratios_x / self.population_ratios
        
        # Check dominance: which z samples are dominated?
        # dominated_mask shape: (num_z, num_a)
        dominated_mask = lr_matrix > self.gamma
        
        # Sum down the column (summing over z samples)
        # counts_dominated shape: (num_a,)
        counts_dominated = dominated_mask.sum(axis=0)
        
        # Step 4: Final Score is fraction of z samples dominated [cite: 567]
        num_z = self.population_ratios.shape[0]
        scores = counts_dominated / num_z
        
        return {a: score for a, score in zip(self.a_vals, scores)}
    
class MaxRenyi:
    def __init__(self, token_probs: torch.Tensor, full_log_probs: torch.Tensor, full_token_probs: torch.Tensor, epsilon: float = 1e-10) -> None:
        """
                    Compute a variety of entropy and Rényi-based metrics from token distributions.
        
                    Parameters
                    ----------
                    token_probs : torch.Tensor
                        Probabilities of the correct tokens (p_y).
                    full_log_probs : torch.Tensor
                        Full vocabulary log probabilities per position.
                    full_token_probs : torch.Tensor
                        Full vocabulary probabilities per position.
                    epsilon : float, optional
                        Numerical floor to avoid log/zero issues.
        
        """
        self.token_probs = token_probs
        self.full_log_probs = full_log_probs
        self.full_token_probs = full_token_probs
        self.epsilon = epsilon
    """
        Args:
        token_probs: Probabilities of the correct tokens (p_y).
        full_token_probs: Full vocabulary probability distribution.
        full_log_probs: Full vocabulary log distribution.
    """
    
    def predict(self) -> Dict[str, float]:
        """
                    Compute various entropy-based summaries and top-k ratio metrics.
        
                    Returns
                    -------
                    dict
                        Dictionary of aggregated metrics (means and top-ratio statistics).
        
        """
        # SAFETY CHECK FOR EMPTY TENSORS
        if self.token_probs.numel() == 0:
            # Return a dict with zeros for all potential keys to avoid KeyError downstream
            # Keys derived from loops below
            dummy = {}
            dummy['gap_prob_mean'] = 0.0
            for name in ["entropies", "renyi_05", "renyi_2", "renyi_inf", "modified_entropies", "mod_renyi_05", "mod_renyi_2"]:
                 dummy[f"{name}_mean"] = 0.0
            for name in ["entropies", "renyi_05", "renyi_2", "renyi_inf"]:
                for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                    dummy[f"{name}_ratio_{int(ratio*100)}"] = 0.0
            return dummy
        # Ensure precision
        token_probs = self.token_probs.to(dtype=torch.float64)
        full_log_probs = self.full_log_probs.to(dtype=torch.float64)
        full_token_probs = self.full_token_probs.to(dtype=torch.float64)
        
        metrics = {}
        # Safe probabilities
        probs_safe = torch.clamp(full_token_probs, min=self.epsilon, max=1-self.epsilon)

        # 1. Standard Entropy (Rényi alpha=1)
        metrics["entropies"] = -(full_token_probs * full_log_probs).sum(dim=-1)

        # 2. Rényi Entropy (Alpha = 0.5)
        alpha_05 = 0.5
        metrics["renyi_05"] = (1 / (1 - alpha_05)) * torch.log(torch.sum(torch.pow(probs_safe, alpha_05), dim=-1))

        # 3. Rényi Entropy (Alpha = 2)
        alpha_2 = 2.0
        metrics["renyi_2"] = (1 / (1 - alpha_2)) * torch.log(torch.sum(torch.pow(probs_safe, alpha_2), dim=-1))

        # 4. Max Prob & Gap Prob
        # Note: max_p is the log probability. Negating it gives Rényi (Alpha = infinity).
        top2_vals, _ = torch.topk(full_log_probs, 2, dim=-1)
        max_p = top2_vals[:, 0]
        second_p = top2_vals[:, 1]
        
        metrics["renyi_inf"] = -max_p  # Corresponds to "Max_...% renyi_inf" in original code
        metrics["gap_prob"] = max_p - second_p # Gap between top1 and top2

        # 5. Modified Entropy
        p_y = token_probs
        term_a = -(1 - p_y) * torch.log(p_y)
        term_b = -(full_token_probs * torch.log(1 - probs_safe)).sum(dim=-1)
        term_c = p_y * torch.log(1 - p_y)
        
        metrics["modified_entropies"] = term_a + term_b + term_c
        
        for alpha in [0.5, 2]:
            k = abs(1 - alpha)
            sum_all = torch.sum(probs_safe * torch.pow(1 - probs_safe, k), dim=-1)
            term_target = p_y * torch.pow(1 - p_y, k)
            sum_remaining = sum_all - term_target
            sum_probs_remaining = (1 - p_y)
            
            val = - (1 / k) * (
                ((1 - p_y) * torch.pow(p_y, k)) 
                - (1 - p_y) 
                + sum_remaining 
                - sum_probs_remaining
            )
            metrics[f"mod_renyi_{str(alpha).replace('.', '')}"] = val

        results = {}

        for name, tensor_val in metrics.items():
            # Handle potential NaNs/Infs
            tensor_val = torch.nan_to_num(tensor_val, nan=0.0, posinf=1e9, neginf=-1e9)
            
            # A. Global Mean
            # Original code negates the mean for gap_prob
            if name == "gap_prob":
                results[f"{name}_mean"] = tensor_val.mean().item()
            else:
                results[f"{name}_mean"] = -tensor_val.mean().item()
            
            # B. Loop Ratios (MaxRényi-K%)
            # Only calculate ratios for the entropy metrics (including renyi_inf)
            if name in ["entropies", "renyi_05", "renyi_2", "renyi_inf"]:
                for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                    k_len = max(1, int(len(tensor_val) * ratio))
                    
                    # MaxRényi selects the LARGEST entropy values (highest uncertainty)
                    # Original code: np.sort(entropies)[-k_length:] -> Largest values
                    k_vals, _ = torch.topk(tensor_val, k_len, largest=True)
                    
                    results[f"{name}_ratio_{int(ratio*100)}"] = -k_vals.mean().item()
        
        return results
    
class DCPDD:
    def __init__(self, freq_dict: np.ndarray, device: torch.device) -> None:
        """
                    Density-based frequency prior scorer initializer.
        
                    Parameters
                    ----------
                    freq_dict : np.ndarray
                        Frequency table (index -> frequency) used for scoring.
                    device : torch.device
                        Device where tensors will be placed.
        
        """
        self.a = 1e-10
        self.device = device
        self.freq_tensor = torch.from_numpy(freq_dict).to(self.device)

    def predict(self, token_probs: torch.Tensor, input_ids: torch.Tensor) -> float:
        """
                    Score input by combining token probabilities with frequency priors.
        
                    Parameters
                    ----------
                    token_probs : torch.Tensor
                        Per-token probabilities for the true labels.
                    input_ids : torch.Tensor
                        Token ids aligned with token_probs.
        
                    Returns
                    -------
                    float
                        Negative mean cross-entropy-like score (higher indicates more likely membership).
        
        """
        # Grab unique indexes. We use np.unique because they can return index.
        ids_np = input_ids.detach().cpu().numpy() 
        _, unique_indices_np = np.unique(ids_np, return_index=True)
        unique_indices = torch.from_numpy(unique_indices_np).long().to(self.device)
        
        x_pro = token_probs[unique_indices]
        valid_ids = input_ids[unique_indices]
        
        max_freq_id = len(self.freq_tensor) - 1
        valid_ids = torch.clip(valid_ids, 0, max_freq_id)
        x_fre = self.freq_tensor[valid_ids]
        
        epsilon = 1e-10
        x_fre = torch.where(x_fre == 0, epsilon, x_fre)
        
        ce = torch.mul(x_pro, torch.log(1 / x_fre))
        ce = torch.where(ce > self.a, ce, self.a)
        return -float(nanmean(ce))

class CIMIA:
    def __init__(self, target_model: Any, target_tokenizer: Any, device: torch.device, max_len: int, calibration_signal: Dict[str, List[float]]) -> None:
        """
                    Initialize combined signals based membership inference (CIMIA) helper.
        
                    Parameters
                    ----------
                    target_model : Any
                        Model used to compute losses.
                    target_tokenizer : Any
                        Tokenizer for the target model.
                    device : torch.device
                        Device used for computation.
                    max_len : int
                        Maximum length used for truncation and signal calculations.
                    calibration_signal : dict
                        Precomputed calibration signal distributions used for p-value calculation.
        
        """
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.device = device
        self.max_len = max_len
        self.T = int(self.max_len/2)
        self.m = math.ceil(np.log10(self.max_len))
        self.calibration_signal = calibration_signal
        
    def diversity_calibration(self, input_ids: torch.Tensor, token_log_probs: torch.Tensor) -> float:
        """
                    Compute a diversity-calibrated loss: average loss divided by token diversity in prefix.
        
                    Parameters
                    ----------
                    input_ids : torch.Tensor
                        Input token ids.
                    token_log_probs : torch.Tensor
                        Per-token log-probabilities.
        
                    Returns
                    -------
                    float
                        Diversity-calibrated loss (or 0.0 if input is empty).
        
        """
        cut_off = input_ids[:self.T]
        if cut_off.numel() == 0: 
            return 0.0
        unique_ids = torch.unique(cut_off)
        d_X = unique_ids.numel() / cut_off.numel() 
        loss = (-token_log_probs[:self.T]).mean().item()
        return loss / d_X
    
    def cut_off_loss(self, input_ids: torch.Tensor) -> float:
        """
                    Compute loss on the truncated prefix of the input (up to T tokens).
        
                    Parameters
                    ----------
                    input_ids : torch.Tensor
                        Input token ids.
        
                    Returns
                    -------
                    float
                        Loss value computed on the cutoff prefix.
        
        """
        cut_off = input_ids[:self.T]
        
        if cut_off.dim() == 1:
            cut_off = cut_off.unsqueeze(0)
            
        with torch.no_grad():
            loss = self.target_model(cut_off, labels=cut_off).loss.item()
        return loss
    
    def slope_loss(self, input_ids: torch.Tensor, token_log_probs: torch.Tensor) -> float:
        """
                    Compute slope-based signal comparing token losses across positions.
        
                    Parameters
                    ----------
                    input_ids : torch.Tensor
                        Input token ids.
                    token_log_probs : torch.Tensor
                        Per-token log-probabilities.
        
                    Returns
                    -------
                    float
                        Slope-derived signal (or 0.0 if undefined).
        
        """
        cut_off = input_ids[:self.T]
        cut_off_batch = cut_off.unsqueeze(0) if cut_off.dim() == 1 else cut_off
        
        t_index = torch.arange(self.T).to(self.device)
        t_average = torch.mean(t_index.float()).to(self.device)
        with torch.no_grad():
            loss = self.target_model(cut_off_batch, labels=cut_off_batch).loss.to(self.device)
            nominator = torch.sum((t_index-t_average)*(-token_log_probs[:self.T]-loss)).item()
            denominator = torch.sum((t_index-t_average)**2).item()
        
        if denominator == 0:
            return 0.0
        
        return nominator/denominator
    
    def robust_low_loss_counting(self, thresholds: List[float], token_log_probs: torch.Tensor) -> List[float]:
        """
                    Count fraction of positions with token losses below given thresholds.
        
                    Parameters
                    ----------
                    thresholds : list of float
                        Thresholds to evaluate.
                    token_log_probs : torch.Tensor
                        Per-token log-probabilities.
        
                    Returns
                    -------
                    list of float
                        Fractions for each threshold plus two aggregate measures.
        
        """
        if token_log_probs[:self.T].numel() == 0:
            return [0.0] * (len(thresholds) + 2)
        
        f_cb = []
        for threshold in thresholds:
            token_losses = -token_log_probs[:self.T]
            loss_binary = torch.where(token_losses < threshold, 1, 0)
            f_cb.append(torch.mean(loss_binary.float()).item())

        loss_mean = torch.mean(-token_log_probs[:self.T]).item()
        loss_binary = torch.where(-token_log_probs[:self.T] < loss_mean, 1, 0)
        f_cbm = torch.mean(loss_binary.float()).item()
        
        token_losses = -token_log_probs[:self.T]
        cumsum = torch.cumsum(token_losses, dim=0)
        indices = torch.arange(1, self.T + 1).to(self.device)
        running_means = cumsum / indices
        f_cbpm = (token_losses[1:] < running_means[:-1]).float().mean().item()

        return f_cb + [f_cbm, f_cbpm]
    
    def repetition_amplification(self, input_ids: torch.Tensor, loss: float) -> float:
        """
                    Measure how much the loss decreases when repeating the input sequence.
        
                    Parameters
                    ----------
                    input_ids : torch.Tensor
                        Input token ids.
                    loss : float
                        Baseline loss for the input.
        
                    Returns
                    -------
                    float
                        Difference between original and repeated-segment loss.
        
        """
        if input_ids.numel() == 0: 
            return 0.0
        safe_len = int(MODEL_MAX_LENGTH / 2)
        
        current_input = input_ids
        current_loss = loss

        # Truncate if necessary
        if input_ids.numel() > safe_len:
            current_input = input_ids[:safe_len]
            
            # Recalculate baseline loss for this specific truncated segment
            if current_input.dim() == 1:
                inp_batch = current_input.unsqueeze(0)
            else:
                inp_batch = current_input
                
            with torch.no_grad():
                current_loss = self.target_model(inp_batch, labels=inp_batch).loss.item()

        # Create Repetition
        if current_input.dim() == 1:
            repeat_input_ids = current_input.unsqueeze(0).repeat(1, 2)
        else:
            repeat_input_ids = current_input.repeat(1, 2)
            
        labels = repeat_input_ids.clone()
        # Mask the first half, so we only calculate loss on the second (repeated) half
        labels[:, :current_input.numel()] = -100
        
        with torch.no_grad():
            rep_loss = self.target_model(repeat_input_ids, labels=labels).loss.item()
            
        return current_loss - rep_loss
    
    def lempel_ziv_complexity(self, bins: int, token_log_probs: torch.Tensor) -> float:
        """
                    Estimate Lempel-Ziv complexity on discretized token-loss sequence.
        
                    Parameters
                    ----------
                    bins : int
                        Number of bins for discretization.
                    token_log_probs : torch.Tensor
                        Per-token log-probabilities.
        
                    Returns
                    -------
                    float
                        Normalized LZ complexity measure.
        
        """
        x = -token_log_probs[:self.T].detach().cpu().numpy()
        x = np.asarray(x)
        if len(x) == 0: 
            return 0.0
        
        bins_edges = np.linspace(min(x), 0, 100, dtype=np.float32)
        x = np.digitize(x, bins=bins_edges, right=True)

        bins = np.linspace(np.min(x), np.max(x), bins + 1, dtype=np.float32)[1:]
        sequence = np.searchsorted(bins, x, side="left")

        sub_strings = set()
        n = len(sequence)

        ind = 0
        inc = 1
        while ind + inc <= n:
            sub_str = tuple(sequence[ind : ind + inc])
            if sub_str in sub_strings:
                inc += 1
            else:
                sub_strings.add(sub_str)
                ind += inc
                inc = 1
        return len(sub_strings) / n
        
    def approximate_entropy(self, m: int, r: float, token_log_probs: torch.Tensor) -> float:
        """
                    Approximate entropy (ApEn) estimator for the token-loss sequence.
        
                    Parameters
                    ----------
                    m : int
                        Embedding dimension.
                    r : float
                        Tolerance multiplier (scaled by std of sequence).
                    token_log_probs : torch.Tensor
                        Per-token log-probabilities.
        
                    Returns
                    -------
                    float
                        Approximate entropy value.
        
        """
        x = -token_log_probs[:self.T].detach().cpu().numpy()
        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)

        N = x.size
        r *= np.std(x, dtype=np.float32)
        if r < 0: raise ValueError("Parameter r must be positive.")
        if N <= m + 1: return 0

        def _phi(m):
            """
                        Internal helper used by approximate_entropy. No external typing enforced.
            
            """
            x_re = np.array([x[i : i + m] for i in range(N - m + 1)])
            C = np.sum(
                np.max(np.abs(x_re[:, np.newaxis] - x_re[np.newaxis, :]), axis=2) <= r,
                axis=0,
            ) / (N - m + 1)
            return np.sum(np.log(C), dtype=np.float32) / (N - m + 1.0)

        return np.abs(_phi(m) - _phi(m + 1))

    def calculate_p_values(self, value: float, calibration_values: Sequence[float], is_higher_better: bool = False) -> float:
        """
                    Compute empirical p-value of `value` relative to calibration distribution.
        
                    Parameters
                    ----------
                    value : float
                        Observed value.
                    calibration_values : sequence of float
                        Calibration (background) values to compare against.
                    is_higher_better : bool, optional
                        If True, higher values are better (invert sign) for ranking purposes.
        
                    Returns
                    -------
                    float
                        Empirical p-value in (0,1], returned as python float.
        
        """
        cal_vals = np.array(calibration_values, dtype=np.float32)
        if len(cal_vals) == 0: 
            return 0.5 
        
        if is_higher_better:
            cal_vals = -cal_vals
            value = -value
            
        p_val = (np.sum(cal_vals <= value) + 1) / (len(cal_vals) + 1)
        return p_val.item()
    
    def predict(self, loss: float, input_ids: torch.Tensor, token_log_probs: torch.Tensor, combining_method: str = "edgington", raw_signals: bool = False) -> Union[float, Dict[str, float]]:
        """
                    Compute combined CIMIA score (or return raw signals if requested).
        
                    Parameters
                    ----------
                    loss : float
                        Baseline loss for the input.
                    input_ids : torch.Tensor
                        Input token ids.
                    token_log_probs : torch.Tensor
                        Per-token log-probabilities.
                    combining_method : str, optional
                        Method to combine p-values into a single score (default 'edgington').
                    raw_signals : bool, optional
                        If True, return the raw signal dictionary instead of a aggregated score.
        
                    Returns
                    -------
                    float or dict
                        Aggregated score (float) or raw signal dictionary.
        
        """
        self.T = max(1, min(input_ids.numel(), int(self.max_len / 2)))
        
        # Calculate each signal
        signals = {}
        signals['div_loss'] = self.diversity_calibration(input_ids, token_log_probs)
        signals['cut_off_loss'] = self.cut_off_loss(input_ids)
        signals['slope'] = self.slope_loss(input_ids, token_log_probs)
        signals['rep_amp'] = self.repetition_amplification(input_ids, loss)
        
        low_loss_vals = self.robust_low_loss_counting(thresholds=[1.0, 2.0, 3.0], token_log_probs=token_log_probs)
        signals['f_cb_1'] = low_loss_vals[0]
        signals['f_cb_2'] = low_loss_vals[1]
        signals['f_cb_3'] = low_loss_vals[2]
        signals['f_cbm'] = low_loss_vals[3]
        signals['f_cbpm'] = low_loss_vals[4]

        signals['lz'] = self.lempel_ziv_complexity(bins=100, token_log_probs=token_log_probs)
        signals['apen'] = self.approximate_entropy(m=self.m, r=0.8, token_log_probs=token_log_probs)
        
        if raw_signals:
            return signals

        # Calculate p-value, which are standardized signals between 0 and 1
        p_values_list = []
        for name, value in signals.items():
            if name in self.calibration_signal:
                is_count = 'f_cb' in name 
                p_val = self.calculate_p_values(value, self.calibration_signal[name], is_higher_better=is_count)
                p_values_list.append(p_val)
            else:
                p_values_list.append(0.5)
                
        p_values = torch.tensor(p_values_list)
        
        # We choose edgington as standard combining method, since it is the most stable. 
        if combining_method == "edgington":
            score = torch.sum(p_values) 
        elif combining_method == "fisher":
            score = -2*torch.sum(torch.log(p_values))
        elif combining_method == "pearson":
            score = -2*torch.sum(torch.log(1 - p_values))
        elif combining_method == "george":
            score = torch.sum(torch.log(p_values / (1 - p_values)))
        else:
            score = torch.sum(p_values)

        return score.item()
    
class ACMIA:
    def __init__(self, device: torch.device, logits: torch.Tensor, probs: torch.Tensor, log_probs: torch.Tensor, token_log_probs: torch.Tensor, input_ids: torch.Tensor, temperatures: np.ndarray) -> None:
        """
                    Auto-calibration and temperature scaling helper initializer.
        
                    Parameters
                    ----------
                    device : torch.device
                        Device used for computations.
                    logits : torch.Tensor
                        Model logits (shape [1, seq_len, vocab]).
                    probs : torch.Tensor
                        Softmax probabilities for logits.
                    log_probs : torch.Tensor
                        Log-softmax for logits.
                    token_log_probs : torch.Tensor
                        Per-token log-probabilities for the true labels.
                    input_ids : torch.Tensor
                        Input ids tensor used for gathering target log-probs.
                    temperatures : np.ndarray
                        Temperature values to use for scaling experiments.
        
        """
        self.device = device
        self.logits = logits
        self.probs = probs
        self.log_probs = log_probs
        self.token_log_probs = token_log_probs
        self.input_ids = input_ids
        self.temperatures = temperatures
        
    def get_fos_mask(self) -> torch.Tensor:
        """
                    Create a boolean mask marking the first occurrence of each token in the sequence (FOS).
        
                    Returns
                    -------
                    torch.Tensor
                        Boolean mask (1D) with True at first occurrences.
        
        """
        # FIX: Squeeze input_ids to 1D [seq_len] to calculate indices
        input_ids_flat = self.input_ids.squeeze().cpu().numpy()
        
        _, first_indices = np.unique(input_ids_flat, return_index=True)
        
        # Create 1D mask
        mask = np.zeros_like(input_ids_flat, dtype=bool)
        mask[first_indices] = True
        
        # FIX: Return 1D mask (shape: [seq_len]). Do not reshape to (1, -1).
        return torch.from_numpy(mask).to(self.device)

    def temperature_scaling(self) -> Tuple[list, list, list]:
        """
                    Apply temperature scaling to the full vocabulary log-probabilities and extract statistics.
        
                    Returns
                    -------
                    tuple of lists
                        (log_probs_by_temperature, mu_by_temperature, sigma_by_temperature)
        
        """
        log_probs_temprature = []
        mu_temprature = []
        sigma_temprature = []

        # [1, seq_len, vocab_size]
        log_probs_expanded = self.log_probs.unsqueeze(0)
        
        # Berechnung in Batches um Memory zu sparen (Logik aus deinem Code übernommen)
        split_num = len(self.temperatures)
        number_temp = 0
        
        for k in range(split_num):
            if(k==split_num-1):
                split_tempratures = self.temperatures[int(k*len(self.temperatures)/split_num):]
            else:
                split_tempratures = self.temperatures[int(k*len(self.temperatures)/split_num):int((k+1)*len(self.temperatures)/split_num)]
            
            # Eq 4: log_p / tau
            temps_tensor = torch.tensor(split_tempratures, device=self.device).view(-1, 1, 1)
            
            # Renormalisierung (log_softmax), da sich die Summe geändert hat
            new_log_probs = F.log_softmax(log_probs_expanded / temps_tensor, dim=-1)

            # Input IDs expandieren für gather
            if(number_temp != len(split_tempratures)):
                number_temp = len(split_tempratures)
                input_ids_expanded = self.input_ids.expand(number_temp, -1).unsqueeze(-1)

            # Extrahiere Log-Probs der Ziel-Token (TSP)
            new_token_log_probs = new_log_probs.gather(dim=-1, index=input_ids_expanded).squeeze(-1)
            log_probs_temprature.extend(new_token_log_probs.tolist())

            # NormAC (Eq 8 & 9)
            # new_log_probs.exp() ist die Wahrscheinlichkeit p(z|x, tau)
            mu = (new_log_probs.exp() * new_log_probs).sum(-1, keepdim=True) # Eq 8
            # Eq 9: E[(log_p - mu)^2]
            sigma = (new_log_probs.exp() * torch.square(new_log_probs - mu)).sum(-1).sqrt()

            mu_temprature.extend(mu.squeeze(-1).tolist())
            sigma_temprature.extend(sigma.tolist())
        
        return log_probs_temprature, mu_temprature, sigma_temprature
    
    def predict(self) -> Dict[str, list]:
        """
                    Compute AC/DerivAC/NormAC scores across temperatures using first-occurrence mask.
        
                    Returns
                    -------
                    dict
                        Mapping score-name -> list of aggregated values per temperature.
        
        """
        log_probs_temprature, mu_temprature, sigma_temprature = self.temperature_scaling()
        # temps_tsp becomes shape [num_temperatures, seq_len]
        temps_tsp = np.array(log_probs_temprature)
        mu_temps = np.array(mu_temprature)
        sigma_temps = np.array(sigma_temprature)
        
        # Original Token Log Probs (als Numpy) -> Shape [seq_len]
        orig_log_probs = self.token_log_probs.cpu().numpy()
        
        scores = defaultdict(list)
        
        # fos_mask is now 1D -> Shape [seq_len]
        fos_mask = self.get_fos_mask().cpu().numpy()
        
        # Loop über Temperaturen
        for i, tau in enumerate(self.temperatures):
            
            # --- DerivAC (Eq 6) ---
            # (TSP(tau+delta) - TSP(tau)). 
            if i > 0:
                deriv_vals = temps_tsp[i] - temps_tsp[i-1]
                # Indexing 1D array with 1D mask works
                scores[f"DerivAC_{tau}"].append(np.mean(deriv_vals[fos_mask]).item())

            # --- AC (Eq 5) ---
            # sgn(1 - tau) * (log_TSP - log_p)
            diff = temps_tsp[i] - orig_log_probs
            
            if tau > 1.0:
                term = -diff
            elif tau < 1.0:
                term = diff
            else:
                term = diff 
            
            scores[f"AC_{tau}"].append(np.mean(term[fos_mask]).item())

            # --- NormAC (Eq 7) ---
            # (log_TSP - mu) / sigma
            if sigma_temps[i].any():
                normalized = (temps_tsp[i] - mu_temps[i]) / (sigma_temps[i] + 1e-10)
                scores[f"NormAC_{tau}"].append(np.mean(normalized[fos_mask]).item())
                
        return scores

def inference(text: str, model: torch.nn.Module, tokenizer: Any, negative_prefix: List[torch.Tensor], member_prefix: List[torch.Tensor], non_member_prefix: List[torch.Tensor], device: torch.device, rel_attacks: RelativeLikelihood, dcpdd: DCPDD, offline_rmia: OfflineRobustMIA, noisy_attack: NoisyNeighbour, tagtab_attack: TagTab, cimia_attack: CIMIA) -> Optional[Dict[str, Any]]:
    """
                Run the full set of inference-based membership and uncertainty metrics for a single text.
    
                Parameters
                ----------
                text : str
                    Input text to evaluate.
                model : torch.nn.Module
                    Target model used for core loss/probability computations.
                tokenizer : Any
                    Tokenizer for the target model.
                negative_prefix : list of torch.Tensor
                    Negative prefix tensors used for recall-like metrics.
                member_prefix : list of torch.Tensor
                    Prefixes corresponding to member class shots.
                non_member_prefix : list of torch.Tensor
                    Prefixes corresponding to non-member class shots.
                device : torch.device
                    Device where models and tensors live.
                rel_attacks : RelativeLikelihood
                    Relative-likelihood helper instance.
                dcpdd : DCPDD
                    DCPDD scorer instance.
                offline_rmia : OfflineRobustMIA
                    Offline robust MIA instance.
                noisy_attack : NoisyNeighbour
                    Noisy neighbour attack instance.
                tagtab_attack : TagTab
                    TagTab scoring instance.
                cimia_attack : CIMIA
                    CIMIA aggregator instance.
    
                Returns
                -------
                dict or None
                    Dictionary of metric names to values, or None if the input produced no tokens.
    
    """
    
    data = raw_values(text, model, tokenizer, device)
    
    if data['input_ids'].numel() == 0:
        return None
    
    base_loss = data['loss'].item()
    
    data_lower = raw_values(text.lower(), model, tokenizer, device)

    input_ids_1d = data['input_ids'].squeeze(0).long()
    token_probs_1d = data['token_probs'].view(-1)
    
    base_attacks = Baseline(
        logits=data['logits'], 
        input_ids=data['input_ids'], 
        token_log_probs=data['token_log_probs']
    )
    max_renyi = MaxRenyi(
        token_probs=data['token_probs'], 
        full_log_probs=data['full_log_probs'], 
        full_token_probs=data['full_token_probs']
    )
    
    auto_calibration = ACMIA(
        device=device, 
        logits=data['logits'], 
        probs=data['full_token_probs'], 
        log_probs=data['full_log_probs'], 
        token_log_probs=data['token_log_probs'], 
        input_ids=data['input_ids'], 
        temperatures=np.concatenate([
            np.arange(0.1, 1.0, 0.2), 
            np.arange(1.0, 3.0, 0.5) 
        ])
    )
    
    pred = {
        **max_renyi.predict(), 
        'ppl': -math.exp(base_loss), 
        'ppl/lowercase_ppl': -(base_loss / (data_lower['loss'].item() + 1e-9)),
        'ppl/zlib': -(base_loss / zlib_entropy(text)),
        'ranks': -base_attacks.ranks(), 
        'dcpdd': dcpdd.predict(token_probs=token_probs_1d, input_ids=input_ids_1d),
        'tagtab': tagtab_attack.predict(text=text),
        'cimia': -cimia_attack.predict(loss=base_loss, input_ids=input_ids_1d, token_log_probs=data['token_log_probs']),
        'acmia': auto_calibration.predict()
    }
    
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:  
        pred[f"Min_ratio_{int(ratio*100)}"] = base_attacks.min_k(ratio=ratio)
        pred[f"Min_++_ratio_{int(ratio*100)}"] = base_attacks.min_k_plus_plus(ratio=ratio)
    
    # del data_lower
    # del data['logits']           
    # del data['full_log_probs']   
    # del data['full_token_probs'] 
    # del base_attacks             
    # del max_renyi                
    
    pred['noisy_neighbourhood'] = -noisy_attack.predict(
        input_ids=data['input_ids'], 
        base_loss=base_loss, 
        num_of_neighbour=20
    )
    
    pred['rmia'] = offline_rmia.predict(target_text=text)
    
    if input_ids_1d.numel() <= MODEL_MAX_LENGTH:
        pred['recall'] = -rel_attacks.calc_recall(text, base_loss, negative_prefix)
        pred.update(rel_attacks.calc_conrecall(text, base_loss, member_prefix, non_member_prefix))

    return pred

# Mostly copied from https://github.com/mireshghallah/neighborhood-curvature-mia/tree/main
# class NeighbourhoodComparison:
#     """
#     Implementation of the paper "Membership Inference Attacks against Language Models via Neighbourhood Comparison" by Mattern et al. that generate neighboring (similar) 
#     sentences for a given target sentence using Masked Language Modeling. Then calculates the average loss over those neighboring sentences and compares it against the loss of the original target sentence.
            
#     :param target_model: target model used to calculate the loss 
#     :type target_model: Any
#     :param target_tokenizer: tokenizer used by the target model
#     :type target_tokenizer: Any
#     :param search_model: masked language model used to generate neighbouring (similar) sentences
#     :type search_model: Any
#     :param search_tokenizer: tokenizer used by the search model
#     :type search_tokenizer: Any
#     :param device: device to run the model on (usually cuda:0)
#     :type device: Any
#     """
#     def __init__(self, target_model, target_tokenizer, search_model, search_tokenizer, device):
#         self.target_model = target_model
#         self.target_tokenizer = target_tokenizer
#         self.search_model = search_model
#         self.search_tokenizer = search_tokenizer
#         self.device = device
#         self.search_model = self.search_model.to(self.device)
        
#     def generate_neighbours(self, text:str, num_neighbours:int = 48, dropout_prob:float = 0.7):
#         text_tokenized = self.search_tokenizer(text, padding = True, truncation = True, pad_to_multiple_of=64, max_length = MODEL_MAX_LENGTH, return_tensors='pt').input_ids.to(self.device)
#         # original_text = self.search_tokenizer.batch_decode(text_tokenized)[0]

#         replacements = dict()
#         token_dropout = torch.nn.Dropout(p=dropout_prob)
#         with torch.no_grad():
#             base_embeds = self.search_model.bert.embeddings(text_tokenized).to(self.device)
            
#         for target_token_index in list(range(len(text_tokenized[0,:])))[1:]:
            
#             target_token = text_tokenized[0,target_token_index]

#             embeds = torch.cat((base_embeds[:,:target_token_index,:], token_dropout(base_embeds[:,target_token_index,:]).unsqueeze(dim=0), base_embeds[:,target_token_index+1:,:]), dim=1)
            
#             token_probs = torch.softmax(self.search_model(inputs_embeds=embeds).logits, dim=2)

#             original_prob = token_probs[0,target_token_index, target_token]

#             top_probabilities, top_candidates = torch.topk(token_probs[:,target_token_index,:], 6, dim=1)

#             for cand, prob in zip(top_candidates[0], top_probabilities[0]):
#                 if not cand == target_token:
#                     if original_prob.item() == 1:
#                         replacements[(target_token_index, cand)] = prob.item()/(1-0.9)
#                     else:
#                         replacements[(target_token_index, cand)] = prob.item()/(1-original_prob.item())
        
#         #highest_scored_texts = max(candidate_scores.iteritems(), key=operator.itemgetter(1))[:100]
#         # highest_scored_texts = nlargest(100, candidate_scores, key = candidate_scores.get)

#         replacement_keys = nlargest(num_neighbours, replacements, key=replacements.get)
#         replacements_new = dict()
#         for rk in replacement_keys:
#             replacements_new[rk] = replacements[rk]
#         replacements = replacements_new

#         highest_scored = nlargest(100, replacements, key=replacements.get)

#         texts = []
#         for single in highest_scored:
#             alt = text_tokenized
#             target_token_index, cand = single
#             alt = torch.cat((alt[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to(self.device), alt[:,target_token_index+1:]), dim=1)
#             alt_text = self.search_tokenizer.batch_decode(alt)[0]
#             texts.append(alt_text) #(alt_text, replacements[single])
#         return texts

#     def get_logprob(self, text):
#         text_tokenized = self.target_tokenizer(text, truncation = True, max_length = MODEL_MAX_LENGTH, return_tensors='pt').input_ids.to(self.device)
#         with torch.no_grad():
#             logprob = - self.target_model(text_tokenized, labels=text_tokenized).loss.item()
#         return logprob

#     def neighbourhood(self, text:str):
#         self.target_model.eval()
#         self.search_model.eval()
        
#         neighbor_loss = 0
        
#         # Generate neighbouring text and clean them
#         neighbours = self.generate_neighbours(text)
#         cleaned_neighbours = list(map(lambda s: s.replace(" [SEP]", " ").replace("[CLS] ", " "), neighbours))
        
#         # Iterate over neighbouring text and calculate log_probabilities 
#         for neighbour_text in cleaned_neighbours:
#             neighbor_loss += self.get_logprob(neighbour_text)
        
#         tok_orig = self.search_tokenizer(text, padding = True, truncation = True, max_length = MODEL_MAX_LENGTH, return_tensors='pt').input_ids.to(self.device)
#         orig_dec = self.search_tokenizer.batch_decode(tok_orig)[0].replace(" [SEP]", " ").replace("[CLS] ", " ")
#         original_loss = self.get_logprob(orig_dec)
        
#         # There must be a threshold (gamma) here, but by aggregating the scores we "learn" the optimal thresholds anyways. 
#         # The output would be usually a boolean and not float. Here is the code if we would use the threshold:
#         # return (original_loss-(neighbor_loss/len(neighbours))) < gamma # Returns true if the difference would be smaller than gamma, false if bigger
#         return original_loss-(neighbor_loss/len(cleaned_neighbours))
