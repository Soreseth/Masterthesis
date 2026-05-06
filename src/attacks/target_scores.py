import torch
from torch import nanmean
import numpy as np
import zlib
import math
import random 
import torch.nn.functional as F
from collections import defaultdict
import os
import re
import string
import json
from wordfreq import word_frequency

MODEL_MAX_LENGTH = 2048

# Maybe remove later, but for now keep it to save some precious VRAM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def fix_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def raw_values(sentence: str, model, tokenizer, device) -> dict:
    """
    Used to calculate the cross-entropy and probabilities of tokens for a SINGLE sentence.
    We compute the raw values only once and re-use them in other Attacks to save computation time.

    Args:
        sentence: The input text to process.
        model: HuggingFace causal language model.
        tokenizer: Tokenizer corresponding to the model.
        device: Torch device to run inference on.

    Returns:
        dict with keys: loss, logits, probs, log_probs, token_log_probs, input_ids.
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
    
    logits_sq = outputs.logits[..., :-1, :].contiguous().squeeze(0)
    labels_sq = encodings.input_ids[..., 1:].contiguous().squeeze(0)

    loss = torch.nn.CrossEntropyLoss(reduction='mean')(logits_sq, labels_sq)

    log_probs = F.log_softmax(logits_sq, dim=-1)
    probs = F.softmax(logits_sq, dim=-1)

    token_log_probs = log_probs.gather(dim=-1, index=labels_sq.unsqueeze(-1)).squeeze(-1)
    token_probs = probs.gather(dim=-1, index=labels_sq.unsqueeze(-1)).squeeze(-1)

    return {
        "loss": loss,
        "token_probs": token_probs,
        "token_log_probs": token_log_probs,
        "logits": logits_sq.unsqueeze(0), 
        "input_ids": labels_sq.unsqueeze(0).long(), 
        "full_token_probs": probs,
        "full_log_probs": log_probs
    }

def perplexity(loss: float):
    return math.exp(loss)

def zlib_entropy(sentence: str):
    return len(zlib.compress(bytes(sentence, 'utf-8')))

class Baseline:
    def __init__(self, logits: torch.Tensor , input_ids: torch.Tensor, token_log_probs: torch.Tensor):
        """
        Args:
            logits: Model output logits [1, seq_len, vocab_size].
            input_ids: Token IDs of the input [1, seq_len].
            token_log_probs: Per-token log probabilities [seq_len].
        """
        self.logits = logits
        self.token_log_probs = token_log_probs
        self.input_ids = input_ids
        
    def min_k(self, ratio:float = 0.05) -> torch.Tensor:
        """
        Mean of the smallest k token log-probs.

        Args:
            ratio: Fraction of tokens to select (default 0.05).
        """
        k_length = max(1, int(len(self.token_log_probs)*ratio))
        topk, _ = torch.topk(self.token_log_probs, k_length, largest=False)
        return nanmean(topk).item()

    def min_k_plus_plus(self, ratio:float = 0.05) -> torch.Tensor:
        """
        Min-K++: normalized version of Min-K using per-token mean and variance.

        Args:
            ratio: Fraction of tokens to select (default 0.05).
        """
        probs = F.softmax(self.logits[0], dim=-1)
        log_probs = F.log_softmax(self.logits[0], dim=-1)

        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

        mink_plus = (self.token_log_probs - mu) / (torch.clamp(sigma, min=0).sqrt() + 1e-10)
        k_length = max(1, int(len(mink_plus) * ratio))
        topk, _ = torch.topk(mink_plus, k_length, largest=False)
        return nanmean(topk).item()

    def ranks(self) -> torch.Tensor:
        """
        Mean rank of the true token in the logit distribution.
        """
        logits = self.logits
        labels = self.input_ids if self.input_ids.dim() == 2 else self.input_ids.squeeze(0)
        
        target_logits = logits.gather(-1, labels.unsqueeze(-1))
        ranks = (logits > target_logits).sum(dim=-1).float() + 1

        return torch.nanmean(ranks).item()

class RelativeLikelihood:
    def __init__(self, base_model, base_tokenizer, device):
        """
        Args:
            base_model: HuggingFace causal language model used as the target.
            base_tokenizer: Tokenizer corresponding to base_model.
            device: Torch device for inference.
        """
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.device = device
        self.max_length = getattr(self.base_model.config, "max_position_embeddings", 2048)

    """
    Since the Pythia models have a small context window (2048 tokens) and the ConRecall Paper recommends at least 12 shots (i.e. number of prefixes), 
    in some cases the target text + prefixe won't fit in the context window. In such cases we split the target text into smaller chunks and concatenate each 
    chunk with each prefix and then calculate the average likelihood over each of this combinations. For large texts this lead to a explosion of combinations 
    (e.g. 7 chunks * 7 shots = 49 combinations), and this approach won't approximate the effective surprisement of a super long prefixes before the target text.
    
    """
    def _get_smart_chunks(self, input_ids: torch.Tensor, num_chunks: int):
        """
        Splits input_ids into 'num_chunks' segments, ensuring each segment 
        ends on a sentence boundary (., ?, !, or newline) if possible to preserve semantics 
        and increase the surprise effect. 
        """
        total_len = input_ids.shape[1]
        max_chunk_size = math.ceil(total_len / num_chunks)
        
        chunks = []
        start_idx = 0
        
        for i in range(num_chunks):
            if i == num_chunks - 1:
                chunks.append(input_ids[:, start_idx:])
                break
                
            end_idx = min(start_idx + max_chunk_size, total_len)
            chunk_ids = input_ids[0, start_idx:end_idx]
            chunk_text = self.base_tokenizer.decode(chunk_ids, skip_special_tokens=True)

            search_start_char = len(chunk_text) // 2
            match = None
            for m in re.finditer(r'[.!?\n]', chunk_text):
                if m.end() > search_start_char:
                    match = m
            
            if match:
                split_char_idx = match.end()
                valid_text = chunk_text[:split_char_idx]
                valid_tokens = self.base_tokenizer(
                    valid_text, 
                    add_special_tokens=False
                )['input_ids']
                split_point = start_idx + len(valid_tokens)
            else:
                split_point = end_idx

            chunks.append(input_ids[:, start_idx:split_point])
            start_idx = split_point
            
        return chunks

    def _get_single_loss(self, prefix_list: list, target: str) -> float:
        """Calculates conditional loss for prefix(es) + target.
        If all shots fit, uses a single pass. Otherwise pairs each shot with target chunks.
        """
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

        target_len = target_ids.shape[1]

        if not prefix_list or all(p.numel() == 0 for p in prefix_list):
            return self._compute_chunk_losses(torch.tensor([[]], dtype=torch.long).to(self.device), target_ids)

        concat_prefix = torch.cat(prefix_list, dim=1).to(self.device)
        total_prefix_len = concat_prefix.shape[1]

        if total_prefix_len + target_len <= self.max_length:
            return self._compute_chunk_losses(concat_prefix, target_ids)

        losses = []
        for shot in prefix_list:
            shot = shot.to(self.device)
            shot_len = shot.shape[1]

            max_shot_len = self.max_length // 2
            if shot_len > max_shot_len:
                shot = shot[:, -max_shot_len:]

            shot_losses = self._compute_chunk_losses(shot, target_ids)
            if shot_losses != 0.0:
                losses.append(shot_losses)

        if not losses:
            return 0.0
        return sum(losses) / len(losses)

    def _compute_chunk_losses(self, prefix: torch.Tensor, target_ids: torch.Tensor) -> float:
        """Compute average loss for prefix + target, splitting target into chunks if needed."""
        prefix_len = prefix.shape[1] if prefix.numel() > 0 else 0
        available_for_target = self.max_length - prefix_len

        if available_for_target <= 0:
            return 0.0

        target_len = target_ids.shape[1]

        if target_len <= available_for_target:
            num_chunks = 1
        else:
            num_chunks = math.ceil(target_len / available_for_target)

        target_chunks = self._get_smart_chunks(target_ids, num_chunks)

        losses = []
        for target_chunk in target_chunks:
            if target_chunk.numel() == 0:
                continue

            if prefix_len > 0:
                input_ids = torch.cat((prefix, target_chunk), dim=1)
            else:
                input_ids = target_chunk

            if input_ids.shape[1] > self.max_length:
                input_ids = input_ids[:, :self.max_length]

            labels = input_ids.clone()
            if prefix_len > 0:
                labels[:, :prefix_len] = -100

            with torch.no_grad():
                outputs = self.base_model(input_ids, labels=labels)
                losses.append(outputs.loss.item())

        if not losses:
            return 0.0
        return sum(losses) / len(losses)

    def calc_recall(self, text: str, base_loss: float, negative_prefix_list: list) -> float:
        """Compute Recall score: conditional loss / base loss.

        Args:
            text: Target text to evaluate.
            base_loss: Pre-computed unconditional loss of the text.
            negative_prefix_list: List of non-member prefix strings.
        """
        cond_loss = self._get_single_loss(negative_prefix_list or [], text)

        if base_loss == 0:
            return 0.0
        return cond_loss / base_loss

    def calc_conrecall(self, text: str, base_loss: float, member_prefix_list: list, non_member_prefix_list: list) -> dict:
        """Compute ConRecall scores for multiple gamma values.

        Args:
            text: Target text to evaluate.
            base_loss: Pre-computed unconditional loss of the text.
            member_prefix_list: List of member prefix strings.
            non_member_prefix_list: List of non-member prefix strings.
        """
        cond_loss_mem = self._get_single_loss(member_prefix_list or [], text)
        cond_loss_non = self._get_single_loss(non_member_prefix_list or [], text)
        scores = {}

        # ConRecall: (LL(x|D_out) - γ·LL(x|D_in)) / LL(x)
        for gamma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            val = 0.0
            if base_loss != 0:
                val = (gamma * cond_loss_mem - cond_loss_non) / base_loss
            scores[f"conrecall_gamma_{int(100*gamma)}"] = val
        return scores

    def calc_recall_multi(self, text: str, base_loss: float, negative_prefix_list: list) -> dict:
        """Compute Recall at each shot count 1..len(prefix_list), saving intermediates."""
        scores = {}
        if base_loss == 0 or not negative_prefix_list:
            for k in range(1, len(negative_prefix_list) + 1):
                scores[f"recall_s{k}"] = 0.0
            return scores

        for k in range(1, len(negative_prefix_list) + 1):
            cond_loss = self._get_single_loss(negative_prefix_list[:k], text)
            scores[f"recall_s{k}"] = cond_loss / base_loss
        return scores

    def calc_conrecall_multi(self, text: str, base_loss: float,
                             member_prefix_list: list, non_member_prefix_list: list) -> dict:
        """Compute ConRecall at each shot count 1..N, saving intermediates for all gammas."""
        scores = {}
        n_shots = min(len(member_prefix_list), len(non_member_prefix_list))

        if base_loss == 0 or n_shots == 0:
            for k in range(1, n_shots + 1):
                for gamma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    scores[f"conrecall_s{k}_gamma_{int(100*gamma)}"] = 0.0
            return scores

        for k in range(1, n_shots + 1):
            cond_loss_mem = self._get_single_loss(member_prefix_list[:k], text)
            cond_loss_non = self._get_single_loss(non_member_prefix_list[:k], text)
            for gamma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                val = (gamma * cond_loss_mem - cond_loss_non) / base_loss
                scores[f"conrecall_s{k}_gamma_{int(100*gamma)}"] = val
        return scores

class TagTab:
    def __init__(self, target_model, target_tokenizer, top_k:int, nlp, device, entropy_map:dict = None, min_sentence_len:int = 7, max_sentence_len:int = 40):
        """
        Args:
            target_model: Model to evaluate
            target_tokenizer: Tokenizer for target model
            top_k: Maximum k value to evaluate (will evaluate k=1 to top_k-1)
            nlp: spaCy NLP model for sentence segmentation and NER
            device: Device to run on
            entropy_map: Pre-computed entropy map (dict: word -> entropy). If None, computes on-the-fly.
            min_sentence_len: Minimum sentence length in words (default: 7 parameter from paper)
            max_sentence_len: Maximum sentence length in words (default: 40 parameter from paper)
        """
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.top_k = top_k
        self.device = device
        self.min_sentence_len = min_sentence_len
        self.max_sentence_len = max_sentence_len
        self.nlp = nlp
        self.entropy_map = entropy_map
        self.punc_table = str.maketrans("", "", string.punctuation)

    def _normalize_token(self, w: str) -> str:
        w = w.lower().translate(self.punc_table)
        w = re.sub(r"\s+", "", w)
        return w

    def get_tab_keywords(self, text:str):
        """
        Extract keywords for multiple k values using low-entropy words + NER.
        Matches original: for each sentence, union low-entropy + NER normalized words,
        then re-sort by entropy to get bottom_k_words. Per-k filtering uses entropy rank.

        Returns:
            List of tuples per sentence: (bottom_k_words, entropy_rank_map)
            where bottom_k_words is the sorted list of normalized keyword strings,
            and entropy_rank_map maps word -> rank index (for per-k filtering).
        """
        docs = self.nlp(text)
        result = []

        for sent_span in docs.sents:
            original_tokens = [t.text for t in sent_span]
            norm_tokens = [self._normalize_token(t.text) for t in sent_span]
            valid_indices = [i for i, t in enumerate(norm_tokens) if t]

            sentence_word_count = len(original_tokens)
            if sentence_word_count < self.min_sentence_len or sentence_word_count > self.max_sentence_len:
                continue

            word_entropies = []
            for i in valid_indices:
                word = norm_tokens[i]
                if self.entropy_map is not None:
                    ent = self.entropy_map.get(word, float('inf'))
                else:
                    p = word_frequency(word, 'en')
                    if p > 0.0:
                        ent = -p * math.log2(p)
                    else:
                        ent = float('inf')
                word_entropies.append((ent, word))

            word_entropies.sort(key=lambda x: x[0])
            low_entropy_words = set(w for _, w in word_entropies[:self.top_k - 1])

            ner_tokens = set()
            sent_doc = self.nlp(sent_span.text)
            for ent in sent_doc.ents:
                for piece in ent.text.split():
                    tok = self._normalize_token(piece)
                    if tok:
                        ner_tokens.add(tok)

            unique_words = list(low_entropy_words.union(ner_tokens))
            bottom_k_words = sorted(
                [w for w in unique_words if w],
                key=lambda w: self.entropy_map.get(w, float('inf')) if self.entropy_map is not None
                    else (-word_frequency(w, 'en') * math.log2(word_frequency(w, 'en')) if word_frequency(w, 'en') > 0 else float('inf'))
            )[:self.top_k - 1]

            result.append(bottom_k_words)

        return result

    def get_tab_score(self, text: str, sentence_keywords_list: list,
                      full_log_probs: torch.Tensor = None,
                      shifted_input_ids: torch.Tensor = None):
        """
        Calculate TagTab scores for bottom k values.
        full normalization during matching, first-occurrence search.

        Args:
            full_log_probs: Pre-computed log_softmax(logits) [seq_len-1, vocab].
                            If provided with shifted_input_ids, skips the forward pass.
            shifted_input_ids: Pre-computed shifted input_ids [1, seq_len-1].
        """
        raw_tokens = self.target_tokenizer.tokenize(text)

        if full_log_probs is not None and shifted_input_ids is not None:
            input_ids_processed = shifted_input_ids.squeeze(0)
            all_prob = []
            for i, token_id in enumerate(input_ids_processed):
                all_prob.append(full_log_probs[i, token_id].item())
        else:
            input_ids_list = self.target_tokenizer.encode(text)
            input_ids = torch.tensor(input_ids_list).unsqueeze(0).to(self.device).long()

            with torch.no_grad():
                outputs = self.target_model(input_ids, labels=input_ids)

            log_probs_full = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
            input_ids_processed = input_ids[0][1:]
            all_prob = []
            for i, token_id in enumerate(input_ids_processed):
                all_prob.append(log_probs_full[0, i, token_id].item())

        decoded_tokens = [t.replace("▁", "") for t in raw_tokens]
        norm_tokens = [re.sub(r"\s+", "", token.lower().translate(self.punc_table)) for token in decoded_tokens]
        concatenated_tokens = "".join(norm_tokens)
        intermediate_at = []
        intermediate_ft = []

        for bottom_k_words in sentence_keywords_list:
            for i, word in enumerate(bottom_k_words):
                if word in concatenated_tokens:
                    start_index = concatenated_tokens.find(word)
                    end_index = start_index + len(word)

                    start_token_index = None
                    end_token_index = None
                    current_length = 0
                    for j, token in enumerate(norm_tokens):
                        current_length += len(token)
                        if current_length > start_index and start_token_index is None:
                            start_token_index = j
                        if current_length >= end_index:
                            end_token_index = j
                            break

                    if start_token_index is not None and end_token_index is not None:
                        if start_token_index < len(all_prob):
                            intermediate_ft.append((i, all_prob[start_token_index]))
                        for idx in range(start_token_index, end_token_index + 1):
                            if idx < len(all_prob):
                                intermediate_at.append((i, all_prob[idx]))

        scores_by_k = {}
        for k in range(1, self.top_k):
            relevant_at = [val for rank, val in intermediate_at if rank < k]
            relevant_ft = [val for rank, val in intermediate_ft if rank < k]

            at_score = np.mean(relevant_at).item() if relevant_at else None
            ft_score = np.mean(relevant_ft).item() if relevant_ft else None

            scores_by_k[k] = (at_score, ft_score)

        return scores_by_k

    def predict(self, text: str, full_log_probs: torch.Tensor = None,
                shifted_input_ids: torch.Tensor = None):
        """
        Predict TagTab scores for multiple k values.

        Args:
            full_log_probs: Pre-computed log_softmax(logits) [seq_len-1, vocab] from raw_values().
            shifted_input_ids: Pre-computed shifted input_ids [1, seq_len-1] from raw_values().

        Returns:
            Dictionary with keys like 'tag_tab_AT_k=1', 'tag_tab_FT_k=1', etc.
            AT = All Tokens, FT = First Token only.
            Keys are omitted (not set to 0) when no keywords match for that k.
        """
        keyword_batches = self.get_tab_keywords(text)
        if not keyword_batches:
            return {}

        scores_by_k = self.get_tab_score(text, keyword_batches,
                                         full_log_probs=full_log_probs,
                                         shifted_input_ids=shifted_input_ids)

        result = {}
        for k, (at_score, ft_score) in scores_by_k.items():
            if at_score is not None:
                result[f"tag_tab_AT_k={k}"] = at_score
            if ft_score is not None:
                result[f"tag_tab_FT_k={k}"] = ft_score

        return result

class NoisyNeighbour:
    """
    NoisyNeighbour is a white-box attack, which operates on the embedding level of the model. 
    It adds gaussian noise to the embeddings and calculates the difference between the average loss of the noisy embeddings and the original embeddings. 
    """
    
    def __init__(self, model, device, batch_size: int = 4):
        """
        Args:
            model: HuggingFace causal language model.
            device: Torch device for inference.
            batch_size: Number of noisy copies per forward pass (default 4).
        """
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.dtype = model.dtype

    def predict(self, input_ids: torch.Tensor, sigma: float = 0.02, base_loss: float = None, num_of_neighbour: int = 20) -> float:
        """Compute NoisyNeighbour score for a single sigma and neighbour count.

        Args:
            input_ids: Tokenized input [1, seq_len].
            sigma: Std of Gaussian noise added to embeddings (default 0.02).
            base_loss: Pre-computed loss of the original input (optional).
            num_of_neighbour: Number of noisy copies (must be divisible by batch_size).

        Returns:
            Difference between original loss and average noisy loss.
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

            input_embeddings = embedding_layer(token_ids).to(dtype=self.dtype, device=self.device)
            seq_len = input_embeddings.shape[1]
            hidden_dim = input_embeddings.shape[2]

            batch_labels = token_ids.repeat(self.batch_size, 1)
            num_batches = num_of_neighbour // self.batch_size

            for _ in range(num_batches):
                noise = torch.randn(
                    (self.batch_size, seq_len, hidden_dim),
                    device=self.device,
                    dtype=self.dtype
                ) * sigma

                noisy_embeddings = torch.add(input_embeddings, noise, alpha=1)

                neighbour_loss += self.model(inputs_embeds=noisy_embeddings, labels=batch_labels).loss.item()
        avg_neighbor_loss = neighbour_loss / num_batches
        return original_loss - avg_neighbor_loss

    def predict_multi(self, input_ids: torch.Tensor, base_loss: float = None,
                      sigmas: list = None, max_neighbours: int = 30,
                      checkpoints: list = None) -> dict:
        """Compute NoisyNeighbour scores for multiple sigmas, saving intermediate
        results at checkpoint neighbour counts.

        Args:
            sigmas: List of sigma values (default: [0.1, 0.01, 0.001])
            max_neighbours: Total neighbours to compute (must be divisible by batch_size)
            checkpoints: Neighbour counts at which to save scores (default: [10, 20, 30])

        Returns:
            dict with keys like 'noisy_s0.01_n10', 'noisy_s0.01_n20', etc.
        """
        if sigmas is None:
            sigmas = [0.1, 0.01, 0.001]
        if checkpoints is None:
            checkpoints = [10, 20, 30]

        if max_neighbours % self.batch_size != 0:
            print(f"ERROR: max_neighbours={max_neighbours} not divisible by batch_size={self.batch_size}")
            return {}

        token_ids = input_ids.to(self.device)
        embedding_layer = self.model.get_input_embeddings()

        with torch.no_grad():
            if base_loss is not None:
                original_loss = base_loss
            else:
                original_loss = self.model(input_ids=token_ids, labels=token_ids).loss.item()

            input_embeddings = embedding_layer(token_ids).to(dtype=self.dtype, device=self.device)
            seq_len = input_embeddings.shape[1]
            hidden_dim = input_embeddings.shape[2]
            batch_labels = token_ids.repeat(self.batch_size, 1)

        results = {}
        num_batches = max_neighbours // self.batch_size
        checkpoint_batches = {cp // self.batch_size: cp for cp in checkpoints
                              if cp % self.batch_size == 0 and cp <= max_neighbours}

        for sigma in sigmas:
            cumulative_loss = 0.0
            with torch.no_grad():
                for b in range(1, num_batches + 1):
                    noise = torch.randn(
                        (self.batch_size, seq_len, hidden_dim),
                        device=self.device, dtype=self.dtype
                    ) * sigma
                    noisy_embeddings = torch.add(input_embeddings, noise, alpha=1)
                    cumulative_loss += self.model(
                        inputs_embeds=noisy_embeddings, labels=batch_labels
                    ).loss.item()

                    if b in checkpoint_batches:
                        n = checkpoint_batches[b]
                        avg_loss = cumulative_loss / b
                        results[f'noisy_s{sigma}_n{n}'] = original_loss - avg_loss

        return results

class MaxRenyi:
    """
    Computes various Rényi entropy and modified cross-entropy metrics for MIA.
    The metrics capture model uncertainty and calibration for membership inference.
    Member texts typically have lower entropy (higher confidence).
    """
    
    def __init__(self, token_probs: torch.Tensor, full_log_probs: torch.Tensor, full_token_probs: torch.Tensor, epsilon: float = 1e-10):
        """
        Args:
            token_probs: Probabilities of the correct tokens (p_y) shape: [seq_len]
            full_log_probs: Log probabilities for full vocabulary, shape: [seq_len, vocab_size]
            full_token_probs: Probabilities for full vocabulary, shape: [seq_len, vocab_size]
            epsilon: Small constant for numerical stability (default: 1e-10)
        """
        self.token_probs = token_probs
        self.full_log_probs = full_log_probs
        self.full_token_probs = full_token_probs
        self.epsilon = epsilon
    
    def predict(self):
        """
        Compute all Rényi entropy and modified metrics.
        
        Returns:
            dict: Dictionary containing:
                - Standard metrics (mean and ratio aggregations):
                    * entropies: Shannon entropy (Rényi a=1)
                    * renyi_05: Rényi entropy a=0.5
                    * renyi_2: Rényi entropy a=2 (collision entropy)
                    * renyi_inf: Rényi entropy a=infty (min-entropy)
                    * gap_prob: Confidence gap between top-2 predictions
                
                - Custom metrics (mean only):
                    * modified_entropies: Custom calibration-aware entropy
                    * mod_renyi_05: Modified Rényi a=0.5
                    * mod_renyi_2: Modified Rényi a=2
                
                Each standard metric has:
                    - {metric}_mean: Negated average over all tokens
                    - {metric}_ratio_{5,10,20,30,40,50,60}: Negated average of top-K% tokens
        
        Notes:
            - All scores are negated for MIA (higher = more member-like)
            - Ratio scores use Max-K% selection (highest entropy tokens)
            - Member texts typically have lower entropy -> higher scores after negation
        """
        
        if self.token_probs.numel() == 0:
            dummy = {}
            dummy['gap_prob_mean'] = 0.0
            for name in ["entropies", "renyi_05", "renyi_2", "renyi_inf", "modified_entropies", "mod_renyi_05", "mod_renyi_2"]:
                 dummy[f"{name}_mean"] = 0.0
            for name in ["entropies", "renyi_05", "renyi_2", "renyi_inf"]:
                for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                    dummy[f"{name}_ratio_{int(ratio*100)}"] = 0.0
            return dummy

        token_probs = self.token_probs.to(dtype=torch.float64)
        full_log_probs = self.full_log_probs.to(dtype=torch.float64)
        full_token_probs = self.full_token_probs.to(dtype=torch.float64)
        
        metrics = {}
        probs_safe = torch.clamp(full_token_probs, min=self.epsilon, max=1-self.epsilon)

        metrics["entropies"] = -(full_token_probs * full_log_probs).sum(dim=-1)

        metrics["renyi_05"] = 2*torch.log(torch.sum(torch.pow(probs_safe, 0.5), dim=-1))
        metrics["renyi_2"] =  -torch.log(torch.sum(torch.pow(probs_safe, 2), dim=-1))

        top2_vals, _ = torch.topk(full_log_probs, 2, dim=-1)
        max_p = top2_vals[:, 0]
        second_p = top2_vals[:, 1]
        
        metrics["renyi_inf"] = -max_p
        metrics["gap_prob"] = max_p - second_p

        # Modified Entropy: -(1-p_y)·log(p_y) - Σ_x p(x)·log(1-p(x)) + p_y·log(1-p_y)
        p_y = token_probs
        term_a = -(1 - p_y) * torch.log(p_y)
        term_b = -(full_token_probs * torch.log(1 - probs_safe)).sum(dim=-1)
        term_c = p_y * torch.log(1 - p_y)
        
        metrics["modified_entropies"] = term_a + term_b + term_c
        
        # Modified Rényi: -(1/k) · [(1-p_y)·p_y^k - (1-p_y) + Σ_{x≠y} p(x)·(1-p(x))^k - (1-p_y)], k=|1-a|
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
            tensor_val = torch.nan_to_num(tensor_val, nan=0.0, posinf=1e9, neginf=-1e9)

            if name == "gap_prob":
                results[f"{name}_mean"] = tensor_val.mean().item()
            else:
                results[f"{name}_mean"] = -tensor_val.mean().item()
            
            if name in ["entropies", "renyi_05", "renyi_2", "renyi_inf"]:
                for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                    k_len = max(1, int(len(tensor_val) * ratio))
                    
                    k_vals, _ = torch.topk(tensor_val, k_len, largest=True)
                    
                    results[f"{name}_ratio_{int(ratio*100)}"] = -k_vals.mean().item()
        
        return results
    
class DCPDD:
    def __init__(self, freq_dict, device, a=0.01, apply_smoothing=False):
        """
        DC-PDD (Divergence-based Calibration for Pretraining Data Detection)
        
        Args:
            freq_dict: Numpy array of token frequency counts (vocabulary size)
            device: Device to run on
            a: Hyperparameter for clipping cross-entropy values (default: 0.01 from paper)
            apply_smoothing: Whether to apply Laplace smoothing to frequency distribution
        """
        self.a = a
        self.device = device
        
        if apply_smoothing:
            freq_array = np.array(freq_dict, dtype=np.float32)
            smoothed_freq = (freq_array + 1) / (freq_array.sum() + len(freq_array))
            self.freq_tensor = torch.from_numpy(smoothed_freq).to(self.device)
        else:
            self.freq_tensor = torch.from_numpy(freq_dict).to(self.device)

    def predict_multi(self, token_probs: torch.Tensor, input_ids: torch.Tensor,
                      a_values: list = None) -> dict:
        """Calculate DC-PDD scores for multiple clipping thresholds.

        Args:
            a_values: List of clipping thresholds (default: [1.0, 0.1, 0.01, 0.001])

        Returns:
            dict with keys like 'dc_pdd_a1.0', 'dc_pdd_a0.01', etc.
        """
        if a_values is None:
            a_values = [1.0, 0.1, 0.01, 0.001]

        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze(0)
        if token_probs.dim() > 1:
            token_probs = token_probs.squeeze(0)

        if input_ids.shape[0] > token_probs.shape[0]:
            input_ids = input_ids[1:]
            input_ids = input_ids[:token_probs.shape[0]]

        ids_np = input_ids.detach().cpu().numpy()
        _, unique_indices_np = np.unique(ids_np, return_index=True)
        unique_indices = torch.from_numpy(unique_indices_np).long().to(self.device)

        x_pro = token_probs[unique_indices]
        valid_ids = input_ids[unique_indices]
        max_freq_id = len(self.freq_tensor) - 1
        valid_ids = torch.clip(valid_ids, 0, max_freq_id)

        x_fre = self.freq_tensor[valid_ids]
        x_fre = torch.where(x_fre == 0, 1e-10, x_fre)

        ce_raw = torch.mul(x_pro, torch.log(1 / x_fre))

        results = {}
        for a in a_values:
            ce = torch.clamp(ce_raw, max=a)
            results[f'dc_pdd_a{a}'] = float(nanmean(ce))
        return results

    def predict(self, token_probs:torch.Tensor, input_ids:torch.Tensor) -> float:
        """
        Calculate DC-PDD score using first occurrence of each token.

        Formula: -mean(min(x_prob * log(1/x_freq), a))
        where only first occurrence of each unique token is considered.
        """
        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze(0)
        if token_probs.dim() > 1:
            token_probs = token_probs.squeeze(0)

        if input_ids.shape[0] > token_probs.shape[0]:
            input_ids = input_ids[1:]
            input_ids = input_ids[:token_probs.shape[0]]

        ids_np = input_ids.detach().cpu().numpy()
        _, unique_indices_np = np.unique(ids_np, return_index=True)
        unique_indices = torch.from_numpy(unique_indices_np).long().to(self.device)

        x_pro = token_probs[unique_indices]
        valid_ids = input_ids[unique_indices]
        max_freq_id = len(self.freq_tensor) - 1
        valid_ids = torch.clip(valid_ids, 0, max_freq_id)

        x_fre = self.freq_tensor[valid_ids]
        x_fre = torch.where(x_fre == 0, 1e-10, x_fre)

        ce = torch.mul(x_pro, torch.log(1 / x_fre))
        ce = torch.clamp(ce, max=self.a)

        return float(nanmean(ce))

class CAMIA:
    """
    Context-Aware Membership Inference Attack.

    Computes ~72 membership inference signals matching the paper's full set:
    - Cut-off loss (f_Cut) at T'=T, 200, 300
    - Token diversity calibration (f_Cal) at T'=T, 200, 300
    - Perplexity (f_PPL) at T'=T, 200, 300
    - Calibrated perplexity (f_CalPPL) at T'=T, 200, 300
    - Count below threshold (f_CB) at T'=200 with tau=1,2,3
    - Count below mean (f_CBM) at T'=T, 200, 300
    - Count below previous mean (f_CBPM) at T'=T, 200, 300
    - Lempel-Ziv complexity (f_LZ) with bins=3,4,5
    - Slope (f_Slope) at T'=600, 800, 1000
    - Approximate entropy (f_ApEn) at T'=600, 800, 1000
    Plus Rep^1 ([X,X]) and Rep^2 ([X," ",X]) variants for Cut, Cal, PPL,
    CalPPL, CB, CBM, and LZ.
    """
    def __init__(self, target_model, target_tokenizer, device, max_len: int, calibration_signal):
        """
        Args:
            target_model: HuggingFace causal language model to attack.
            target_tokenizer: Tokenizer corresponding to target_model.
            device: Torch device for inference.
            max_len: Maximum token length for input sequences.
            calibration_signal: Pre-computed calibration signal for diversity calibration.
        """
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.device = device
        self.max_len = max_len
        self.calibration_signal = calibration_signal

    def _cut_off_loss(self, token_log_probs, T_prime):
        """f_Cut: mean of log probs for first T_prime tokens."""
        lp = token_log_probs[:T_prime]
        if lp.numel() == 0:
            return 0.0
        return lp.mean().item()

    def _diversity_calibration(self, input_ids, token_log_probs, T_prime):
        """f_Cal: mean(log_probs[:T']) / token_diversity."""
        cut_off = input_ids[:T_prime]
        if cut_off.numel() == 0:
            return 0.0
        d_X = torch.unique(cut_off).numel() / cut_off.numel()
        if d_X == 0:
            return 0.0
        return token_log_probs[:T_prime].mean().item() / d_X

    def _perplexity(self, token_log_probs, T_prime):
        """f_PPL: exp(-mean(log_probs[:T']))."""
        lp = token_log_probs[:T_prime]
        if lp.numel() == 0:
            return 0.0
        return math.exp(-lp.mean().item())

    def _cal_perplexity(self, input_ids, token_log_probs, T_prime):
        """f_CalPPL: perplexity / token_diversity."""
        lp = token_log_probs[:T_prime]
        if lp.numel() == 0:
            return 0.0
        ppl = math.exp(-lp.mean().item())
        cut_off = input_ids[:T_prime]
        d_X = torch.unique(cut_off).numel() / cut_off.numel()
        if d_X == 0:
            return 0.0
        return ppl / d_X

    def _slope_loss(self, token_log_probs, T_prime):
        """f_Slope: linear slope of log probs for first T_prime tokens."""
        lp = token_log_probs[:T_prime]
        if lp.numel() < 2:
            return 0.0
        n = lp.numel()
        t_index = torch.arange(n, dtype=torch.float32, device=self.device)
        t_avg = t_index.mean()
        y_avg = lp.mean()
        with torch.no_grad():
            numer = torch.sum((t_index - t_avg) * (lp - y_avg)).item()
            denom = torch.sum((t_index - t_avg) ** 2).item()
        if denom == 0:
            return 0.0
        return numer / denom

    def _count_below(self, token_log_probs, T_prime, thresholds):
        """f_CB: fraction of tokens with log_prob >= threshold."""
        lp = token_log_probs[:T_prime]
        if lp.numel() == 0:
            return [0.0] * len(thresholds)
        return [(lp >= t).float().mean().item() for t in thresholds]

    def _count_below_mean(self, token_log_probs, T_prime):
        """f_CBM: fraction of tokens above the sequence mean."""
        lp = token_log_probs[:T_prime]
        if lp.numel() == 0:
            return 0.0
        return (lp > lp.mean().item()).float().mean().item()

    def _count_below_prev_mean(self, token_log_probs, T_prime):
        """f_CBPM: fraction of tokens above the running mean."""
        lp = token_log_probs[:T_prime]
        if lp.numel() < 2:
            return 0.0
        cumsum = torch.cumsum(lp, dim=0)
        indices = torch.arange(1, lp.numel() + 1, device=self.device).float()
        running_means = cumsum / indices
        return (lp[1:] > running_means[:-1]).float().mean().item()

    def _lempel_ziv(self, token_log_probs, T_prime, bins):
        """f_LZ: Lempel-Ziv complexity with specified number of bins."""
        x = token_log_probs[:T_prime].detach().cpu().numpy()
        x = np.asarray(x)
        if len(x) == 0:
            return 0.0
        bins_edges = np.linspace(min(x), 0, 100)
        x = np.digitize(x, bins=bins_edges, right=True)
        bins_arr = np.linspace(np.min(x), np.max(x), bins + 1)[1:]
        sequence = np.searchsorted(bins_arr, x, side="left")
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

    def _approximate_entropy(self, token_log_probs, T_prime, m=8, r=0.8):
        """f_ApEn: approximate entropy of the loss sequence."""
        x = token_log_probs[:T_prime].detach().cpu().numpy()
        x = np.asarray(x)
        N = x.size
        r_val = r * np.std(x)
        if r_val <= 0 or N <= m + 1:
            return 0.0

        def _phi(m_val):
            x_re = np.array([x[i : i + m_val] for i in range(N - m_val + 1)])
            C = np.sum(
                np.max(np.abs(x_re[:, np.newaxis] - x_re[np.newaxis, :]), axis=2) <= r_val,
                axis=0,
            ) / (N - m_val + 1)
            return np.sum(np.log(C)) / (N - m_val + 1.0)

        return np.abs(_phi(m) - _phi(m + 1))

    def _get_rep_token_log_probs(self, input_ids, rep_type):
        """
        Forward pass on repeated input, return token_log_probs for the 2nd copy.

        rep_type=1: [X, X]       (paper f_Rep^1)
        rep_type=2: [X, " ", X]  (paper f_Rep^2)
        """
        vocab_size = self.target_model.config.vocab_size
        ids = torch.clamp(input_ids, max=vocab_size - 1)

        if rep_type == 1:
            safe_len = MODEL_MAX_LENGTH // 2
        else:
            space_id = self.target_tokenizer.encode(" ", add_special_tokens=False)
            safe_len = (MODEL_MAX_LENGTH - len(space_id)) // 2

        if ids.numel() > safe_len:
            ids = ids[:safe_len]

        if rep_type == 1:
            repeated = torch.cat([ids, ids])
            prefix_len = ids.numel()
        else:
            space_id = self.target_tokenizer.encode(" ", add_special_tokens=False)
            space_tensor = torch.tensor(space_id, device=self.device)
            repeated = torch.cat([ids, space_tensor, ids])
            prefix_len = ids.numel() + len(space_id)

        repeated_batch = repeated.unsqueeze(0)
        with torch.no_grad():
            outputs = self.target_model(repeated_batch)

        logits = outputs.logits[0]
        log_probs = F.log_softmax(logits, dim=-1)

        n_tokens = ids.numel()
        logit_positions = torch.arange(
            prefix_len - 1, prefix_len + n_tokens - 1, device=self.device
        )
        rep_token_log_probs = log_probs[logit_positions, ids[:n_tokens]]

        del logits, outputs, log_probs
        torch.cuda.empty_cache()

        return rep_token_log_probs.detach()

    def _compute_base_signals(self, input_ids, token_log_probs, T_full):
        """Compute all base (non-rep) signals. Returns dict."""
        signals = {}

        for tp, label in [(T_full, 'T'), (200, '200'), (300, '300')]:
            signals[f'cut_{label}'] = self._cut_off_loss(token_log_probs, tp)

        for tp, label in [(T_full, 'T'), (200, '200'), (300, '300')]:
            signals[f'cal_{label}'] = self._diversity_calibration(input_ids, token_log_probs, tp)

        for tp, label in [(T_full, 'T'), (200, '200'), (300, '300')]:
            signals[f'ppl_{label}'] = self._perplexity(token_log_probs, tp)

        for tp, label in [(T_full, 'T'), (200, '200'), (300, '300')]:
            signals[f'cal_ppl_{label}'] = self._cal_perplexity(input_ids, token_log_probs, tp)

        cb_vals = self._count_below(token_log_probs, 200, [-1.0, -2.0, -3.0])
        for i, tau in enumerate([1, 2, 3]):
            signals[f'cb_t{tau}'] = cb_vals[i]

        for tp, label in [(T_full, 'T'), (200, '200'), (300, '300')]:
            signals[f'cbm_{label}'] = self._count_below_mean(token_log_probs, tp)

        for tp, label in [(T_full, 'T'), (200, '200'), (300, '300')]:
            signals[f'cbpm_{label}'] = self._count_below_prev_mean(token_log_probs, tp)

        for bins in [3, 4, 5]:
            signals[f'lz_b{bins}'] = self._lempel_ziv(token_log_probs, T_full, bins)

        for tp in [600, 800, 1000]:
            signals[f'slope_{tp}'] = self._slope_loss(token_log_probs, tp)

        for tp in [600, 800, 1000]:
            signals[f'apen_{tp}'] = self._approximate_entropy(token_log_probs, tp)

        return signals

    def _compute_rep_signals(self, input_ids, base_signals, rep_token_log_probs, rep_label, T_full):
        """
        Compute Rep difference signals: base_signal - rep_signal.
        rep_label: 'rep1' or 'rep2'
        """
        signals = {}

        for tp, tp_label in [(T_full, 'T'), (200, '200'), (300, '300')]:
            rep_val = self._cut_off_loss(rep_token_log_probs, tp)
            signals[f'{rep_label}_cut_{tp_label}'] = base_signals[f'cut_{tp_label}'] - rep_val

        for tp, tp_label in [(T_full, 'T'), (200, '200'), (300, '300')]:
            rep_val = self._diversity_calibration(input_ids, rep_token_log_probs, tp)
            signals[f'{rep_label}_cal_{tp_label}'] = base_signals[f'cal_{tp_label}'] - rep_val

        for tp, tp_label in [(T_full, 'T'), (200, '200'), (300, '300')]:
            rep_val = self._perplexity(rep_token_log_probs, tp)
            signals[f'{rep_label}_ppl_{tp_label}'] = base_signals[f'ppl_{tp_label}'] - rep_val

        for tp, tp_label in [(T_full, 'T'), (200, '200'), (300, '300')]:
            rep_val = self._cal_perplexity(input_ids, rep_token_log_probs, tp)
            signals[f'{rep_label}_cal_ppl_{tp_label}'] = base_signals[f'cal_ppl_{tp_label}'] - rep_val

        rep_cb_vals = self._count_below(rep_token_log_probs, 200, [-1.0, -2.0, -3.0])
        for i, tau in enumerate([1, 2, 3]):
            signals[f'{rep_label}_cb_t{tau}'] = base_signals[f'cb_t{tau}'] - rep_cb_vals[i]

        for tp, tp_label in [(T_full, 'T'), (200, '200'), (300, '300')]:
            rep_val = self._count_below_mean(rep_token_log_probs, tp)
            signals[f'{rep_label}_cbm_{tp_label}'] = base_signals[f'cbm_{tp_label}'] - rep_val

        for bins in [3, 4, 5]:
            rep_val = self._lempel_ziv(rep_token_log_probs, T_full, bins)
            signals[f'{rep_label}_lz_b{bins}'] = base_signals[f'lz_b{bins}'] - rep_val

        return signals

    def predict(self, input_ids, token_log_probs, combining_method: str = "edgington", raw_signals: bool = False):
        """Compute all CAMIA signals (base + rep1 + rep2) and optionally combine them.

        Args:
            input_ids: Tokenized input [seq_len].
            token_log_probs: Per-token log probabilities [seq_len].
            combining_method: P-value combining method (default "edgington").
            raw_signals: If True, return individual signals instead of combined score.
        """
        T_full = max(1, input_ids.numel())

        base_signals = self._compute_base_signals(input_ids, token_log_probs, T_full)

        try:
            rep1_log_probs = self._get_rep_token_log_probs(input_ids, rep_type=1)
            rep1_signals = self._compute_rep_signals(
                input_ids, base_signals, rep1_log_probs, 'rep1', T_full
            )
            del rep1_log_probs
        except Exception as e:
            print(f"CAMIA rep1 failed: {e}")
            rep1_signals = {}

        try:
            rep2_log_probs = self._get_rep_token_log_probs(input_ids, rep_type=2)
            rep2_signals = self._compute_rep_signals(
                input_ids, base_signals, rep2_log_probs, 'rep2', T_full
            )
            del rep2_log_probs
        except Exception as e:
            print(f"CAMIA rep2 failed: {e}")
            rep2_signals = {}

        torch.cuda.empty_cache()

        signals = {}
        signals.update(base_signals)
        signals.update(rep1_signals)
        signals.update(rep2_signals)

        if raw_signals:
            return signals

        p_values_list = []
        for name, value in signals.items():
            if name in self.calibration_signal:
                is_count = 'cb' in name and 'rep' not in name
                p_val = self.calculate_p_values(value, self.calibration_signal[name], is_higher_better=is_count)
                p_values_list.append(p_val)
            else:
                p_values_list.append(0.5)

        p_values = np.array(p_values_list, dtype=np.float64)
        p_values = np.clip(p_values, 1e-10, 1 - 1e-10)

        if combining_method == "edgington":
            score = -np.mean(p_values)
        elif combining_method == "fisher":
            from scipy.stats import combine_pvalues as _combine_pv
            _, score = _combine_pv(p_values, method="fisher")
        elif combining_method == "pearson":
            from scipy.stats import combine_pvalues as _combine_pv
            _, score = _combine_pv(p_values, method="pearson")
        elif combining_method == "george":
            from scipy.stats import combine_pvalues as _combine_pv
            _, score = _combine_pv(p_values, method="mudholkar_george")
        else:
            score = -np.mean(p_values)

        return float(score)

    def calculate_p_values(self, value, calibration_values, is_higher_better: bool = False):
        cal_vals = np.array(calibration_values, dtype=np.float32)
        if len(cal_vals) == 0:
            return 0.5
        if is_higher_better:
            cal_vals = -cal_vals
            value = -value
        p_val = (np.sum(cal_vals <= value) + 1) / (len(cal_vals) + 1)
        return p_val.item()

    @classmethod
    def signal_group(cls, feature_name):
        """Map a CAMIA feature name (e.g. 'camia_rep1_cut_200') to its signal group.

        Groups: cut, cal, ppl, cal_ppl, cb, cbm, cbpm, lz, slope, apen.
        Returns 'other' for unrecognized names (e.g. old-style signals).
        """
        name = feature_name.replace('camia_', '')
        name = name.replace('rep1_', '').replace('rep2_', '')
        if name.startswith('cal_ppl_'):
            return 'cal_ppl'
        elif name.startswith('cal_'):
            return 'cal'
        elif name.startswith('cut_'):
            return 'cut'
        elif name.startswith('ppl_'):
            return 'ppl'
        elif name.startswith('cb_t'):
            return 'cb'
        elif name.startswith('cbm_'):
            return 'cbm'
        elif name.startswith('cbpm_'):
            return 'cbpm'
        elif name.startswith('lz_'):
            return 'lz'
        elif name.startswith('slope_'):
            return 'slope'
        elif name.startswith('apen_'):
            return 'apen'
        return 'other'

    @classmethod
    def combine_signals(cls, X_train, y_train, X_known_non, X_eval,
                        camia_indices, feature_keys, n_components=2, seed=42):
        """Combine CAMIA features via Group PCA + Logistic Regression (CAMIA paper Appendix).

        1. Group the CAMIA signals by type (cut, cal, ppl, cb, etc.)
        2. StandardScaler + PCA per group with c components to reduce redundancy
        3. Train LogisticRegression on PCA-transformed features
        4. Return predicted membership probabilities (higher = more member-like)

        Args:
            X_train: training feature matrix (M x D), raw
            y_train: training labels (M,)
            X_known_non: known non-member features (K x D), raw
            X_eval: evaluation features (N x D), raw
            camia_indices: list of column indices for CAMIA features
            feature_keys: list of all feature names
            n_components: PCA components per signal group (default 2, paper optimal)
            seed: random seed for reproducibility

        Returns:
            (train_scores, known_scores, eval_scores): membership probability arrays
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from collections import defaultdict

        camia_names = [feature_keys[i] for i in camia_indices]

        groups = defaultdict(list)
        for local_idx, global_idx in enumerate(camia_indices):
            group = cls.signal_group(camia_names[local_idx])
            groups[group].append(local_idx)

        X_train_camia = X_train[:, camia_indices].copy()
        X_known_camia = X_known_non[:, camia_indices].copy()
        X_eval_camia = X_eval[:, camia_indices].copy()

        for arr in [X_train_camia, X_known_camia, X_eval_camia]:
            arr[~np.isfinite(arr)] = 0.0

        pca_train_parts = []
        pca_known_parts = []
        pca_eval_parts = []

        for group_name in sorted(groups.keys()):
            col_indices = groups[group_name]
            n_cols = len(col_indices)
            c = min(n_components, n_cols)

            train_block = X_train_camia[:, col_indices]
            known_block = X_known_camia[:, col_indices]
            eval_block = X_eval_camia[:, col_indices]

            if n_cols == 1:
                scaler = StandardScaler()
                pca_train_parts.append(scaler.fit_transform(train_block))
                pca_known_parts.append(scaler.transform(known_block))
                pca_eval_parts.append(scaler.transform(eval_block))
            else:
                scaler = StandardScaler()
                train_scaled = scaler.fit_transform(train_block)
                known_scaled = scaler.transform(known_block)
                eval_scaled = scaler.transform(eval_block)

                pca = PCA(n_components=c, random_state=seed)
                pca_train_parts.append(pca.fit_transform(train_scaled))
                pca_known_parts.append(pca.transform(known_scaled))
                pca_eval_parts.append(pca.transform(eval_scaled))

        Z_train = np.hstack(pca_train_parts)
        Z_known = np.hstack(pca_known_parts)
        Z_eval = np.hstack(pca_eval_parts)

        lr = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            random_state=seed,
            C=1.0,
        )
        lr.fit(Z_train, y_train)

        train_scores = lr.predict_proba(Z_train)[:, 1]
        known_scores = lr.predict_proba(Z_known)[:, 1]
        eval_scores = lr.predict_proba(Z_eval)[:, 1]

        return train_scores, known_scores, eval_scores

class ACMIA:
    """
    Automatic Calibration for Membership Inference Attack (ACMIA).

    Computes three families of temperature-calibrated signals:
      - AC   (Eq 5): sgn(1-τ) * mean_FOS(log TSP - log p)
      - DerivAC (Eq 6): mean_FOS(log TSP(τ+δ) - log TSP(τ))
      - NormAC  (Eq 7): mean_FOS((log TSP - μ) / σ)

    Uses beta (β = 1/τ) scaling internally, matching the original implementation:
      β ∈ {0, 2^(-2.5), 2^(-2.4), ..., 2^(0), ..., 2^(2.5)}  (52 values)
    where β=0 produces the uniform distribution baseline.
    """

    def __init__(self, device, logits: torch.Tensor, probs: torch.Tensor,
                 log_probs: torch.Tensor, token_log_probs: torch.Tensor,
                 input_ids: torch.Tensor, betas: list = None):
        """
        Args:
            device: Torch device for computation.
            logits: Model output logits [1, seq_len, vocab_size].
            probs: Softmax probabilities [seq_len, vocab_size].
            log_probs: Log-softmax probabilities [seq_len, vocab_size].
            token_log_probs: Per-token log probabilities [seq_len].
            input_ids: Token IDs of the input [1, seq_len].
            betas: Temperature scaling values (default: [0.0] + [2^(i*0.1) for i in -25..25]).
        """
        self.device = device
        self.logits = logits
        self.probs = probs
        self.log_probs = log_probs
        self.token_log_probs = token_log_probs
        self.input_ids = input_ids

        if betas is None:
            self.betas = [0.0] + [2.0 ** (i * 0.1) for i in range(-25, 26)]
        else:
            self.betas = list(betas)

    def get_fos_mask(self):
        """Compute First Occurrence Subset (FOS) boolean mask."""
        input_ids_flat = self.input_ids.cpu().numpy().flatten()
        _, first_indices = np.unique(input_ids_flat, return_index=True)
        mask = np.zeros_like(input_ids_flat, dtype=bool)
        mask[first_indices] = True
        return mask

    def temperature_scaling(self, beta_chunk_size=8):
        """Apply temperature scaling for all non-zero betas (chunked to avoid OOM).

        Args:
            beta_chunk_size: number of betas to process per GPU chunk.

        Returns:
            log_probs_temps: list of lists, each [seq_len] token log probs
            mu_temps: list of lists, each [seq_len] expected log probs
            sigma_temps: list of lists, each [seq_len] std of log probs
        """
        nonzero_betas = [b for b in self.betas if b > 0]

        log_probs_temps = []
        mu_temps = []
        sigma_temps = []

        log_probs_expanded = self.log_probs.unsqueeze(0)
        input_ids_1 = self.input_ids.unsqueeze(-1)

        for start in range(0, len(nonzero_betas), beta_chunk_size):
            chunk_betas = nonzero_betas[start:start + beta_chunk_size]
            n = len(chunk_betas)

            betas_tensor = torch.tensor(chunk_betas, device=self.device, dtype=torch.float32).view(-1, 1, 1)
            new_log_probs = F.log_softmax(log_probs_expanded * betas_tensor, dim=-1)

            ids_exp = input_ids_1.expand(n, -1, -1)
            token_lp = new_log_probs.gather(-1, ids_exp).squeeze(-1)

            probs = new_log_probs.exp()
            mu = (probs * new_log_probs).sum(-1)
            sigma = (probs * torch.square(new_log_probs - mu.unsqueeze(-1))).sum(-1).sqrt()

            log_probs_temps.extend(token_lp.tolist())
            mu_temps.extend(mu.tolist())
            sigma_temps.extend(sigma.tolist())

            del betas_tensor, new_log_probs, ids_exp, token_lp, probs, mu, sigma
            torch.cuda.empty_cache()

        return log_probs_temps, mu_temps, sigma_temps

    def predict(self):
        log_probs_temps, mu_temps_list, sigma_temps_list = self.temperature_scaling()

        vocab_size = self.log_probs.shape[-1]
        seq_len = self.token_log_probs.shape[-1] if self.token_log_probs.dim() > 0 else 1
        uniform_log_prob = -np.log(vocab_size)

        beta0_tsp = np.full(seq_len, uniform_log_prob)
        beta0_mu = np.full(seq_len, uniform_log_prob)
        beta0_sigma = np.zeros(seq_len)

        all_tsp = np.array([beta0_tsp] + log_probs_temps)
        all_mu = np.array([beta0_mu] + mu_temps_list)
        all_sigma = np.array([beta0_sigma] + sigma_temps_list)

        orig_log_probs = self.token_log_probs.cpu().numpy().flatten()
        fos_mask = self.get_fos_mask()

        min_value = np.finfo(np.float32).min
        max_value = np.finfo(np.float32).max

        num_betas = len(self.betas)
        mid = num_betas // 2

        scores = {}
        for i in range(num_betas):
            idx_label = i - mid

            if i > 0:
                deriv_vals = (all_tsp[i] - all_tsp[i - 1])[fos_mask]
                scores[f"acmia_DerivAC_{idx_label}"] = np.mean(deriv_vals).item()

            diff = all_tsp[i] - orig_log_probs
            if i <= mid:
                diff = -diff
            scores[f"acmia_AC_{idx_label}"] = np.mean(diff[fos_mask]).item()

            if i > 0:
                normalized = (all_tsp[i] - all_mu[i]) / (all_sigma[i] + 1e-10)
                normalized = np.clip(
                    np.nan_to_num(normalized, nan=0.0),
                    a_min=min_value, a_max=max_value
                )
                scores[f"acmia_NormAC_{idx_label}"] = np.mean(normalized[fos_mask]).item()

        return scores

def inference(text: str, model, tokenizer, negative_prefix: list, member_prefix: list, non_member_prefix: list, device, rel_attacks: RelativeLikelihood, dcpdd: DCPDD, noisy_attack: NoisyNeighbour, tagtab_attack: TagTab, camia_attack: CAMIA) -> tuple:
    """
    Performs inference for a single text string.
    Reference-based attacks (OfflineRobustMIA, TokenLevelInfoRMIA, MemorizationGuidedInfoRMIA)
    have been moved to reference_scores.py.

    Returns:
        (results_dict, raw_data_dict) or (None, None) if input is empty.
    """

    data = raw_values(text, model, tokenizer, device)

    if data['input_ids'].numel() == 0:
        return None, None

    base_loss = data['loss'].item()

    results = {
        'loss': -base_loss,
        'ppl': math.exp(base_loss)
    }

    try:
        lower_data = raw_values(text.lower(), model, tokenizer, device)
        lower_loss = lower_data['loss'].item()
        results['lowercase_ppl'] = math.exp(lower_loss)
        del lower_data
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Lowercase perplexity failed: {e}")
        results['lowercase_ppl'] = 0.0

    try:
        baseline = Baseline(
            logits=data['logits'],
            input_ids=data['input_ids'],
            token_log_probs=data['token_log_probs']
        )
        for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            pct = int(ratio * 100)
            results[f'min_k_{pct}'] = baseline.min_k(ratio=ratio)
            results[f'min_k_plus_{pct}'] = baseline.min_k_plus_plus(ratio=ratio)
        results['ranks'] = baseline.ranks()
    except Exception as e:
        print(f"Baseline attacks failed: {e}")
        for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            pct = int(ratio * 100)
            results[f'min_k_{pct}'] = 0.0
            results[f'min_k_plus_{pct}'] = 0.0
        results['ranks'] = 0.0

    try:
        max_renyi = MaxRenyi(
            token_probs=data['token_probs'],
            full_log_probs=data['full_log_probs'],
            full_token_probs=data['full_token_probs'],
            epsilon=1e-10
        )
        renyi_scores = max_renyi.predict()
        results.update(renyi_scores)
    except Exception as e:
        print(f"MaxRenyi failed: {e}")

    try:
        results['zlib'] = zlib_entropy(text)
        results['ppl_zlib_ratio'] = results['ppl'] / results['zlib'] if results['zlib'] > 0 else 0.0
    except Exception as e:
        print(f"Zlib failed: {e}")
        results['zlib'] = 0.0
        results['ppl_zlib_ratio'] = 0.0

    if rel_attacks is not None:
        try:
            recall_multi = rel_attacks.calc_recall_multi(text, base_loss, negative_prefix)
            results.update(recall_multi)
            if negative_prefix:
                results['recall'] = recall_multi.get(f"recall_s{len(negative_prefix)}", 0.0)
        except Exception as e:
            print(f"Recall multi failed: {e}")
            results['recall'] = 0.0

        try:
            conrecall_multi = rel_attacks.calc_conrecall_multi(
                text, base_loss, member_prefix, non_member_prefix
            )
            results.update(conrecall_multi)
        except Exception as e:
            print(f"ConRecall multi failed: {e}")

    if dcpdd is not None:
        try:
            token_probs = data.get('token_probs')
            if token_probs is not None:
                dcpdd_scores = dcpdd.predict_multi(token_probs, data['input_ids'],
                                                    a_values=[1.0, 0.1, 0.01, 0.001])
                results.update(dcpdd_scores)
        except Exception as e:
            print(f"DC-PDD failed: {e}")

    if tagtab_attack is not None:
        try:
            tagtab_scores = tagtab_attack.predict(
                text,
                full_log_probs=data['full_log_probs'],
                shifted_input_ids=data['input_ids']
            )
            results.update(tagtab_scores)
        except Exception as e:
            print(f"TagTab failed: {e}")

    if camia_attack is not None:
        try:
            camia_raw = camia_attack.predict(
                input_ids=data['input_ids'].squeeze(0),
                token_log_probs=data['token_log_probs'],
                raw_signals=True
            )
            results.update({f"camia_{k}": v for k, v in camia_raw.items()})
        except Exception as e:
            print(f"CAMIA failed: {e}")

    try:
        acmia = ACMIA(
            device=device,
            logits=data['logits'].squeeze(0),
            probs=data['full_token_probs'],
            log_probs=data['full_log_probs'],
            token_log_probs=data['token_log_probs'],
            input_ids=data['input_ids'],
        )
        acmia_scores = acmia.predict()
        results.update(acmia_scores)
        del acmia
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"ACMIA failed: {e}")

    if noisy_attack is not None:
        try:
            noisy_scores = noisy_attack.predict_multi(
                input_ids=data['input_ids'],
                base_loss=base_loss,
                sigmas=[0.1, 0.01, 0.001],
                max_neighbours=30,
                checkpoints=[10, 20, 30]
            )
            results.update(noisy_scores)
        except Exception as e:
            print(f"NoisyNeighbour failed: {e}")

    return results, data
