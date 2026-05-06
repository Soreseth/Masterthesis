"""reference_scores.py -- Reference-based Membership Inference Attacks for LLMs

All attacks here use a reference model (or reference data distribution) to
calibrate membership scores. This contrasts with target-only attacks (loss,
perplexity, Min-K, zlib, ...) that use only the target model.

Attacks (ordered by compute cost, cheapest first):
    RefLossDiff             -- loss_ref(x) - loss_target(x)
    WBC                     -- Window-Based Comparison (sign-based sliding windows)
    TokenLevelInfoRMIA      -- Token-level log(p_target/p_ref) + KL
"""
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


MODEL_MAX_LENGTH = 2048


# -----------------------------------------------------------------------------
# Shared utilities
# -----------------------------------------------------------------------------
def raw_values(sentence: str, model, tokenizer, device) -> dict:
    """Compute per-token log-probs, probs, logits and loss for a single text."""
    encodings = tokenizer(
        sentence, return_tensors="pt", truncation=True,
        max_length=MODEL_MAX_LENGTH, padding=False,
    ).to(device)

    with torch.no_grad():
        outputs = model(**encodings)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = encodings.input_ids[..., 1:].contiguous()

    logits_sq = shift_logits.squeeze(0)
    labels_sq = shift_labels.squeeze(0)

    loss = F.cross_entropy(logits_sq, labels_sq)
    log_probs = F.log_softmax(logits_sq, dim=-1)
    probs = F.softmax(logits_sq, dim=-1)
    token_log_probs = log_probs.gather(-1, labels_sq.unsqueeze(-1)).squeeze(-1)
    token_probs = probs.gather(-1, labels_sq.unsqueeze(-1)).squeeze(-1)

    return {
        "loss": loss,
        "token_probs": token_probs,
        "token_log_probs": token_log_probs,
        "logits": logits_sq.unsqueeze(0),
        "input_ids": labels_sq.unsqueeze(0).long(),
        "full_token_probs": probs,
        "full_log_probs": log_probs,
    }


def compute_loss(text: str, model, tokenizer, device,
                 max_length: int = MODEL_MAX_LENGTH) -> float:
    """Mean cross-entropy loss for `text` under `model`."""
    ids = tokenizer.encode(text, truncation=True, max_length=max_length)
    if len(ids) < 3:
        return float("nan")
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        loss = model(input_ids=input_ids, labels=input_ids).loss.item()
    return loss


# -----------------------------------------------------------------------------
# 1. RefLossDiff -- simple loss difference
# -----------------------------------------------------------------------------
class RefLossDiff:
    """Simplest reference-based attack: ``loss_ref(x) - loss_target(x)``.

    Members have low target loss (memorised) but normal ref loss -> diff is
    large and positive. Non-members have similar losses under both models ->
    diff near zero.

    Score: ``ref_loss - target_loss``  (higher -> more likely member).
    """

    def __init__(self, target_model, target_tokenizer,
                 ref_model, ref_tokenizer, device):
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.ref_model = ref_model
        self.ref_tokenizer = ref_tokenizer
        self.device = device

    def predict(self, text: str, target_loss: float = None) -> float:
        """Return ``ref_loss - target_loss``. Pass `target_loss` to skip the
        redundant target-side forward pass."""
        if target_loss is None:
            target_loss = compute_loss(text, self.target_model,
                                       self.target_tokenizer, self.device)
        ref_loss = compute_loss(text, self.ref_model,
                                self.ref_tokenizer, self.device)
        if np.isnan(target_loss) or np.isnan(ref_loss):
            return float("nan")
        return ref_loss - target_loss


# -----------------------------------------------------------------------------
# 2. TokenLevelInfoRMIA -- token-level likelihood ratio + KL
# -----------------------------------------------------------------------------
class TokenLevelInfoRMIA:
    """Token-Level Information-Theoretic RMIA (Tao & Shokri, 2025).

    For each token position i, computes:
        score_i = log p_target(x_i | x_{<i}) - log p_ref(x_i | x_{<i})
                  + KL(p_ref(V | x_{<i}) || p_target(V | x_{<i}))

    The first term measures whether the target model is more confident about
    the actual token; the KL term captures broader distributional differences
    over the full vocabulary. Aggregated to sequence level via mean / min /
    max / min-k.

    Score: ``mean(score_i)``  (higher -> more likely member after the negate
    convention applied at the end of `predict`).
    """

    def __init__(self, target_model, target_tokenizer,
                 reference_models, reference_tokenizers,
                 temperature: float = 2.0, aggregation: str = "mean",
                 device=None):
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.reference_models = reference_models
        self.reference_tokenizers = reference_tokenizers
        self.temperature = temperature
        self.aggregation = aggregation
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _get_token_logits(self, model, tokenizer, text):
        data = raw_values(text, model, tokenizer, self.device)
        if data["input_ids"].numel() == 0:
            return None, None
        return data["logits"], data["input_ids"]

    def predict_tokens(self, text: str) -> list:
        """Per-token InfoRMIA scores (list of floats)."""
        target_logits, labels = self._get_token_logits(
            self.target_model, self.target_tokenizer, text)
        if target_logits is None:
            return []

        # Average reference probabilities across the provided ref models.
        ref_probs_list = []
        for ref_model, ref_tok in zip(self.reference_models,
                                      self.reference_tokenizers):
            ref_logits, _ = self._get_token_logits(ref_model, ref_tok, text)
            if ref_logits is not None:
                ref_probs_list.append(
                    F.softmax(ref_logits / self.temperature, dim=-1))
        if not ref_probs_list:
            return []

        ref_probs_mean = torch.stack(ref_probs_list).mean(dim=0)
        target_logits = target_logits.squeeze(0)
        ref_probs_mean = ref_probs_mean.squeeze(0)
        labels = labels.squeeze(0) if labels.dim() > 1 else labels
        eps = 1e-10

        valid_mask = labels != -100
        if not valid_mask.any():
            return []
        valid_labels = labels[valid_mask]

        p_target = F.softmax(target_logits[valid_mask] / self.temperature, dim=-1)
        p_ref = ref_probs_mean[valid_mask]
        p_target = p_target / p_target.sum(-1, keepdim=True)
        p_ref = p_ref / p_ref.sum(-1, keepdim=True)
        p_target = p_target.clamp(min=eps)
        p_ref = p_ref.clamp(min=eps)

        log_ratio = (
            torch.log(p_target.gather(-1, valid_labels.unsqueeze(-1)))
            - torch.log(p_ref.gather(-1, valid_labels.unsqueeze(-1)))
        ).squeeze(-1)
        kl = (p_ref * (torch.log(p_ref) - torch.log(p_target))).sum(-1)

        return (log_ratio + kl).cpu().tolist()

    def predict(self, text: str) -> dict:
        """Aggregated sequence-level score under `self.aggregation`.

        Negates the raw score so "higher = more likely member" -- the raw
        (log-ratio + KL) is empirically lower for members on pretrained LLMs.
        """
        token_scores = self.predict_tokens(text)
        if not token_scores:
            return {"token_level_informia": np.nan}

        scores_arr = np.array(token_scores)
        agg = self.aggregation
        if agg == "mean":
            seq = np.mean(scores_arr)
        elif agg == "min":
            seq = np.min(scores_arr)
        elif agg == "max":
            seq = np.max(scores_arr)
        elif agg.startswith("min-k"):
            k_pct = int(agg.split("-")[-1])
            k = max(1, len(scores_arr) * k_pct // 100)
            seq = np.mean(np.sort(scores_arr)[:k])
        else:
            seq = np.mean(scores_arr)

        return {"token_level_informia": -seq}

    def predict_multi(self, text: str, temperatures: list = None,
                      aggregations: list = None,
                      ref_labels: list = None,
                      target_logits: torch.Tensor = None,
                      target_labels: torch.Tensor = None) -> dict:
        """Per-(reference model, temperature, aggregation) scores.

        Instead of averaging reference probabilities, computes a separate
        score for each reference model, every temperature in `temperatures`,
        and every aggregation in `aggregations`.

        Args:
            temperatures: defaults to [0.5, 1.0, 2.0, 5.0].
            aggregations: float < 1.0 -> min-k ratio (e.g. 0.1 = min-k-10);
                          1.0 -> mean.
            ref_labels:   string label per reference model.
            target_logits / target_labels: pre-computed (skip a forward pass).

        Returns dict like
            ``{'tl_informia_<ref_label>_t<T>_<agg>': score, ...}``
        """
        if temperatures is None:
            temperatures = [0.5, 1.0, 2.0, 5.0]
        if aggregations is None:
            aggregations = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
        if ref_labels is None:
            ref_labels = [f"ref{i}" for i in range(len(self.reference_models))]

        if target_logits is not None and target_labels is not None:
            tgt_logits, labels = target_logits, target_labels
        else:
            tgt_logits, labels = self._get_token_logits(
                self.target_model, self.target_tokenizer, text)
        if tgt_logits is None:
            return {}

        target_logits_sq = tgt_logits.squeeze(0)
        labels_sq = labels.squeeze(0) if labels.dim() > 1 else labels
        eps = 1e-10
        results = {}

        for ref_idx, (ref_model, ref_tok) in enumerate(
                zip(self.reference_models, self.reference_tokenizers)):
            ref_logits, _ = self._get_token_logits(ref_model, ref_tok, text)
            if ref_logits is None:
                continue
            ref_logits_sq = ref_logits.squeeze(0)
            rl = ref_labels[ref_idx]

            valid_mask = labels_sq != -100
            if not valid_mask.any():
                continue
            valid_labels = labels_sq[valid_mask]

            min_vocab = min(target_logits_sq.shape[-1], ref_logits_sq.shape[-1])
            tgt_logits_v = target_logits_sq[..., :min_vocab]
            ref_logits_v = ref_logits_sq[..., :min_vocab]

            for temp in temperatures:
                p_target = F.softmax(tgt_logits_v / temp, dim=-1)
                p_ref = F.softmax(ref_logits_v / temp, dim=-1)
                p_target_v = p_target[valid_mask]
                p_ref_v = p_ref[valid_mask]
                p_target_v = p_target_v / p_target_v.sum(-1, keepdim=True)
                p_ref_v = p_ref_v / p_ref_v.sum(-1, keepdim=True)
                p_target_c = p_target_v.clamp(min=eps)
                p_ref_c = p_ref_v.clamp(min=eps)

                log_ratio = (
                    torch.log(p_target_c.gather(-1, valid_labels.unsqueeze(-1)))
                    - torch.log(p_ref_c.gather(-1, valid_labels.unsqueeze(-1)))
                ).squeeze(-1)
                kl = (p_ref_c * (torch.log(p_ref_c) - torch.log(p_target_c))).sum(-1)

                raw_scores = (log_ratio + kl).cpu().numpy()
                scores_arr = np.nan_to_num(
                    raw_scores, nan=0.0, posinf=0.0, neginf=0.0)
                t_label = f"t{temp}"

                for agg in aggregations:
                    if agg >= 1.0:
                        seq = np.mean(scores_arr) if len(scores_arr) > 0 else 0.0
                        agg_label = "mean"
                    else:
                        k_pct = int(agg * 100)
                        k = max(1, len(scores_arr) * k_pct // 100)
                        seq = (np.mean(np.sort(scores_arr)[:k])
                               if len(scores_arr) > 0 else 0.0)
                        agg_label = f"mk{k_pct}"
                    results[f"tl_informia_{rl}_{t_label}_{agg_label}"] = -seq

        return results


# -----------------------------------------------------------------------------
# 3. WBC -- Window-Based Comparison
# -----------------------------------------------------------------------------
class WBC:
    """Window-Based Comparison Attack (Y. Chen et al., 2026).

    Compares per-token losses between target and reference models using
    sliding windows with sign-based (binary) aggregation. For each window,
    only the *direction* of the loss difference matters -- making the score
    robust to long-tailed extrema from domain-specific tokens.

    Multi-window ensemble averages over a geometric progression of window
    sizes (default: ``{2,3,4,6,9,13,18,25,32,40}``).

    Score: fraction of windows where ``ref_loss > target_loss`` (higher -> member).
    """

    # Geometric progression w_k = round(2 * 20 ** ((k-1)/9)) for k = 1..10.
    DEFAULT_WINDOWS = [2, 3, 4, 6, 9, 13, 18, 25, 32, 40]

    def __init__(self, target_model, target_tokenizer,
                 ref_model, ref_tokenizer, device,
                 window_sizes: Optional[list] = None):
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.ref_model = ref_model
        self.ref_tokenizer = ref_tokenizer
        self.device = device
        self.window_sizes = window_sizes or self.DEFAULT_WINDOWS

    def _per_token_losses(self, text: str, model, tokenizer) -> np.ndarray:
        """Compute per-token NLL losses for a single text."""
        enc = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=MODEL_MAX_LENGTH, padding=False,
        ).to(self.device)

        input_ids = enc["input_ids"]
        if input_ids.shape[1] < 3:
            return np.array([])

        with torch.no_grad():
            logits = model(**enc).logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )
        return loss_per_token.cpu().numpy()

    @staticmethod
    def _window_sign_score(target_losses: np.ndarray,
                           ref_losses: np.ndarray,
                           window_size: int) -> float:
        """Fraction of sliding windows where sum(ref) > sum(target). Uses
        np.convolve for efficient sliding sums."""
        min_len = min(len(target_losses), len(ref_losses))
        if min_len == 0:
            return 0.5
        effective_w = min(window_size, min_len)
        kernel = np.ones(effective_w)
        t_sums = np.convolve(target_losses[:min_len], kernel, mode="valid")
        r_sums = np.convolve(ref_losses[:min_len], kernel, mode="valid")
        return float(np.mean(r_sums > t_sums))

    def predict(self, text: str, target_losses: np.ndarray = None) -> float:
        """Mean sign-based score across all window sizes (higher -> member)."""
        if target_losses is None:
            target_losses = self._per_token_losses(
                text, self.target_model, self.target_tokenizer)
        ref_losses = self._per_token_losses(
            text, self.ref_model, self.ref_tokenizer)

        if len(target_losses) == 0 or len(ref_losses) == 0:
            return float("nan")

        scores = [
            self._window_sign_score(target_losses, ref_losses, w)
            for w in self.window_sizes
        ]
        return float(np.mean(scores))

    def predict_per_window(self, text: str, label: str = "",
                           target_losses: np.ndarray = None) -> dict:
        """Per-window-size scores, keyed ``wbc_<label>_w<size>``."""
        if target_losses is None:
            target_losses = self._per_token_losses(
                text, self.target_model, self.target_tokenizer)
        ref_losses = self._per_token_losses(
            text, self.ref_model, self.ref_tokenizer)

        if len(target_losses) == 0 or len(ref_losses) == 0:
            return {}

        prefix = f"wbc_{label}_" if label else "wbc_"
        return {
            f"{prefix}w{w}": self._window_sign_score(target_losses, ref_losses, w)
            for w in self.window_sizes
        }
