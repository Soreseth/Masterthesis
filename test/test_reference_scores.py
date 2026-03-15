"""Tests for reference_scores.py — Reference-based MIA attacks.

Requires GPU + two models (pythia-70m as both target and ref) for most tests.
Run with:
    pytest tests/test_reference_scores.py -v
"""

import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conftest import SAMPLE_TEXTS, MEDIUM_TEXT


# ════════════════════════════════════════════════════════════════════
# raw_values (reference_scores version)
# ════════════════════════════════════════════════════════════════════

class TestRefRawValues:
    def test_output_keys(self, pythia_70m, device):
        from reference_scores import raw_values
        model, tokenizer = pythia_70m
        data = raw_values(MEDIUM_TEXT, model, tokenizer, device)
        expected = {"loss", "token_probs", "token_log_probs", "logits",
                    "input_ids", "full_token_probs", "full_log_probs"}
        assert expected == set(data.keys())


# ════════════════════════════════════════════════════════════════════
# RefLossDiff
# ════════════════════════════════════════════════════════════════════

class TestRefLossDiff:
    def test_predict_with_precomputed_loss(self, pythia_70m, device):
        from reference_scores import RefLossDiff, compute_loss
        model, tokenizer = pythia_70m
        # Use same model as both target and ref for testing
        rld = RefLossDiff(
            target_model=model, target_tokenizer=tokenizer,
            ref_model=model, ref_tokenizer=tokenizer, device=device
        )
        target_loss = compute_loss(MEDIUM_TEXT, model, tokenizer, device)
        score = rld.predict(MEDIUM_TEXT, target_loss=target_loss)
        # Same model → ref_loss == target_loss → diff ≈ 0
        assert abs(score) < 0.01

    def test_predict_without_precomputed(self, pythia_70m, device):
        from reference_scores import RefLossDiff
        model, tokenizer = pythia_70m
        rld = RefLossDiff(
            target_model=model, target_tokenizer=tokenizer,
            ref_model=model, ref_tokenizer=tokenizer, device=device
        )
        score = rld.predict(MEDIUM_TEXT)
        assert isinstance(score, float)
        assert np.isfinite(score)


# ════════════════════════════════════════════════════════════════════
# TokenLevelInfoRMIA
# ════════════════════════════════════════════════════════════════════

class TestTokenLevelInfoRMIA:
    def test_predict_tokens(self, pythia_70m, device):
        from reference_scores import TokenLevelInfoRMIA
        model, tokenizer = pythia_70m
        tl = TokenLevelInfoRMIA(
            target_model=model, target_tokenizer=tokenizer,
            reference_models=[model], reference_tokenizers=[tokenizer],
            temperature=2.0, aggregation="mean", device=device
        )
        tokens = tl.predict_tokens(MEDIUM_TEXT)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_predict_returns_dict(self, pythia_70m, device):
        from reference_scores import TokenLevelInfoRMIA
        model, tokenizer = pythia_70m
        tl = TokenLevelInfoRMIA(
            target_model=model, target_tokenizer=tokenizer,
            reference_models=[model], reference_tokenizers=[tokenizer],
            temperature=2.0, aggregation="mean", device=device
        )
        result = tl.predict(MEDIUM_TEXT)
        assert isinstance(result, dict)
        assert "token_level_informia" in result

    def test_predict_multi(self, pythia_70m, raw_data_70m, device):
        from reference_scores import TokenLevelInfoRMIA
        model, tokenizer = pythia_70m
        tl = TokenLevelInfoRMIA(
            target_model=model, target_tokenizer=tokenizer,
            reference_models=[model], reference_tokenizers=[tokenizer],
            temperature=2.0, aggregation="mean", device=device
        )
        target_logits = raw_data_70m["logits"].to(device)
        target_labels = raw_data_70m["input_ids"].to(device)
        scores = tl.predict_multi(
            MEDIUM_TEXT,
            temperatures=[1.0, 2.0],
            aggregations=[0.1, 0.5],
            ref_labels=["70m_test"],
            target_logits=target_logits,
            target_labels=target_labels
        )
        assert isinstance(scores, dict)
        assert len(scores) > 0


# ════════════════════════════════════════════════════════════════════
# WBC
# ════════════════════════════════════════════════════════════════════

class TestWBC:
    def test_window_sign_score_static(self):
        from reference_scores import WBC
        target = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ref = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        # ref > target everywhere → score = 1.0
        score = WBC._window_sign_score(target, ref, window_size=2)
        assert score == 1.0

    def test_window_sign_score_equal(self):
        from reference_scores import WBC
        same = np.array([1.0, 2.0, 3.0])
        score = WBC._window_sign_score(same, same, window_size=2)
        assert score == 0.0  # never strictly greater

    def test_window_sign_score_empty(self):
        from reference_scores import WBC
        score = WBC._window_sign_score(np.array([]), np.array([1.0]), window_size=2)
        assert score == 0.5

    def test_predict(self, pythia_70m, device):
        from reference_scores import WBC
        model, tokenizer = pythia_70m
        wbc = WBC(
            target_model=model, target_tokenizer=tokenizer,
            ref_model=model, ref_tokenizer=tokenizer, device=device,
            window_sizes=[2, 4]
        )
        score = wbc.predict(MEDIUM_TEXT)
        assert isinstance(score, float)
        # Same model → ref_losses == target_losses → score = 0.0
        assert abs(score) < 0.01

    def test_predict_per_window(self, pythia_70m, device):
        from reference_scores import WBC
        model, tokenizer = pythia_70m
        wbc = WBC(
            target_model=model, target_tokenizer=tokenizer,
            ref_model=model, ref_tokenizer=tokenizer, device=device,
            window_sizes=[2, 4, 10]
        )
        scores = wbc.predict_per_window(MEDIUM_TEXT, label="test")
        assert isinstance(scores, dict)
        assert len(scores) == 3  # one per window size

    def test_predict_with_precomputed_losses(self, pythia_70m, device):
        from reference_scores import WBC
        model, tokenizer = pythia_70m
        wbc = WBC(
            target_model=model, target_tokenizer=tokenizer,
            ref_model=model, ref_tokenizer=tokenizer, device=device,
            window_sizes=[2, 4]
        )
        target_losses = wbc._per_token_losses(MEDIUM_TEXT, model, tokenizer)
        score = wbc.predict(MEDIUM_TEXT, target_losses=target_losses)
        assert isinstance(score, float)
