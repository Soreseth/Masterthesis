"""Tests for scores.py — MIA attack classes.

Requires GPU + pythia-70m for most tests. Run with:
    pytest tests/test_scores.py -v
    pytest tests/test_scores.py -v -k "not slow"  # skip integration tests
"""

import sys
import os
import math
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conftest import SAMPLE_TEXTS, MEDIUM_TEXT, SHORT_TEXT, LONG_TEXT


# ════════════════════════════════════════════════════════════════════
# raw_values
# ════════════════════════════════════════════════════════════════════

class TestRawValues:
    def test_output_keys(self, raw_data_70m):
        expected = {"loss", "token_probs", "token_log_probs", "logits",
                    "input_ids", "full_token_probs", "full_log_probs"}
        assert expected == set(raw_data_70m.keys())

    def test_loss_is_scalar(self, raw_data_70m):
        assert raw_data_70m["loss"].dim() == 0

    def test_loss_is_positive(self, raw_data_70m):
        assert raw_data_70m["loss"].item() > 0

    def test_logits_shape(self, raw_data_70m):
        # [1, seq_len, vocab_size]
        assert raw_data_70m["logits"].dim() == 3
        assert raw_data_70m["logits"].shape[0] == 1

    def test_input_ids_shape(self, raw_data_70m):
        # [1, seq_len]
        assert raw_data_70m["input_ids"].dim() == 2
        assert raw_data_70m["input_ids"].shape[0] == 1

    def test_token_log_probs_negative(self, raw_data_70m):
        # log probs should be <= 0
        assert (raw_data_70m["token_log_probs"] <= 0).all()

    def test_probs_sum_to_one(self, raw_data_70m):
        sums = raw_data_70m["full_token_probs"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-3)

    def test_short_text(self, pythia_70m, device):
        from scores import raw_values
        model, tokenizer = pythia_70m
        data = raw_values(SHORT_TEXT, model, tokenizer, device)
        assert data["loss"].item() > 0
        assert data["input_ids"].shape[1] >= 1


# ════════════════════════════════════════════════════════════════════
# Baseline (Min-K, Min-K++, Ranks)
# ════════════════════════════════════════════════════════════════════

class TestBaseline:
    def test_min_k_returns_float(self, raw_data_70m):
        from scores import Baseline
        b = Baseline(
            logits=raw_data_70m["logits"],
            input_ids=raw_data_70m["input_ids"],
            token_log_probs=raw_data_70m["token_log_probs"]
        )
        result = b.min_k(ratio=0.1)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_min_k_negative(self, raw_data_70m):
        from scores import Baseline
        b = Baseline(
            logits=raw_data_70m["logits"],
            input_ids=raw_data_70m["input_ids"],
            token_log_probs=raw_data_70m["token_log_probs"]
        )
        # min-k selects lowest log probs, which are negative
        assert b.min_k(ratio=0.05) < 0

    def test_min_k_plus_plus_returns_float(self, raw_data_70m):
        from scores import Baseline
        b = Baseline(
            logits=raw_data_70m["logits"],
            input_ids=raw_data_70m["input_ids"],
            token_log_probs=raw_data_70m["token_log_probs"]
        )
        result = b.min_k_plus_plus(ratio=0.1)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_ranks_positive(self, raw_data_70m):
        from scores import Baseline
        b = Baseline(
            logits=raw_data_70m["logits"],
            input_ids=raw_data_70m["input_ids"],
            token_log_probs=raw_data_70m["token_log_probs"]
        )
        rank = b.ranks()
        assert isinstance(rank, float)
        assert rank >= 1.0

    @pytest.mark.parametrize("ratio", [0.05, 0.1, 0.2, 0.3, 0.5])
    def test_min_k_ratios(self, raw_data_70m, ratio):
        from scores import Baseline
        b = Baseline(
            logits=raw_data_70m["logits"],
            input_ids=raw_data_70m["input_ids"],
            token_log_probs=raw_data_70m["token_log_probs"]
        )
        result = b.min_k(ratio=ratio)
        assert np.isfinite(result)


# ════════════════════════════════════════════════════════════════════
# MaxRenyi
# ════════════════════════════════════════════════════════════════════

class TestMaxRenyi:
    def test_predict_returns_dict(self, raw_data_70m):
        from scores import MaxRenyi
        mr = MaxRenyi(
            token_probs=raw_data_70m["token_probs"],
            full_log_probs=raw_data_70m["full_log_probs"],
            full_token_probs=raw_data_70m["full_token_probs"],
        )
        scores = mr.predict()
        assert isinstance(scores, dict)
        assert len(scores) > 0

    def test_predict_values_finite(self, raw_data_70m):
        from scores import MaxRenyi
        mr = MaxRenyi(
            token_probs=raw_data_70m["token_probs"],
            full_log_probs=raw_data_70m["full_log_probs"],
            full_token_probs=raw_data_70m["full_token_probs"],
        )
        scores = mr.predict()
        for key, val in scores.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"


# ════════════════════════════════════════════════════════════════════
# DCPDD
# ════════════════════════════════════════════════════════════════════

class TestDCPDD:
    def test_predict(self, raw_data_70m, freq_dist, device):
        from scores import DCPDD
        dcpdd = DCPDD(freq_dist, device=device, a=0.01, apply_smoothing=True)
        score = dcpdd.predict(
            raw_data_70m["token_probs"],
            raw_data_70m["input_ids"]
        )
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_predict_multi(self, raw_data_70m, freq_dist, device):
        from scores import DCPDD
        dcpdd = DCPDD(freq_dist, device=device, a=0.01, apply_smoothing=True)
        scores = dcpdd.predict_multi(
            raw_data_70m["token_probs"],
            raw_data_70m["input_ids"],
            a_values=[1.0, 0.1, 0.01, 0.001]
        )
        assert isinstance(scores, dict)
        assert len(scores) == 4
        for key, val in scores.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"


# ════════════════════════════════════════════════════════════════════
# NoisyNeighbour
# ════════════════════════════════════════════════════════════════════

class TestNoisyNeighbour:
    def test_predict(self, pythia_70m, raw_data_70m, device):
        from scores import NoisyNeighbour
        model, _ = pythia_70m
        nn = NoisyNeighbour(model=model, device=device, batch_size=2)
        score = nn.predict(
            input_ids=raw_data_70m["input_ids"],
            base_loss=raw_data_70m["loss"].item(),
            sigma=0.1,
            n_neighbours=10
        )
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_predict_multi(self, pythia_70m, raw_data_70m, device):
        from scores import NoisyNeighbour
        model, _ = pythia_70m
        nn = NoisyNeighbour(model=model, device=device, batch_size=2)
        scores = nn.predict_multi(
            input_ids=raw_data_70m["input_ids"],
            base_loss=raw_data_70m["loss"].item(),
            sigmas=[0.1, 0.01],
            max_neighbours=10,
            checkpoints=[10]
        )
        assert isinstance(scores, dict)
        assert len(scores) > 0
        for key, val in scores.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"


# ════════════════════════════════════════════════════════════════════
# CAMIA
# ════════════════════════════════════════════════════════════════════

class TestCAMIA:
    def test_predict_raw_signals(self, pythia_70m, raw_data_70m, device):
        from scores import CAMIA
        model, tokenizer = pythia_70m
        camia = CAMIA(
            target_model=model, target_tokenizer=tokenizer,
            device=device, max_len=512, calibration_signal={}
        )
        signals = camia.predict(
            input_ids=raw_data_70m["input_ids"].squeeze(0),
            token_log_probs=raw_data_70m["token_log_probs"],
            raw_signals=True
        )
        assert isinstance(signals, dict)
        assert len(signals) > 0

    def test_signal_keys_present(self, pythia_70m, raw_data_70m, device):
        from scores import CAMIA
        model, tokenizer = pythia_70m
        camia = CAMIA(
            target_model=model, target_tokenizer=tokenizer,
            device=device, max_len=512, calibration_signal={}
        )
        signals = camia.predict(
            input_ids=raw_data_70m["input_ids"].squeeze(0),
            token_log_probs=raw_data_70m["token_log_probs"],
            raw_signals=True
        )
        # Should contain cut-off, cal, ppl signals
        signal_prefixes = [k.split("_")[0] for k in signals.keys()]
        assert any("Cut" in k for k in signals.keys()) or any("PPL" in k for k in signals.keys())


# ════════════════════════════════════════════════════════════════════
# ACMIA
# ════════════════════════════════════════════════════════════════════

class TestACMIA:
    def test_predict_returns_dict(self, raw_data_70m, device):
        from scores import ACMIA
        acmia = ACMIA(
            device=device,
            logits=raw_data_70m["logits"].squeeze(0),
            probs=raw_data_70m["full_token_probs"],
            log_probs=raw_data_70m["full_log_probs"],
            token_log_probs=raw_data_70m["token_log_probs"],
            input_ids=raw_data_70m["input_ids"],
        )
        scores = acmia.predict()
        assert isinstance(scores, dict)
        assert len(scores) > 0

    def test_acmia_key_structure(self, raw_data_70m, device):
        from scores import ACMIA
        acmia = ACMIA(
            device=device,
            logits=raw_data_70m["logits"].squeeze(0),
            probs=raw_data_70m["full_token_probs"],
            log_probs=raw_data_70m["full_log_probs"],
            token_log_probs=raw_data_70m["token_log_probs"],
            input_ids=raw_data_70m["input_ids"],
        )
        scores = acmia.predict()
        # Keys should follow pattern: acmia_{AC|DerivAC|NormAC}_{beta_value}
        for key in scores.keys():
            assert key.startswith("acmia_"), f"Unexpected key: {key}"

    def test_default_betas_count(self):
        # 52 betas: [0.0] + 51 powers of 2
        betas = [0.0] + [2.0 ** (i * 0.1) for i in range(-25, 26)]
        assert len(betas) == 52


# ════════════════════════════════════════════════════════════════════
# RelativeLikelihood (Recall / ConRecall)
# ════════════════════════════════════════════════════════════════════

class TestRelativeLikelihood:
    def test_calc_recall_multi(self, pythia_70m, raw_data_70m, device):
        from scores import RelativeLikelihood
        model, tokenizer = pythia_70m

        rel = RelativeLikelihood(base_model=model, base_tokenizer=tokenizer, device=device)

        prefix_text = SAMPLE_TEXTS[1]
        prefix_enc = [tokenizer.encode(
            prefix_text, add_special_tokens=False,
            return_tensors='pt', truncation=True, max_length=256
        )]

        base_loss = raw_data_70m["loss"].item()
        scores = rel.calc_recall_multi(MEDIUM_TEXT, base_loss, prefix_enc)
        assert isinstance(scores, dict)
        assert len(scores) > 0

    def test_calc_conrecall_multi(self, pythia_70m, raw_data_70m, device):
        from scores import RelativeLikelihood
        model, tokenizer = pythia_70m
        rel = RelativeLikelihood(base_model=model, base_tokenizer=tokenizer, device=device)

        prefix_text = SAMPLE_TEXTS[1]
        prefix_enc = [tokenizer.encode(
            prefix_text, add_special_tokens=False,
            return_tensors='pt', truncation=True, max_length=256
        )]

        base_loss = raw_data_70m["loss"].item()
        scores = rel.calc_conrecall_multi(
            MEDIUM_TEXT, base_loss, prefix_enc, prefix_enc
        )
        assert isinstance(scores, dict)


# ════════════════════════════════════════════════════════════════════
# Utility functions
# ════════════════════════════════════════════════════════════════════

class TestUtilityFunctions:
    def test_perplexity(self):
        from scores import perplexity
        assert perplexity(0.0) == 1.0
        assert perplexity(1.0) == pytest.approx(math.e, rel=1e-5)
        assert perplexity(2.0) > perplexity(1.0)

    def test_zlib_entropy(self):
        from scores import zlib_entropy
        short_val = zlib_entropy("hello")
        long_val = zlib_entropy("hello " * 100)
        # Compressed size should increase with longer text
        assert long_val > short_val
        assert isinstance(short_val, int)

    def test_fix_seed_determinism(self):
        from scores import fix_seed
        fix_seed(42)
        a = torch.randn(5)
        fix_seed(42)
        b = torch.randn(5)
        assert torch.equal(a, b)


# ════════════════════════════════════════════════════════════════════
# TagTab
# ════════════════════════════════════════════════════════════════════

class TestTagTab:
    def test_predict(self, pythia_70m, raw_data_70m, device):
        import spacy
        from scores import TagTab

        model, tokenizer = pythia_70m
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("spacy model en_core_web_sm not installed")

        tagtab = TagTab(
            target_model=model, target_tokenizer=tokenizer,
            top_k=5, nlp=nlp, device=device,
            entropy_map=None, min_sentence_len=3, max_sentence_len=40
        )
        scores = tagtab.predict(
            MEDIUM_TEXT,
            full_log_probs=raw_data_70m["full_log_probs"],
            shifted_input_ids=raw_data_70m["input_ids"]
        )
        assert isinstance(scores, dict)


# ════════════════════════════════════════════════════════════════════
# Full inference function
# ════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestInference:
    def test_inference_returns_results(self, pythia_70m, freq_dist, device):
        import spacy
        from scores import inference, RelativeLikelihood, DCPDD, NoisyNeighbour, TagTab, CAMIA

        model, tokenizer = pythia_70m
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("spacy model en_core_web_sm not installed")

        rel = RelativeLikelihood(base_model=model, base_tokenizer=tokenizer, device=device)
        dcpdd = DCPDD(freq_dist, device=device, a=0.01, apply_smoothing=True)
        nn = NoisyNeighbour(model=model, device=device, batch_size=2)
        tagtab = TagTab(
            target_model=model, target_tokenizer=tokenizer,
            top_k=5, nlp=nlp, device=device,
            entropy_map=None, min_sentence_len=3, max_sentence_len=40
        )
        camia = CAMIA(
            target_model=model, target_tokenizer=tokenizer,
            device=device, max_len=512, calibration_signal={}
        )

        prefix_enc = [tokenizer.encode(
            SAMPLE_TEXTS[1], add_special_tokens=False,
            return_tensors='pt', truncation=True, max_length=256
        )]

        results, data = inference(
            text=MEDIUM_TEXT,
            model=model, tokenizer=tokenizer,
            negative_prefix=prefix_enc,
            member_prefix=prefix_enc,
            non_member_prefix=prefix_enc,
            device=device,
            rel_attacks=rel, dcpdd=dcpdd,
            noisy_attack=nn, tagtab_attack=tagtab,
            camia_attack=camia
        )

        assert results is not None
        assert data is not None
        assert "loss" in results
        assert "ppl" in results
        assert "min_k_5" in results
        assert "ranks" in results

    def test_inference_empty_input(self, pythia_70m, device):
        from scores import inference, RelativeLikelihood, DCPDD, NoisyNeighbour, CAMIA

        model, tokenizer = pythia_70m
        results, data = inference(
            text="",
            model=model, tokenizer=tokenizer,
            negative_prefix=[], member_prefix=[], non_member_prefix=[],
            device=device,
            rel_attacks=None, dcpdd=None,
            noisy_attack=None, tagtab_attack=None,
            camia_attack=None
        )
        # Empty text should return None, None
        assert results is None
        assert data is None
