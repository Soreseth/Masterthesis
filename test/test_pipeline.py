"""Integration tests for the full MIA pipeline.

Tests the end-to-end flow: text → chunking → inference → scores.
Requires GPU + pythia-70m. Mark with @pytest.mark.slow.

Run with:
    pytest tests/test_pipeline.py -v --tb=short
"""

import sys
import os
import json
import tempfile
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conftest import SAMPLE_TEXTS, MEDIUM_TEXT, LONG_TEXT


@pytest.mark.slow
class TestEndToEndPipeline:
    """Full pipeline: text → chunks → raw_values → all attacks → save."""

    def test_chunk_and_score(self, pythia_70m, freq_dist, device):
        """Process a text through chunking and all base attacks."""
        from preprocess import create_chunks, TensorEncoder
        from scores import raw_values, Baseline, MaxRenyi, DCPDD, CAMIA, ACMIA

        model, tokenizer = pythia_70m
        text = LONG_TEXT
        chunks = create_chunks(text, tokenizer, max_length=512)

        all_results = []
        for chunk in chunks:
            if len(chunk) <= 25:
                continue
            data = raw_values(chunk, model, tokenizer, device)

            results = {"loss": -data["loss"].item()}

            # Baseline
            baseline = Baseline(
                logits=data["logits"],
                input_ids=data["input_ids"],
                token_log_probs=data["token_log_probs"]
            )
            results["min_k_5"] = baseline.min_k(ratio=0.05)
            results["min_k_plus_5"] = baseline.min_k_plus_plus(ratio=0.05)
            results["ranks"] = baseline.ranks()

            # MaxRenyi
            mr = MaxRenyi(
                token_probs=data["token_probs"],
                full_log_probs=data["full_log_probs"],
                full_token_probs=data["full_token_probs"],
            )
            results.update(mr.predict())

            # DCPDD
            dcpdd = DCPDD(freq_dist, device=device, a=0.01, apply_smoothing=True)
            results.update(dcpdd.predict_multi(
                data["token_probs"], data["input_ids"],
                a_values=[0.01, 0.001]
            ))

            # CAMIA
            camia = CAMIA(
                target_model=model, target_tokenizer=tokenizer,
                device=device, max_len=512, calibration_signal={}
            )
            signals = camia.predict(
                input_ids=data["input_ids"].squeeze(0),
                token_log_probs=data["token_log_probs"],
                raw_signals=True
            )
            results.update({f"camia_{k}": v for k, v in signals.items()})

            # ACMIA
            acmia = ACMIA(
                device=device,
                logits=data["logits"].squeeze(0),
                probs=data["full_token_probs"],
                log_probs=data["full_log_probs"],
                token_log_probs=data["token_log_probs"],
                input_ids=data["input_ids"],
            )
            results.update(acmia.predict())
            del acmia

            all_results.append(results)

        assert len(all_results) > 0
        for res in all_results:
            assert "loss" in res
            assert "min_k_5" in res
            assert "ranks" in res

    def test_save_and_reload_jsonl(self, pythia_70m, device):
        """Verify scores can be serialized and deserialized correctly."""
        from preprocess import TensorEncoder
        from scores import raw_values, Baseline

        model, tokenizer = pythia_70m
        data = raw_values(MEDIUM_TEXT, model, tokenizer, device)
        baseline = Baseline(
            logits=data["logits"],
            input_ids=data["input_ids"],
            token_log_probs=data["token_log_probs"]
        )

        results = {
            "loss": -data["loss"].item(),
            "min_k_5": baseline.min_k(0.05),
            "ranks": baseline.ranks(),
        }

        doc = {"pred": [results], "label": 1}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps(doc, cls=TensorEncoder) + "\n")
            path = f.name

        try:
            with open(path) as f:
                loaded = json.loads(f.readline())
            assert loaded["label"] == 1
            assert len(loaded["pred"]) == 1
            assert abs(loaded["pred"][0]["loss"] - results["loss"]) < 1e-6
        finally:
            os.unlink(path)

    def test_multiple_texts_consistency(self, pythia_70m, device):
        """Same text should produce identical scores."""
        from scores import raw_values, Baseline

        model, tokenizer = pythia_70m
        results_a = raw_values(MEDIUM_TEXT, model, tokenizer, device)
        results_b = raw_values(MEDIUM_TEXT, model, tokenizer, device)

        assert abs(results_a["loss"].item() - results_b["loss"].item()) < 1e-5

        ba = Baseline(results_a["logits"], results_a["input_ids"], results_a["token_log_probs"])
        bb = Baseline(results_b["logits"], results_b["input_ids"], results_b["token_log_probs"])

        assert abs(ba.min_k(0.1) - bb.min_k(0.1)) < 1e-5
        assert abs(ba.ranks() - bb.ranks()) < 1e-5


@pytest.mark.slow
class TestReferenceModelPipeline:
    """Test the reference-model attack pipeline (Pass 2 of precompute_mia_scores)."""

    def test_ref_loss_diff_pipeline(self, pythia_70m, device):
        from scores import raw_values
        from reference_scores import RefLossDiff

        model, tokenizer = pythia_70m
        data = raw_values(MEDIUM_TEXT, model, tokenizer, device)

        rld = RefLossDiff(
            target_model=model, target_tokenizer=tokenizer,
            ref_model=model, ref_tokenizer=tokenizer, device=device
        )
        score = rld.predict(MEDIUM_TEXT, target_loss=data["loss"].item())
        assert np.isfinite(score)

    def test_wbc_pipeline(self, pythia_70m, device):
        from scores import raw_values
        from reference_scores import WBC
        import torch.nn.functional as F

        model, tokenizer = pythia_70m
        data = raw_values(MEDIUM_TEXT, model, tokenizer, device)

        # Compute per-token losses (like precompute_mia_scores does)
        per_token_losses = F.cross_entropy(
            data["logits"].squeeze(0),
            data["input_ids"].squeeze(0),
            reduction="none"
        ).cpu().numpy()

        wbc = WBC(
            target_model=model, target_tokenizer=tokenizer,
            ref_model=model, ref_tokenizer=tokenizer, device=device,
            window_sizes=[2, 4]
        )
        scores = wbc.predict_per_window(
            MEDIUM_TEXT, label="70m_test", target_losses=per_token_losses
        )
        assert isinstance(scores, dict)
        assert len(scores) == 2


@pytest.mark.slow
class TestPrecomputeHelpers:
    """Test helper functions from precompute_mia_scores.py."""

    def test_get_mapped_value(self):
        from precompute_mia_scores import get_mapped_value
        assert get_mapped_value(43) == 10
        assert get_mapped_value(512) == 5
        assert get_mapped_value(1024) == 2
        assert get_mapped_value(2048) == 1
        assert get_mapped_value(999) == 1  # default

    def test_load_ref_model(self, device):
        from precompute_mia_scores import load_ref_model
        model, tokenizer = load_ref_model("EleutherAI__pythia-70m", device)
        assert model is not None
        assert tokenizer is not None
        assert tokenizer.pad_token is not None
        del model
        torch.cuda.empty_cache()

    def test_load_ref_model_missing(self, device):
        from precompute_mia_scores import load_ref_model
        model, tokenizer = load_ref_model("nonexistent_model", device)
        assert model is None
        assert tokenizer is None
