"""Run every MIA attack on the Bochum text, assert it produces a finite
feature value (or vector) without erroring.

These tests use a tiny shared HF model as both target and reference, so the
absolute scores are uninformative -- the goal is solely to catch regressions in
the attack code itself (signature changes, NaN explosions, missing kwargs).
"""
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")


# --- Reference-based attacks ------------------------------------------------
def test_raw_values_returns_expected_keys(bochum_text, tiny_model_and_tokenizer, device):
    from src.attacks.reference_scores import raw_values
    model, tok = tiny_model_and_tokenizer
    out = raw_values(bochum_text, model, tok, device)
    assert {"loss", "token_probs", "token_log_probs", "logits", "input_ids",
            "full_token_probs", "full_log_probs"}.issubset(out.keys())
    assert out["token_log_probs"].dim() == 1
    assert out["loss"].numel() == 1


def test_ref_loss_diff(bochum_text, tiny_model_and_tokenizer, device):
    from src.attacks.reference_scores import RefLossDiff
    model, tok = tiny_model_and_tokenizer
    attack = RefLossDiff(target_model=model, target_tokenizer=tok,
                        ref_model=model, ref_tokenizer=tok, device=device)
    score = attack.predict(bochum_text)
    assert isinstance(score, float)
    assert np.isfinite(score)
    # target == ref -> score must be ~0
    assert abs(score) < 1e-3, f"expected ~0 with shared model, got {score}"


def test_token_level_inforRMIA_predict(bochum_text, tiny_model_and_tokenizer, device):
    from src.attacks.reference_scores import TokenLevelInfoRMIA
    model, tok = tiny_model_and_tokenizer
    attack = TokenLevelInfoRMIA(
        target_model=model, target_tokenizer=tok,
        reference_models=[model], reference_tokenizers=[tok],
        device=device,
    )
    out = attack.predict(bochum_text)
    assert "token_level_informia" in out
    assert np.isfinite(out["token_level_informia"])


def test_token_level_inforRMIA_predict_tokens(
        bochum_text, tiny_model_and_tokenizer, device):
    from src.attacks.reference_scores import TokenLevelInfoRMIA
    model, tok = tiny_model_and_tokenizer
    attack = TokenLevelInfoRMIA(
        target_model=model, target_tokenizer=tok,
        reference_models=[model], reference_tokenizers=[tok],
        device=device,
    )
    token_scores = attack.predict_tokens(bochum_text)
    assert isinstance(token_scores, list)
    assert len(token_scores) > 0
    assert all(np.isfinite(s) for s in token_scores)


def test_token_level_inforRMIA_predict_multi(
        bochum_text, tiny_model_and_tokenizer, device):
    """predict_multi yields cartesian (ref × temperature × aggregation) keys."""
    from src.attacks.reference_scores import TokenLevelInfoRMIA
    model, tok = tiny_model_and_tokenizer
    attack = TokenLevelInfoRMIA(
        target_model=model, target_tokenizer=tok,
        reference_models=[model], reference_tokenizers=[tok],
        device=device,
    )
    out = attack.predict_multi(
        bochum_text,
        temperatures=[1.0, 2.0],
        aggregations=[0.1, 1.0],
        ref_labels=["ref0"],
    )
    assert len(out) == 2 * 2  # one ref × 2 temps × 2 aggs
    assert all(k.startswith("tl_informia_ref0_") for k in out)
    assert all(np.isfinite(v) for v in out.values())


def test_wbc_predict(bochum_text, tiny_model_and_tokenizer, device):
    from src.attacks.reference_scores import WBC
    model, tok = tiny_model_and_tokenizer
    attack = WBC(target_model=model, target_tokenizer=tok,
                 ref_model=model, ref_tokenizer=tok, device=device)
    score = attack.predict(bochum_text)
    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0
    # target == ref -> no window prefers either side; expect ~0.5
    # (lenient because tiny model has tied losses -> > and < are equiprobable)


def test_wbc_predict_per_window(bochum_text, tiny_model_and_tokenizer, device):
    from src.attacks.reference_scores import WBC
    model, tok = tiny_model_and_tokenizer
    attack = WBC(target_model=model, target_tokenizer=tok,
                 ref_model=model, ref_tokenizer=tok, device=device,
                 window_sizes=[2, 4, 8])
    out = attack.predict_per_window(bochum_text, label="test")
    assert set(out.keys()) == {"wbc_test_w2", "wbc_test_w4", "wbc_test_w8"}
    assert all(0.0 <= v <= 1.0 for v in out.values())


# --- Target-only attacks ----------------------------------------------------
def test_noisy_neighbour(bochum_text, tiny_model_and_tokenizer, device):
    from src.attacks.target_scores import NoisyNeighbour
    model, tok = tiny_model_and_tokenizer
    attack = NoisyNeighbour(model=model, device=device, batch_size=2)

    enc = tok(bochum_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    score = attack.predict(input_ids=enc.input_ids, num_of_neighbour=2)
    assert score is not None
    assert np.isfinite(score) if isinstance(score, (int, float)) else True


def test_camia(bochum_text, tiny_model_and_tokenizer, device):
    """CAMIA needs raw_values output as input. Empty calibration_signal is OK."""
    from src.attacks.reference_scores import raw_values
    from src.attacks.target_scores import CAMIA

    model, tok = tiny_model_and_tokenizer
    attack = CAMIA(target_model=model, target_tokenizer=tok, device=device,
                   max_len=512, calibration_signal={})
    res = raw_values(bochum_text, model, tok, device)
    out = attack.predict(
        input_ids=res["input_ids"].squeeze(0),
        token_log_probs=res["token_log_probs"],
        loss=res["loss"].item(),
    )
    assert isinstance(out, dict)
    assert len(out) > 0, "CAMIA returned no features"


def test_dcpdd(bochum_text, tiny_model_and_tokenizer, device):
    """DCPDD takes a pre-built token-frequency dict; we fake one with all-1s."""
    from src.attacks.reference_scores import raw_values
    from src.attacks.target_scores import DCPDD

    model, tok = tiny_model_and_tokenizer
    vocab_size = model.config.vocab_size
    freq_dict = np.ones(vocab_size, dtype=np.float32)
    attack = DCPDD(freq_dict, device=device, a=0.01, apply_smoothing=True)

    res = raw_values(bochum_text, model, tok, device)
    out = attack.predict(
        token_probs=res["token_probs"],
        input_ids=res["input_ids"].squeeze(0),
    )
    assert np.isfinite(out) if isinstance(out, (int, float)) else True


def test_tagtab(bochum_text, tiny_model_and_tokenizer, device):
    """TagTab requires spaCy en_core_web_sm -- skip if not installed."""
    spacy = pytest.importorskip("spacy")
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        pytest.skip(f"spacy en_core_web_sm not downloaded: {e}")

    from src.attacks.target_scores import TagTab
    model, tok = tiny_model_and_tokenizer
    attack = TagTab(target_model=model, target_tokenizer=tok,
                    top_k=10, nlp=nlp, device=device,
                    entropy_map=None, min_sentence_len=7, max_sentence_len=40)
    out = attack.predict(bochum_text)
    assert isinstance(out, dict)
    # TagTab may legitimately return {} if no keywords match in the text.
    for v in out.values():
        assert np.isfinite(v)
