"""Document- and collection-level aggregation tests.

Synthesise a PCS dict where members get higher per-paragraph scores than
non-members, then run it through ``evaluate_paragraph``, ``evaluate_document``
(MWU + Brunner-Munzel), and ``evaluate_collection`` (Student's t + BM).
Each level should reach AUROC well above 0.5 on a clean signal.
"""
import pytest

np = pytest.importorskip("numpy")
sklearn = pytest.importorskip("sklearn")


# --- Synthetic PCS dict -----------------------------------------------------
@pytest.fixture
def synthetic_pcs():
    """Build a PCS dict with:
        - 50 known non-member paragraphs (loc=0)
        - 30 eval member docs    (3 paragraphs each, loc=1.0)
        - 30 eval non-member docs (3 paragraphs each, loc=0.0)
    """
    rng = np.random.default_rng(0)
    classifier = "TestClf"

    known = rng.normal(0, 1, 50).tolist()

    eval_members, eval_non_members = [], []
    for doc_idx in range(30):
        member_scores    = rng.normal(1.0, 1, 3).tolist()
        nonmember_scores = rng.normal(0.0, 1, 3).tolist()
        eval_members.append({
            "doc_id": f"mem_{doc_idx}", "label": 1, "n_paragraphs": 3,
            "scores": {classifier: member_scores},
        })
        eval_non_members.append({
            "doc_id": f"non_{doc_idx}", "label": 0, "n_paragraphs": 3,
            "scores": {classifier: nonmember_scores},
        })

    return {
        "seed": 0,
        "config": {"n_train": 100, "n_known": 50, "ctx": 1024},
        "models": [classifier],
        "model_params": {classifier: {"note": "synthetic"}},
        "known_scores": {classifier: known},
        "eval_members": eval_members,
        "eval_non_members": eval_non_members,
    }, classifier


# --- Paragraph-level: just AUROC over flattened scores ----------------------
def test_evaluate_paragraph(synthetic_pcs):
    from src.evaluation.run_stats import evaluate_paragraph
    pcs, clf = synthetic_pcs

    out = evaluate_paragraph(pcs, clf)
    assert "auroc" in out
    assert "n_samples" in out
    assert out["n_samples"] == 30 * 3 * 2     # 60 mem + 60 non-mem paragraphs
    assert out["auroc"] > 0.6, (
        f"paragraph AUROC={out['auroc']:.3f} on a +1.0 shift -- expected >0.6")


# --- Document-level: MWU + BM each turn per-paragraph distributions into a
#     per-document statistic, then AUROC over those statistics ---------------
@pytest.mark.parametrize("stat_test", ["mwu", "bm"])
def test_evaluate_document(synthetic_pcs, stat_test):
    from src.evaluation.run_stats import evaluate_document
    pcs, clf = synthetic_pcs

    out = evaluate_document(pcs, clf, stat_test=stat_test)
    assert "auroc" in out
    assert "n_samples" in out
    assert out["n_samples"] == 60               # 30 mem + 30 non-mem docs
    assert out["auroc"] > 0.6, (
        f"document AUROC ({stat_test})={out['auroc']:.3f}; expected >0.6")
    assert out.get("level") == "document"


# --- Collection-level: Student's t and BM over collections of docs ---------
@pytest.mark.parametrize("stat_test", ["ttest", "bm"])
def test_evaluate_collection(synthetic_pcs, stat_test):
    from src.evaluation.run_stats import evaluate_collection
    pcs, clf = synthetic_pcs

    coll_size = 5
    out = evaluate_collection(pcs, clf, coll_size, seed=42,
                              n_collections=20, stat_test=stat_test)
    assert "auroc" in out
    assert out["auroc"] > 0.7, (
        f"collection AUROC ({stat_test}, c={coll_size})={out['auroc']:.3f}; "
        f"expected >0.7 -- collection-level should amplify the signal")
    # Collection-level strictly stronger than paragraph for clean signal.
    from src.evaluation.run_stats import evaluate_paragraph
    para_auroc = evaluate_paragraph(pcs, clf)["auroc"]
    assert out["auroc"] >= para_auroc, (
        f"collection AUROC {out['auroc']:.3f} < paragraph AUROC {para_auroc:.3f}; "
        f"this should never happen for a clean +1.0 shift signal")


# --- TPR@FPR is reported alongside AUROC at every level ---------------------
def test_tpr_at_fpr_keys_present_at_every_level(synthetic_pcs):
    from src.evaluation.run_stats import (evaluate_collection, evaluate_document,
                                          evaluate_paragraph)
    pcs, clf = synthetic_pcs
    expected_tpr_keys = {"tpr@0.05fpr", "tpr@0.01fpr", "tpr@0.001fpr",
                         "tpr@0.0001fpr"}

    para = evaluate_paragraph(pcs, clf)
    doc = evaluate_document(pcs, clf, stat_test="mwu")
    coll = evaluate_collection(pcs, clf, 5, seed=0, n_collections=20,
                               stat_test="ttest")

    for name, out in [("paragraph", para), ("document", doc), ("collection", coll)]:
        missing = expected_tpr_keys - set(out.keys())
        assert not missing, f"{name}: missing TPR keys {missing}"
        for k in expected_tpr_keys:
            assert 0.0 <= out[k] <= 1.0, f"{name}: {k}={out[k]} out of [0,1]"


# --- Sign convention: members should have HIGHER per-doc statistic than
#     non-members under our convention -------------------------------------
def test_higher_member_scores_yield_auroc_above_half(synthetic_pcs):
    """If we negate all scores, the per-doc AUROC should drop below 0.5
    (sanity of the sign convention)."""
    from src.evaluation.run_stats import evaluate_document
    pcs, clf = synthetic_pcs

    auroc_pos = evaluate_document(pcs, clf, stat_test="mwu")["auroc"]

    # Build a copy with all scores negated.
    import copy
    neg_pcs = copy.deepcopy(pcs)
    for side in ("eval_members", "eval_non_members"):
        for doc in neg_pcs[side]:
            doc["scores"][clf] = [-x for x in doc["scores"][clf]]
    neg_pcs["known_scores"][clf] = [-x for x in neg_pcs["known_scores"][clf]]

    auroc_neg = evaluate_document(neg_pcs, clf, stat_test="mwu")["auroc"]
    assert auroc_pos > 0.5 and auroc_neg < 0.5, (
        f"sign convention broken: pos={auroc_pos:.3f}, neg={auroc_neg:.3f}")
