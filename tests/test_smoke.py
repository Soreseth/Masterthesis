"""Smoke tests -- verifies every src module is importable from a fresh clone.

Run from repo root:
    pip install -r requirements.txt
    python -m pytest tests/ -v

Heavy-deps modules (torch, sklearn, ...) are imported behind a try/except that
skips the test if the dep isn't installed, so the same tests run both in CI
(minimal deps) and on the cluster (full env).
"""
import importlib
import pytest

INTERNAL_MODULES = [
    # No third-party imports at module load
    "src.utils.vectorized_stats",
    "src.utils.merge_jsonl",
    "src.evaluation.shap_xgboost",
]

HEAVY_MODULES = [
    # These pull torch / sklearn / spacy / xgboost at import time
    "src.utils.aggregate",
    "src.utils.preprocess",
    "src.utils.download",
    "src.attacks.target_scores",
    "src.attacks.reference_scores",
    "src.attacks.precompute_mia_scores",
    "src.attacks.aggregator.motivation_lazypredict",
    "src.attacks.aggregator.cv_params",
    "src.attacks.aggregator.puerto_baseline",
    "src.attacks.aggregator.extended_aggregator",
    "src.attacks.aggregator.majority_voting_agg",
    "src.attacks.aggregator.group_lda",
    "src.evaluation.run_stats",
    "src.evaluation.blind_baseline",
    "src.defended.dpsgd",
    "src.defended.duolearn",
]


@pytest.mark.parametrize("mod", INTERNAL_MODULES)
def test_internal_modules_import(mod):
    importlib.import_module(mod)


@pytest.mark.parametrize("mod", HEAVY_MODULES)
def test_heavy_modules_import(mod):
    try:
        importlib.import_module(mod)
    except ModuleNotFoundError as e:
        # Tolerate missing third-party deps; still flag missing INTERNAL ones.
        if e.name and e.name.startswith("src"):
            pytest.fail(f"{mod}: broken internal import -- {e}")
        pytest.skip(f"{mod}: third-party dep missing ({e.name})")


def test_vectorized_stats_mwu_matches_scipy():
    """Sanity-check our MWU against scipy on small inputs."""
    import numpy as np
    from scipy.stats import mannwhitneyu

    from src.utils.vectorized_stats import mannwhitneyu_u_only

    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, (3, 50))
    ref = rng.normal(0.5, 1, 100)

    ours = mannwhitneyu_u_only(samples, ref)
    for i, sample in enumerate(samples):
        scipy_u, _ = mannwhitneyu(sample, ref, alternative="two-sided")
        assert abs(ours[i] - scipy_u) < 1e-6, f"sample {i}: ours={ours[i]} scipy={scipy_u}"
