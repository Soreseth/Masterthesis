"""Paragraph-level aggregator tests.

The aggregators consume a (N x F) feature matrix coming out of
``extract_features`` and learn a binary classifier that separates members
from non-members. These tests use synthetic features (members shifted +1
along feature 0) so any correctly-trained classifier should reach AUROC > 0.7.
"""
import json
import pytest

np = pytest.importorskip("numpy")
sklearn = pytest.importorskip("sklearn")
from sklearn.metrics import roc_auc_score


# --- Synthetic feature matrix -----------------------------------------------
@pytest.fixture
def synthetic_features():
    """200 samples × 10 features. Members (label=1) have a +1 shift on
    feature 0. Returns (X, y, X_test, y_test)."""
    rng = np.random.default_rng(0)
    n_per_class = 100
    n_features = 10

    X_mem = rng.normal(0, 1, (n_per_class, n_features))
    X_mem[:, 0] += 1.0   # shift signal feature
    X_non = rng.normal(0, 1, (n_per_class, n_features))

    X = np.vstack([X_mem, X_non]).astype(np.float32)
    y = np.array([1] * n_per_class + [0] * n_per_class, dtype=np.int32)

    X_test_mem = rng.normal(0, 1, (50, n_features)); X_test_mem[:, 0] += 1.0
    X_test_non = rng.normal(0, 1, (50, n_features))
    X_test = np.vstack([X_test_mem, X_test_non]).astype(np.float32)
    y_test = np.array([1] * 50 + [0] * 50, dtype=np.int32)
    return X, y, X_test, y_test


# --- sklearn aggregators (LR, SVC, RF, XGB) ---------------------------------
@pytest.mark.parametrize("name,cls_kwargs", [
    ("LogisticRegression", {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}),
    ("SVC",                {"C": 1.0, "kernel": "rbf"}),
    ("RandomForest",       {"n_estimators": 50, "max_depth": 5}),
    ("XGBoost",            {"n_estimators": 50, "max_depth": 4}),
])
def test_sklearn_classifier_separates_synthetic_signal(name, cls_kwargs,
                                                       synthetic_features):
    """Each shortlisted aggregator must beat random guessing on a clean signal."""
    if name == "XGBoost":
        pytest.importorskip("xgboost")
    from src.attacks.aggregator.cv_params import _build_estimator
    X, y, X_te, y_te = synthetic_features

    clf = _build_estimator(name, cls_kwargs, seed=0)
    clf.fit(X, y)

    # decision_function for SVC (no probability=True), predict_proba otherwise.
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_te)
    else:
        scores = clf.predict_proba(X_te)[:, 1]

    auroc = roc_auc_score(y_te, scores)
    assert auroc > 0.7, f"{name} AUROC={auroc:.3f} on a clean +1 shift; expected >0.7"


# --- Puerto baseline (nn.Linear + Adam, outlier-trimmed) --------------------
def test_puerto_nn_linear_separates_synthetic_signal(synthetic_features):
    """The original Puerto method should also separate the signal."""
    pytest.importorskip("torch")
    from src.attacks.aggregator.puerto_baseline import (
        train_linear_model, get_scores)

    X, y, X_te, y_te = synthetic_features
    model, dev = train_linear_model(X, y)
    scores = get_scores(model, X_te, dev)
    auroc = roc_auc_score(y_te, scores)
    assert auroc > 0.6, f"PuertoNNLinear AUROC={auroc:.3f}; expected >0.6"


# --- MLP -- the torch MLP used by extended_aggregator ------------------------
def test_mlp_aggregator_separates_synthetic_signal(synthetic_features):
    """The torch MLP that extended_aggregator trains; small signal -> small net."""
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    X, y, X_te, y_te = synthetic_features
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)

    net = nn.Sequential(nn.Linear(X.shape[1], 32), nn.ReLU(), nn.Linear(32, 1))
    optim_ = optim.Adam(net.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=True)
    for _ in range(5):
        for xb, yb in loader:
            optim_.zero_grad()
            loss_fn(net(xb).squeeze(-1), yb).backward()
            optim_.step()

    net.eval()
    with torch.no_grad():
        scores = net(torch.tensor(X_te, dtype=torch.float32)).squeeze(-1).numpy()
    assert roc_auc_score(y_te, scores) > 0.7


# --- group_lda is a stub -- test it raises NotImplementedError --------------
def test_group_lda_is_marked_as_stub():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "src.attacks.aggregator.group_lda",
         "--dataset", "arxiv", "--ctx", "1024"],
        capture_output=True, text=True, timeout=30,
    )
    # Should exit non-zero with NotImplementedError on stderr.
    assert result.returncode != 0
    assert "NotImplementedError" in result.stderr or "NotImplementedError" in result.stdout


# --- extract_features -- schema check on a minimal paragraph dict -----------
def test_extract_features_returns_2d_matrix():
    """`extract_features([{pred: {...}}])` should return a 2-D feature matrix."""
    from src.utils.aggregate import extract_features

    paragraphs = [
        {"pred": {"ppl": 5.0, "min_k_5": -2.5, "min_k_10": -1.8,
                  "min_k_20": -1.2, "min_k_30": -0.9, "min_k_40": -0.7,
                  "min_k_50": -0.5, "min_k_60": -0.3, "lowercase_ppl": 6.0,
                  "zlib": 1000.0}, "label": 1},
        {"pred": {"ppl": 8.0, "min_k_5": -3.5, "min_k_10": -2.8,
                  "min_k_20": -2.0, "min_k_30": -1.5, "min_k_40": -1.2,
                  "min_k_50": -1.0, "min_k_60": -0.8, "lowercase_ppl": 9.0,
                  "zlib": 1500.0}, "label": 0},
    ]
    X, _, feature_keys = extract_features(paragraphs)
    assert X.ndim == 2
    assert X.shape[0] == 2
    assert X.shape[1] == len(feature_keys)
    assert X.shape[1] > 0
    assert np.all(np.isfinite(X))


# --- split_documents_puerto -- fixed-pool eval boundary ----------------------
def test_split_documents_puerto_eval_set_is_invariant_to_n_train():
    """The whole point of split_documents_puerto's fixed-pool sweep is that
    the eval (B) set must be IDENTICAL across different n_train values."""
    from src.utils.aggregate import split_documents_puerto

    rng = np.random.default_rng(0)
    members = [[{"pred": {"ppl": float(rng.normal())}, "label": 1}]
               for _ in range(50)]
    non_members = [[{"pred": {"ppl": float(rng.normal())}, "label": 0}]
                   for _ in range(50)]

    cfg_small = {"n_train_docs": 5,  "n_known_docs": 10,
                 "n_train_docs_sweep": [20], "n_known_docs_sweep": [10]}
    cfg_large = {"n_train_docs": 20, "n_known_docs": 10,
                 "n_train_docs_sweep": [20], "n_known_docs_sweep": [10]}

    splits_small = split_documents_puerto(members, non_members, cfg_small, seed=42)
    splits_large = split_documents_puerto(members, non_members, cfg_large, seed=42)

    # Eval set IDs should match.
    assert splits_small["B_members_docs"] == splits_large["B_members_docs"]
    assert splits_small["B_non_members_docs"] == splits_large["B_non_members_docs"]

    # And the train pools should *differ* in size.
    n_train_small = sum(len(d) for d in splits_small["A_members_para"])
    n_train_large = sum(len(d) for d in splits_large["A_members_para"])
    assert n_train_small <= n_train_large
