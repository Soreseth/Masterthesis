"""Shared fixtures for MIA pipeline tests."""

import sys
import os
import pytest
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

HF_DIR = "/lustre/selvaah3/hf_home"
FREQ_DIST_PATH = "/lustre/selvaah3/projects/Masterthesis/GPTNeoXTokenizerFast_realnewslike_freq_dist.pkl"

# Sample texts from The Pile (Pile-CC subset) for testing
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet and is commonly used for testing purposes.",
    "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. These systems improve their performance over time without being explicitly programmed.",
    "In 2023, researchers at several universities published groundbreaking papers on membership inference attacks against large language models. These attacks aim to determine whether specific data was used during training.",
    "Python is a high-level programming language known for its readability and versatility. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
    "The Pile is a large-scale curated dataset designed for training language models. It consists of 22 diverse sub-datasets including academic papers, books, and web content.",
]

SHORT_TEXT = "Hello world."
MEDIUM_TEXT = SAMPLE_TEXTS[0]
LONG_TEXT = " ".join(SAMPLE_TEXTS)


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def device():
    return _get_device()


@pytest.fixture(scope="session")
def cpu_device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def pythia_70m(device):
    """Load pythia-70m (smallest model) for fast tests."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = f"{HF_DIR}/models/EleutherAI__pythia-70m"
    if not os.path.exists(path):
        pytest.skip(f"Model not found at {path}")

    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, return_dict=True,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


@pytest.fixture(scope="session")
def pythia_2_8b(device):
    """Load pythia-2.8b (target model) for integration tests."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = f"{HF_DIR}/models/EleutherAI__pythia-2.8b"
    if not os.path.exists(path):
        pytest.skip(f"Model not found at {path}")

    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, return_dict=True,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


@pytest.fixture(scope="session")
def freq_dist():
    """Load frequency distribution for DC-PDD."""
    import pickle
    if not os.path.exists(FREQ_DIST_PATH):
        pytest.skip(f"Freq dist not found at {FREQ_DIST_PATH}")
    with open(FREQ_DIST_PATH, "rb") as f:
        return np.array(pickle.load(f), dtype=np.float32)


@pytest.fixture(scope="session")
def raw_data_70m(pythia_70m, device):
    """Pre-compute raw_values for MEDIUM_TEXT using pythia-70m."""
    from scores import raw_values
    model, tokenizer = pythia_70m
    return raw_values(MEDIUM_TEXT, model, tokenizer, device)
