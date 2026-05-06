"""Shared pytest fixtures.

Heavy fixtures (a HF tokenizer + tiny model) are session-scoped so the
download/load cost is paid once across the whole test run.
"""
import os
import pytest

BOCHUM_TEXT = (
    "Unlike many older German universities, the buildings of Ruhr-University are "
    "all centralized on one campus, located south of Bochum city. The Faculty of "
    "Medicine includes several university clinics that are located at different "
    "centres in Bochum and the Ruhr area. A major facility for patient care is "
    "the University Hospital/Knappschaftskrankenhaus in the district Langendreer "
    "of Bochum. Internationally renowned experts in their respective fields include "
    "professors Wolff Schmiegel in oncology and Burkhard Dick in ophthalmology. "
    "The centralized university campus architecture is comprised almost exclusively "
    "of the 1960s architecture style referred to as brutalism, consisting of 14 "
    "almost identical high-rise buildings. One striking feature of these buildings "
    "is that although their roofs are all at the same apparent height (sky level), "
    "the absolute heights of the buildings vary in accordance with their placement "
    "on the undulating landscape in which the university is located: the campus is "
    "at the edge of a green belt on high ground adjacent to the Ruhr valley."
)


@pytest.fixture(scope="session")
def bochum_text() -> str:
    """The fixed sample text used to exercise every MIA attack."""
    return BOCHUM_TEXT


@pytest.fixture(scope="session")
def device():
    """Torch device (CUDA if available, else CPU). Skips if torch missing."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def tiny_model_name() -> str:
    """HF model id used for attack tests. Override with TEST_MODEL=<name>.

    Defaults to ``sshleifer/tiny-gpt2`` (~5 MB) so CI / a fresh laptop can
    download it in seconds. On a cluster, set TEST_MODEL=EleutherAI/pythia-70m
    to exercise a Pythia-shaped model (still <200 MB).
    """
    return os.environ.get("TEST_MODEL", "sshleifer/tiny-gpt2")


@pytest.fixture(scope="session")
def tiny_model_and_tokenizer(tiny_model_name, device):
    """Returns (model, tokenizer). Skips the test if the model can't be loaded
    (e.g. no internet on a fresh laptop with HF cache empty)."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tiny_model_name)
        model = AutoModelForCausalLM.from_pretrained(tiny_model_name).to(device)
    except Exception as e:
        pytest.skip(f"Could not load {tiny_model_name}: {e}")

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer
