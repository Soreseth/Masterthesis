"""Tests for preprocess.py — data preprocessing functions.

Run with:
    pytest tests/test_preprocess.py -v
"""

import sys
import os
import json
import tempfile
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conftest import SAMPLE_TEXTS, MEDIUM_TEXT


# ════════════════════════════════════════════════════════════════════
# TensorEncoder
# ════════════════════════════════════════════════════════════════════

class TestTensorEncoder:
    def test_tensor_to_json(self):
        from preprocess import TensorEncoder
        t = torch.tensor([1.0, 2.0, 3.0])
        result = json.dumps({"val": t}, cls=TensorEncoder)
        parsed = json.loads(result)
        assert parsed["val"] == [1.0, 2.0, 3.0]

    def test_numpy_array_to_json(self):
        from preprocess import TensorEncoder
        arr = np.array([1.5, 2.5])
        result = json.dumps({"val": arr}, cls=TensorEncoder)
        parsed = json.loads(result)
        assert parsed["val"] == [1.5, 2.5]

    def test_numpy_scalar_to_json(self):
        from preprocess import TensorEncoder
        val = np.float32(3.14)
        result = json.dumps({"val": val}, cls=TensorEncoder)
        parsed = json.loads(result)
        assert abs(parsed["val"] - 3.14) < 0.01

    def test_nested_structure(self):
        from preprocess import TensorEncoder
        data = {
            "loss": torch.tensor(2.5),
            "scores": np.array([1.0, 2.0]),
            "count": 42
        }
        result = json.dumps(data, cls=TensorEncoder)
        parsed = json.loads(result)
        assert abs(parsed["loss"] - 2.5) < 0.01
        assert parsed["scores"] == [1.0, 2.0]
        assert parsed["count"] == 42


# ════════════════════════════════════════════════════════════════════
# create_chunks
# ════════════════════════════════════════════════════════════════════

class TestCreateChunks:
    def test_paragraph_chunking(self, pythia_70m):
        from preprocess import create_chunks
        _, tokenizer = pythia_70m
        text = " ".join(SAMPLE_TEXTS)
        chunks = create_chunks(text, tokenizer, max_length=512)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

    def test_small_max_length(self, pythia_70m):
        from preprocess import create_chunks
        _, tokenizer = pythia_70m
        text = " ".join(SAMPLE_TEXTS)
        chunks = create_chunks(text, tokenizer, max_length=32)
        # Should produce multiple small chunks
        assert len(chunks) > 1

    def test_sentence_chunking(self, pythia_70m):
        from preprocess import create_chunks
        _, tokenizer = pythia_70m
        text = "First sentence. Second sentence. Third sentence."
        chunks = create_chunks(text, tokenizer, max_length=43)
        # max_length=43 triggers sentence tokenization
        assert len(chunks) >= 2


# ════════════════════════════════════════════════════════════════════
# safe_pre_encode_shots
# ════════════════════════════════════════════════════════════════════

class TestSafePreEncodeShots:
    def test_empty_list(self, pythia_70m):
        from preprocess import safe_pre_encode_shots
        _, tokenizer = pythia_70m
        result = safe_pre_encode_shots([], tokenizer, max_shot_len=256)
        assert result == []

    def test_single_shot(self, pythia_70m):
        from preprocess import safe_pre_encode_shots
        _, tokenizer = pythia_70m
        result = safe_pre_encode_shots(
            [SAMPLE_TEXTS[0]], tokenizer, max_shot_len=256
        )
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        assert result[0].shape[1] <= 256

    def test_multiple_shots(self, pythia_70m):
        from preprocess import safe_pre_encode_shots
        _, tokenizer = pythia_70m
        result = safe_pre_encode_shots(
            SAMPLE_TEXTS[:3], tokenizer, max_shot_len=128
        )
        assert len(result) == 3
        for enc in result:
            assert enc.shape[1] <= 128

    def test_truncation_when_overflow(self, pythia_70m):
        from preprocess import safe_pre_encode_shots
        _, tokenizer = pythia_70m
        # 10 long texts with tiny context window should trigger truncation
        long_texts = [SAMPLE_TEXTS[0]] * 10
        result = safe_pre_encode_shots(
            long_texts, tokenizer, max_shot_len=512,
            reserve_for_target=400, context_window=1024
        )
        assert len(result) == 10
        total_tokens = sum(enc.shape[1] for enc in result)
        # Should be within context_window - reserve_for_target
        assert total_tokens <= 1024


# ════════════════════════════════════════════════════════════════════
# flatten_and_clean_chunk
# ════════════════════════════════════════════════════════════════════

class TestFlattenAndCleanChunk:
    def test_flat_dict(self):
        from preprocess import flatten_and_clean_chunk
        chunk = {"loss": -2.5, "ppl": 12.0, "zlib": 45}
        result = flatten_and_clean_chunk(chunk)
        assert result == {"loss": -2.5, "ppl": 12.0, "zlib": 45}

    def test_nested_dict(self):
        from preprocess import flatten_and_clean_chunk
        chunk = {
            "loss": -2.5,
            "acmia": {"AC_0.1": 0.5, "AC_0.3": 0.7}
        }
        result = flatten_and_clean_chunk(chunk)
        assert "acmia_AC_0.1" in result
        assert "acmia_AC_0.3" in result
        assert result["acmia_AC_0.1"] == 0.5

    def test_list_wrapper(self):
        from preprocess import flatten_and_clean_chunk
        chunk = {"ppl": [10.5], "loss": [-2.3]}
        result = flatten_and_clean_chunk(chunk)
        assert result["ppl"] == 10.5
        assert result["loss"] == -2.3

    def test_empty_list(self):
        from preprocess import flatten_and_clean_chunk
        chunk = {"empty": []}
        result = flatten_and_clean_chunk(chunk)
        assert result["empty"] == 0.0

    def test_nested_list_in_dict(self):
        from preprocess import flatten_and_clean_chunk
        chunk = {"acmia": {"AC_0.1": [-11.12]}}
        result = flatten_and_clean_chunk(chunk)
        assert result["acmia_AC_0.1"] == -11.12


# ════════════════════════════════════════════════════════════════════
# KEY_FIXES and fix_chunk_keys
# ════════════════════════════════════════════════════════════════════

class TestKeyFixes:
    def test_fix_chunk_keys(self):
        from preprocess import fix_chunk_keys, KEY_FIXES
        chunk = {
            "acmia_AC_0.30000000000000004": 0.5,
            "loss": -2.0,
            "acmia_AC_0.5000000000000001": 0.7,
        }
        fixed = fix_chunk_keys(chunk)
        assert "acmia_AC_0.3" in fixed
        assert "acmia_AC_0.5" in fixed
        assert "loss" in fixed
        assert fixed["acmia_AC_0.3"] == 0.5

    def test_fix_jsonl_file(self):
        from preprocess import fix_jsonl_file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            doc = {
                "pred": [
                    {"acmia_AC_0.30000000000000004": 0.5, "loss": -2.0},
                    {"acmia_AC_0.5000000000000001": 0.7, "ppl": 10.0},
                ],
                "label": 1
            }
            f.write(json.dumps(doc) + "\n")
            f.flush()
            path = f.name

        try:
            count = fix_jsonl_file(path)
            assert count == 1

            with open(path) as f:
                fixed_doc = json.loads(f.readline())
            assert "acmia_AC_0.3" in fixed_doc["pred"][0]
            assert "acmia_AC_0.5" in fixed_doc["pred"][1]
        finally:
            os.unlink(path)
