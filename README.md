# Enhanced Membership Inference Attacks through Statistical Score Aggregation in Large Language Models

This repository contains code and experiments for my masterthesis with the title "Enhanced Membership Inference Attacks through Statistical Score Aggregation in Large Language Models". It implements multiple membership inference and detection methods, calibration/aggregation pipelines, and experiments to evaluate attacks on causal language models (e.g., the Pythia family).

Key features
- Implements a variety of attack methods and signal extraction utilities (CIMIA, ACMIA/AC, MaxRényi, DCPDD, NoisyNeighbour, TagTab, RelativeLikelihood and OfflineRobustMIA).
- Tools for dataset preprocessing and chunking at sentence/paragraph/document scales.
- Calibration signal collection and score aggregation for robust membership inference attacks (MIA).
- Experiment runner (run.py) to compute per-text and per-document scores and save outputs in JSONL format.


Repository structure
- Masterthesis/
  - run.py — main experiment runner (loads models, collects calibration signals, runs attacks, writes JSONL outputs)
  - preprocess.py — dataset loading, chunking, batched score computation, TopPref prefix search, calibration and utility functions
  - scores.py — attack implementations and signal extractors (CIMIA, DCPDD, MaxRenyi, ACMIA, NoisyNeighbour, OfflineRobustMIA, TagTab, etc.)
  - aggregation.py — small wrappers for classifier aggregation (XGBoost, SVM)
  - src/config.py — config constants and model name mappings
  - test.py — legacy/experimental scripts
  - GPTNeoXTokenizerFast_realnewslike_freq_dist.pkl — token frequency distribution used by DCPDD
  - LICENSE — license file

Requirements
- Python 3.8+
- PyTorch (with CUDA if available)
- transformers
- datasets
- scikit-learn
- xgboost
- nltk
- spacy (and language models, e.g., en_core_web_sm)
- wordfreq
- sentence_transformers (optional for some tests)
- pandas, numpy, tqdm

Install (example)

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install transformers datasets scikit-learn xgboost nltk spacy wordfreq sentence-transformers pandas tqdm

Note: adapt the PyTorch command to your CUDA and platform configuration.

Configuration and data
- The code expects local copies of Hugging Face models and datasets. Many paths are configured in run.py and preprocess.py using HF_DIR and MIA_SCORE_SAVING_DIR. By default these point to the author's filesystem. Edit HF_DIR and MIA_SCORE_SAVING_DIR in run.py or set environment variables accordingly before running.

- run.py expects preprocessed dataset directories at:
  output_mia/<pythia_model>/<dataset>/members
  output_mia/<pythia_model>/<dataset>/non_members

- After calibration and execution, outputs are written under:
  output_mia/<pythia_model>/<dataset>/paragraph_<max_length>/mia_members_*.jsonl
  output_mia/<pythia_model>/<dataset>/paragraph_<max_length>/mia_nonmembers_*.jsonl
  output_mia/<pythia_model>/<dataset>/document_<max_length>/mia_*.jsonl

Usage
- run.py will automatically collect calibration signals if calibration_signals.json is missing. Example invocation:

python3 run.py --pythia_model pythia-2.8b --max_length 512 --miaset nonmember --dataset YoutubeSubtitles --range 0 100

Arguments
- --pythia_model: model folder name (e.g., pythia-2.8b)
- --max_length: chunk length to evaluate (43, 512, 1024, 2048)
- --miaset: "member" or "nonmember" to process members or non-members
- --dataset: dataset short name used under output_mia
- --range: two integers specifying the document range (start end)

Important notes
- Many scripts run in an offline Hugging Face setup (HF_HUB_OFFLINE). Ensure required model files are present in HF_DIR or change local_files_only flags to allow downloads.
- run.py creates a simple lock directory during calibration to coordinate concurrent jobs.
- Several file paths in the code are hard-coded to the original author's environment (e.g. /lustre/...). Update these paths before running in your environment.

License
See the LICENSE file in the repository for the project license and terms.
