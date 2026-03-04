# Mayan-mT5: Bidirectional English/Spanish ↔ Q'eqchi' Translation Pipeline

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9+-brightgreen.svg)

## Overview
This repository contains the data generation and training pipeline for fine-tuning an mT5-based adapter model for bidirectional machine translation between English/Spanish and Q'eqchi'.

**Status:** Phase 1 complete. The accompanying academic research paper detailing the synthetic data curriculum, methodology, and metrics is currently under peer review. Phase 2 (synthetic/authentic mixture training) and Phase 3 (community-in-the-loop RLHF) is currently in active planning.

## Hugging Face Assets
The resulting artifacts from this pipeline (datasets and adapter weights) are hosted separately on Hugging Face:
* **Model Checkpoint (22000):** *Coming soon*
* **Training Datasets:** *Coming soon*

*(Note: The `mT5_train_v4.jsonl` and validation sets are omitted from this GitHub repository due to file size constraints and are managed entirely via the Hugging Face `datasets` library).*

## Repository Structure
The codebase is modularized into three primary workflows:

* `/generator`: Contains the rule-based synthetic data generator. The `data/kek/` subdirectory houses the lightweight, vetted seed CSV files used by the generator program. Master `.xlsx` files are used locally for staging and iterating on grammatical rules.
* `/training`: Contains the scripts used for fine-tuning the base `google/mt5-base` model using PEFT/LoRA.
* `/analysis`: Contains the quantitative output of the training runs, including inference results, `metrics_summary.csv`, and comprehensive training logs.

**Utility Scripts:**
Additional tools are available in `generator/tools/` and `/training/utils/`. This includes `add_frequencies.py`, which applies Zipf's Law distributions to the lexicon CSV files to ensure generated sentences reflect natural language word frequencies.

---

## Installation

Clone the repository and set up your virtual environment:

```bash
git clone [https://github.com/achulzhanov/mayan-mt5.git](https://github.com/achulzhanov/mayan-mt5.git)
cd mayan-mt5
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

Install the required dependencies for your specific workflow:

```bash
# For generator dependencies
pip install -r generator/requirements.txt

# For training dependencies
pip install -r training/trainer/requirements.txt
```

---

## 1. Generating Synthetic Data

The generator utilizes CSV seed files and grammatical templates to produce synthetic parallel translation pairs. When using the `--jsonl_output` argument, the program automatically formats the output for bidirectional training (e.g., generating both `train_en_kek.jsonl` and `train_es_kek.jsonl`). `data-split.py` in `/training/utils/` is used to prepare and split the generated output into the final training and validation files.

### Basic Generation
```bash
python -m qeqchi_generator.main --n 10000 --jsonl_output data/synthetic/train
```

### Generator Arguments

| **Flag** | **Type** | **Description** | **Default** |
| :--- | :--- | :--- | :--- |
| `--n` | `int` | **Target Goal.** Total number of valid sentences to generate. | `10` |
| `--batch_size` | `int` | **Save Interval.** Number of sentences to generate per save cycle. | `1000` |
| `--api_batch_size` | `int` | **API Prompt Size.** Sentences sent per Gemini request (if filtering). | `50` |
| `--jsonl_output` | `str` | **Primary Output.** Base path for saving mirrored JSONL files. | `None` |
| `--out` | `str` | **Fallback Output.** File path for a simple `.txt` or `.csv` dump. | `generated_output.txt` |
| `--use_gemini_filter` | `flag` | **Enable Filter.** Activates Vertex AI to remove nonsensical output. | `Disabled` |
| `--project_id` | `str` | **GCP Project.** Google Cloud Project ID (Required for filter). | `None` |
| `--location` | `str` | **GCP Region.** Vertex AI region for inference. | `us-central1` |
| `--person` | `str` | **Grammar Force.** Forces a specific grammatical person (e.g., `1S`). | `Random` |
| `--data_dir` | `str` | **Source Data.** Directory containing the seed dictionaries. | `./data` |

### Using the Gemini Semantic Filter (Vertex AI)

> **Important Note on Dataset Entropy:** We strongly advise against using the semantic filter for generating training corpora. While it successfully removes nonsensical translations, this artificial curation severely reduces the natural entropy and structural variation of the dataset. Empirical testing showed this reduction leads directly to early overfitting and mode collapse during the LoRA fine-tuning phase. This feature is preserved in the codebase strictly for archival purposes and methodological transparency. 

To use the optional Gemini API (Gemini 2.5 Pro) for these archival purposes, configure your Google Cloud environment first.

**Step 1: Install and Authenticate the Google Cloud CLI**
Install `gcloud` from the official Google Cloud documentation, then authenticate:
```bash
gcloud auth login
gcloud config set project your-gcp-id
gcloud auth application-default login
```

**Step 2: Enable the API**
Ensure the Vertex AI API is enabled on your project:
```bash
gcloud services enable aiplatform.googleapis.com
```

**Step 3: Run the Generator**
Execute the script with the filter flags enabled:
```bash
python -m qeqchi_generator.main --n 10000 --use_gemini_filter --project_id "your-gcp-id" --jsonl_output data/synthetic/filtered_train
```

---

## 2. Training the Adapter

The training script fine-tunes the base `google/mt5-base` model using PEFT/LoRA. Ensure you have your generated JSONL files ready before executing.

### Hardware Optimization Note
The local training script (`train-local-v2.py`) is optimized for Apple Silicon utilizing PyTorch's MPS (Metal Performance Shaders) backend. It does **not** require or utilize the Apple MLX framework at the time of this release.

### Basic Training Run
```bash
cd training/trainer
python train-local-v2.py --train_file data/mT5_train_v4.jsonl --valid_file data/mT5_val_mini_v4.jsonl
```

### Training Arguments

| **Flag** | **Type** | **Description** | **Default** |
| :--- | :--- | :--- | :--- |
| `--train_file` | `str` | Path to the training JSONL dataset. | **Required** |
| `--valid_file` | `str` | Path to the validation JSONL dataset. | **Required** |
| `--output_dir` | `str` | Directory to save the trained adapter weights. | `lora_output` |
| `--model_name` | `str` | The base Hugging Face model to fine-tune. | `google/mt5-base` |
| `--adapter_path` | `str` | Path to existing LoRA weights to resume training. | `None` |
| `--epochs` | `int` | Total number of training epochs. | `3` |
| `--train_batch_size` | `int` | Batch size for training steps. | `8` |
| `--learning_rate` | `float` | Peak learning rate for the optimizer. | `1e-4` |
| `--max_source_length`| `int` | Maximum token length for the input text. | `128` |
| `--max_target_length`| `int` | Maximum token length for the target text. | `128` |
| `--lora_r` | `int` | LoRA rank dimension. | `32` |
| `--lora_alpha` | `int` | LoRA alpha scaling parameter. | `32` |
| `--target_modules` | `list` | Model layers to apply LoRA. | `q, v, k, o, wi_0, wi_1, wo`|
| `--focal_gamma` | `float` | Gamma value for Focal Loss (0.0 = Cross Entropy). | `2.0` |

*(Additional hyperparameters like `--weight_decay`, `--warmup_ratio`, and `--grad_accum_steps` are available and detailed in the script source).*

## Citation
*Placeholder: Citation details will be added upon the successful publication of the accompanying research paper.*