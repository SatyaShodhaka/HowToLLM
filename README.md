# HowToLLM

HowToLLM is a step-by-step tutorial and codebase for building, training, and running inference with a Large Language Model (LLM) from scratch using PyTorch. This project is designed to be educational and approachable, even for those new to machine learning or deep learning.

---

## Table of Contents

- Project Structure
- Installation
- Configuration
- Key Components
	- Data Preparation
	- Tokenizer
	- Model
	- Training
	- Inference Server & UI
- How to Train Your Own LLM
- How to Run Inference
- System Monitoring
- File-by-File Guide
- Troubleshooting
- License

---

## Project Structure

```
HowToLLM/
├── configs/
│   └── train_config.yaml      # All model and training settings
├── data/
│   ├── dataset/               # Raw and processed text data
│   ├── logs/                  # Training logs
│   ├── model/                 # Saved model checkpoints
│   ├── tokenizer/             # Trained tokenizer files
├── outputs/                   # Output directories for experiments
├── setup/
│   ├── setup.sh               # Shell script to set up the environment
│   └── setup.ps1              # PowerShell setup script
├── src/
│   ├── model/
│   │   └── model.py           # The core LLM model definition
│   ├── train/
│   │   └── train.py           # Training script
│   ├── utils/
│   │   └── data.py            # Data loading and tokenizer training
│   ├── server/
│   │   └── app.py             # FastAPI server for inference and web UI
│   └── howtollm.egg-info/     # Python packaging info (auto-generated)
├── test_gpu.py                # Simple script to test GPU availability
├── pyproject.toml             # Python project and dependency management
├── README.md                  # This file
└── LICENSE                    # License file
```

---

## Installation

1. **Clone the repository:**
	 ```bash
	 git clone https://github.com/yourusername/HowToLLM.git
	 cd HowToLLM
	 ```

2. **Set up the environment:**
	 ```bash
	 ./setup/setup.sh
	 uv sync
	 ```

3. **Install PyTorch (choose your hardware):**
	 - For most NVIDIA GPUs:
		 ```bash
		 uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
		 ```
	 - For RTX 5000 series (Blackwell):
		 ```bash
		 uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
		 ```
	 - For CPU only:
		 ```bash
		 uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
		 ```

4. **Activate the environment:**
	 ```bash
	 source .venv/bin/activate
	 ```

---

## Configuration

All model and training settings are in [`configs/train_config.yaml`](configs/train_config.yaml ).  
You can adjust model size, training parameters, and data paths here.

Example:
```yaml
model:
	num_layers: 12
	d_model: 512
	nhead: 8
	dim_feedforward: 2048
	dropout: 0.1
	context_length: 4096
	vocab_size: 40629

training:
	batch_size: 8
	epochs: 5
	learning_rate: 5e-5
	weight_decay: 0.01
	warmup_steps: 1000
	model_save_path: data/model/model.pth

dataset:
	tokenizer_dir: data/tokenizer
	data_path: data/dataset/shake.txt
```

---

## Key Components

### Data Preparation

- **Raw data** goes in [`data/dataset`](data/dataset ) (e.g., `shake.txt`).
- You can use any large text file for training.

### Tokenizer

- The tokenizer splits text into tokens (words/subwords).
- Train a tokenizer with:
	```bash
	python src/utils/data.py data/dataset/your_text.txt --vocab_size 32000 --save_dir data/tokenizer
	```
- This creates `tokenizer.json` and metadata in [`data/tokenizer`](data/tokenizer ).

### Model

- Defined in [`src/model/model.py`](src/model/model.py ).
- Implements a Transformer-based language model (like GPT/Llama).
- Model size and architecture are set in `train_config.yaml`.

### Training

- Run training with:
	```bash
	python src/train/train.py
	```
- Logs metrics to MLflow and saves the model to [`data/model/model.pth`](data/model/model.pth ).

### Inference Server & UI

- Run the FastAPI server:
	```bash
	python src/server/app.py
	```
- Open [http://localhost:8000](http://localhost:8000) for a web UI to generate text with your model.

---

## How to Train Your Own LLM

1. **Prepare your dataset:**  
	 Place your text file in [`data/dataset`](data/dataset ).

2. **Train the tokenizer:**  
	 ```bash
	 python src/utils/data.py data/dataset/your_text.txt --vocab_size 32000 --save_dir data/tokenizer
	 ```

3. **Edit [`configs/train_config.yaml`](configs/train_config.yaml ):**  
	 Set the correct [`data_path`](src/train/train.py ), [`tokenizer_dir`](src/utils/data.py ), and model/training parameters.

4. **Train the model:**  
	 ```bash
	 python src/train/train.py
	 ```

5. **Monitor training:**  
	 MLflow will log metrics and system usage (if enabled).

---

## How to Run Inference

1. **Start the server:**
	 ```bash
	 python src/server/app.py
	 ```
2. **Open your browser:**  
	 Go to [http://localhost:8000](http://localhost:8000)
3. **Load the model and generate text** using the web UI.

---

## System Monitoring

To enable automatic logging of CPU, memory, and GPU usage during training, set:

```bash
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true
```

MLflow will then log system metrics alongside your training metrics.

---

## File-by-File Guide

### [`src/utils/data.py`](src/utils/data.py )
- **Purpose:** Data loading and tokenizer training.
- **Key functions:**
	- [`LMDataset`](src/utils/data.py ): Loads and tokenizes text for training.
	- `train_tokenizer`: Trains a BPE tokenizer on your dataset.
- **CLI usage:**  
	```bash
	python src/utils/data.py data/dataset/your_text.txt --vocab_size 32000 --save_dir data/tokenizer
	```

### [`src/model/model.py`](src/model/model.py )
- **Purpose:** Defines the Transformer-based language model.
- **Contents:**  
	- Model class with embedding, transformer layers, and output head.
	- Configurable via `train_config.yaml`.

### [`src/train/train.py`](src/train/train.py )
- **Purpose:** Training script for the LLM.
- **Contents:**  
	- Loads config, tokenizer, and dataset.
	- Initializes model, optimizer, and loss.
	- Runs training and validation loops.
	- Logs metrics to MLflow.
	- Saves the trained model.

### [`src/server/app.py`](src/server/app.py )
- **Purpose:** FastAPI server for inference and web UI.
- **Contents:**  
	- REST API endpoints for loading model, generating text, and checking status.
	- Web UI for interactive text generation.

### [`configs/train_config.yaml`](configs/train_config.yaml )
- **Purpose:** All model, training, and data settings in one place.

### [`setup/setup.sh`](setup/setup.sh )
- **Purpose:** Shell script to set up the Python environment and dependencies.

### [`pyproject.toml`](pyproject.toml )
- **Purpose:** Python project and dependency management for reproducible installs.

### [`test_gpu.py`](test_gpu.py )
- **Purpose:** Simple script to check if your GPU is available to PyTorch.

---

## Troubleshooting

- **CUDA out of memory:**  
	Lower [`batch_size`](.venv/lib/python3.12/site-packages/torch/utils/data/dataloader.py ) or [`context_length`](src/utils/data.py ) in `train_config.yaml`.
- **Tokenizer not found:**  
	Make sure to run the tokenizer training step before training the model.
- **MLflow not logging system metrics:**  
	Ensure you set `export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true` before training.

---

## License

See [`LICENSE`](LICENSE ) for details.

---

**This project is designed for learning and experimentation. For production or research use, see official LLM libraries and frameworks.**