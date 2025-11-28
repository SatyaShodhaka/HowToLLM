# HowToLLM
This projects implements the fundamentals of training an LLM. 




### Installation

To install the required dependencies, run:

```bash
./setup/setup.sh

uv sync 
```

### PyTorch (install after `uv sync`)

Run one of the following commands depending on your hardware:

For most NVIDIA GPUs (RTX 3000/4000 series, etc.):
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

For RTX 5000 series (Blackwell architecture, sm_120):
```bash
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
```

For CPU only:
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```



### Activate uv environment
To activate the uv environment, run:

```bash
source .venv/bin/activate
```