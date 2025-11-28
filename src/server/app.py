"""
FastAPI server for model inference with a web UI.
"""
import os
import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import yaml

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model import LanguageModel

app = FastAPI(
    title="HowToLLM Inference Server",
    description="A FastAPI server for running inference on the trained language model",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None
config = None


class GenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str
    max_length: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    generated_text: str
    prompt: str
    tokens_generated: int


class ModelLoadRequest(BaseModel):
    """Request model for loading the model."""
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    config_path: Optional[str] = None


class ModelStatus(BaseModel):
    """Response model for model status."""
    loaded: bool
    model_path: Optional[str] = None
    device: Optional[str] = None
    vocab_size: Optional[int] = None
    d_model: Optional[int] = None
    num_layers: Optional[int] = None


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_tokenizer(tokenizer_path: str):
    """Load the tokenizer from the specified path."""
    from tokenizers import Tokenizer
    tokenizer_file = Path(tokenizer_path) / "tokenizer.json"
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_file}")
    return Tokenizer.from_file(str(tokenizer_file))


def load_model_weights(model: LanguageModel, model_path: str, device: torch.device):
    """Load model weights from checkpoint."""
    checkpoint_path = Path(model_path)
    
    # Find the latest checkpoint
    if checkpoint_path.is_dir():
        checkpoints = list(checkpoint_path.glob("*.pt")) + list(checkpoint_path.glob("*.pth"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {model_path}")
        checkpoint_file = max(checkpoints, key=os.path.getctime)
    else:
        checkpoint_file = checkpoint_path
    
    print(f"Loading model from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    return model


@torch.no_grad()
def generate_text(
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95
) -> tuple[str, int]:
    """Generate text from the model given a prompt."""
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        raise ValueError("Model or tokenizer not loaded")
    
    model.eval()
    
    # Encode the prompt
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    
    generated_tokens = 0
    
    for _ in range(max_length):
        # Get model output
        outputs = model(input_ids.transpose(0, 1))  # Model expects (seq_len, batch)
        logits = outputs[-1, 0, :]  # Get last token logits
        
        # Apply temperature
        logits = logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the distribution
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to input
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        generated_tokens += 1
        
        # Check for EOS token
        if hasattr(tokenizer, 'token_to_id'):
            eos_id = tokenizer.token_to_id('[EOS]') or tokenizer.token_to_id('</s>')
            if eos_id and next_token.item() == eos_id:
                break
    
    # Decode the generated text
    generated_text = tokenizer.decode(input_ids[0].tolist())
    
    return generated_text, generated_tokens


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI page."""
    return get_html_ui()


@app.get("/status", response_model=ModelStatus)
async def get_status():
    """Get the current status of the model."""
    global model, device, config
    
    if model is None:
        return ModelStatus(loaded=False)
    
    return ModelStatus(
        loaded=True,
        model_path=config.get('model_path', 'Unknown') if config else 'Unknown',
        device=str(device),
        vocab_size=model.embedding.num_embeddings,
        d_model=model.d_model,
        num_layers=len(model.transformer_encoder.layers)
    )


@app.post("/load")
async def load_model_endpoint(request: ModelLoadRequest):
    """Load the model and tokenizer."""
    global model, tokenizer, device, config
    
    try:
        # Determine paths
        base_path = Path(__file__).parent.parent.parent
        config_path = request.config_path or str(base_path / "configs" / "train_config.yaml")
        
        # Load configuration
        config = load_config(config_path)
        
        model_path = request.model_path or config['training']['model_save_path']
        tokenizer_path = request.tokenizer_path or config['dataset']['tokenizer_dir']
        
        # Make paths absolute if relative
        if not Path(model_path).is_absolute():
            model_path = str(base_path / model_path)
        if not Path(tokenizer_path).is_absolute():
            tokenizer_path = str(base_path / tokenizer_path)
        
        config['model_path'] = model_path
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = load_tokenizer(tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()
        print(f"Loaded tokenizer with vocab size: {vocab_size}")
        
        # Create model
        model = LanguageModel(
            vocab_size=vocab_size,
            d_model=config['model']['d_model'],
            nhead=config['model']['nhead'],
            num_layers=config['model']['num_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout']
        )
        
        # Load weights
        model = load_model_weights(model, model_path, device)
        model = model.to(device)
        model.eval()
        
        print("Model loaded successfully!")
        
        return {
            "status": "success",
            "message": "Model loaded successfully",
            "device": str(device),
            "vocab_size": vocab_size
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Generate text from a prompt."""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Please load the model first.")
    
    try:
        generated_text, tokens_generated = generate_text(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )
        
        return GenerationResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            tokens_generated=tokens_generated
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload")
async def unload_model():
    """Unload the current model to free memory."""
    global model, tokenizer, config
    
    if model is not None:
        del model
        model = None
    
    if tokenizer is not None:
        del tokenizer
        tokenizer = None
    
    config = None
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {"status": "success", "message": "Model unloaded successfully"}


def get_html_ui() -> str:
    """Return the HTML UI for the inference server."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HowToLLM Inference</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e4e4e4;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        h1 {
            color: #00d4ff;
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #888;
            font-size: 1rem;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        
        .card h2 {
            color: #00d4ff;
            margin-bottom: 20px;
            font-size: 1.3rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }
        
        .status-loaded {
            background: #00ff88;
            box-shadow: 0 0 10px #00ff88;
        }
        
        .status-unloaded {
            background: #ff4444;
            box-shadow: 0 0 10px #ff4444;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #aaa;
            font-size: 0.9rem;
        }
        
        input[type="text"],
        input[type="number"],
        textarea {
            width: 100%;
            padding: 12px 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.3);
            color: #fff;
            font-size: 1rem;
            transition: border-color 0.3s;
        }
        
        input:focus,
        textarea:focus {
            outline: none;
            border-color: #00d4ff;
        }
        
        textarea {
            min-height: 120px;
            resize: vertical;
            font-family: inherit;
        }
        
        .slider-container {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        input[type="range"] {
            flex: 1;
            -webkit-appearance: none;
            height: 6px;
            border-radius: 3px;
            background: rgba(255, 255, 255, 0.2);
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #00d4ff;
            cursor: pointer;
        }
        
        .slider-value {
            min-width: 50px;
            text-align: center;
            color: #00d4ff;
            font-weight: bold;
        }
        
        .params-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        
        button {
            padding: 12px 30px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            color: #000;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0, 212, 255, 0.4);
        }
        
        .btn-secondary {
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        
        .btn-danger {
            background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
            color: #fff;
        }
        
        .btn-group {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .output-box {
            background: rgba(0, 0, 0, 0.4);
            border-radius: 8px;
            padding: 20px;
            min-height: 150px;
            white-space: pre-wrap;
            font-family: 'Consolas', 'Monaco', monospace;
            line-height: 1.6;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .output-box.empty {
            color: #666;
            font-style: italic;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .loading.active {
            display: block;
        }
        
        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid rgba(0, 212, 255, 0.3);
            border-top-color: #00d4ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .stats {
            display: flex;
            gap: 20px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        
        .stat-item {
            background: rgba(0, 212, 255, 0.1);
            padding: 10px 20px;
            border-radius: 8px;
            border: 1px solid rgba(0, 212, 255, 0.3);
        }
        
        .stat-label {
            font-size: 0.8rem;
            color: #888;
        }
        
        .stat-value {
            font-size: 1.2rem;
            color: #00d4ff;
            font-weight: bold;
        }
        
        .error-message {
            background: rgba(255, 68, 68, 0.2);
            border: 1px solid #ff4444;
            border-radius: 8px;
            padding: 15px;
            color: #ff6666;
            margin-top: 10px;
            display: none;
        }
        
        .error-message.active {
            display: block;
        }

        .model-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }

        .info-item {
            background: rgba(0, 0, 0, 0.2);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }

        .info-label {
            font-size: 0.75rem;
            color: #666;
            text-transform: uppercase;
        }

        .info-value {
            font-size: 1rem;
            color: #00d4ff;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 HowToLLM</h1>
            <p class="subtitle">Transformer Language Model Inference Server</p>
        </header>
        
        <!-- Model Status Card -->
        <div class="card">
            <h2>
                <span class="status-indicator status-unloaded" id="statusIndicator"></span>
                Model Status
            </h2>
            <div id="modelStatus">Model not loaded</div>
            <div class="model-info" id="modelInfo" style="display: none;">
                <div class="info-item">
                    <div class="info-label">Device</div>
                    <div class="info-value" id="infoDevice">-</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Vocab Size</div>
                    <div class="info-value" id="infoVocab">-</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Model Dim</div>
                    <div class="info-value" id="infoDModel">-</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Layers</div>
                    <div class="info-value" id="infoLayers">-</div>
                </div>
            </div>
            <div class="btn-group" style="margin-top: 20px;">
                <button class="btn-primary" onclick="loadModel()">Load Model</button>
                <button class="btn-danger" onclick="unloadModel()">Unload Model</button>
            </div>
            <div class="error-message" id="loadError"></div>
        </div>
        
        <!-- Generation Card -->
        <div class="card">
            <h2>✨ Text Generation</h2>
            
            <div class="form-group">
                <label for="prompt">Prompt</label>
                <textarea id="prompt" placeholder="Enter your prompt here...">Once upon a time</textarea>
            </div>
            
            <div class="params-grid">
                <div class="form-group">
                    <label>Max Length: <span id="maxLengthValue">100</span></label>
                    <div class="slider-container">
                        <input type="range" id="maxLength" min="10" max="500" value="100" 
                               oninput="document.getElementById('maxLengthValue').textContent = this.value">
                    </div>
                </div>
                
                <div class="form-group">
                    <label>Temperature: <span id="temperatureValue">1.0</span></label>
                    <div class="slider-container">
                        <input type="range" id="temperature" min="0.1" max="2" step="0.1" value="1.0"
                               oninput="document.getElementById('temperatureValue').textContent = this.value">
                    </div>
                </div>
                
                <div class="form-group">
                    <label>Top-K: <span id="topKValue">50</span></label>
                    <div class="slider-container">
                        <input type="range" id="topK" min="1" max="100" value="50"
                               oninput="document.getElementById('topKValue').textContent = this.value">
                    </div>
                </div>
                
                <div class="form-group">
                    <label>Top-P: <span id="topPValue">0.95</span></label>
                    <div class="slider-container">
                        <input type="range" id="topP" min="0.1" max="1" step="0.05" value="0.95"
                               oninput="document.getElementById('topPValue').textContent = this.value">
                    </div>
                </div>
            </div>
            
            <button class="btn-primary" onclick="generate()" id="generateBtn">Generate Text</button>
            <div class="error-message" id="generateError"></div>
        </div>
        
        <!-- Output Card -->
        <div class="card">
            <h2>📝 Generated Output</h2>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <div>Generating...</div>
            </div>
            
            <div class="output-box empty" id="output">Generated text will appear here...</div>
            
            <div class="stats" id="stats" style="display: none;">
                <div class="stat-item">
                    <div class="stat-label">Tokens Generated</div>
                    <div class="stat-value" id="tokensGenerated">0</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Check model status on page load
        window.onload = function() {
            checkStatus();
        };
        
        async function checkStatus() {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                updateStatusUI(data);
            } catch (error) {
                console.error('Error checking status:', error);
            }
        }
        
        function updateStatusUI(data) {
            const indicator = document.getElementById('statusIndicator');
            const statusText = document.getElementById('modelStatus');
            const modelInfo = document.getElementById('modelInfo');
            
            if (data.loaded) {
                indicator.className = 'status-indicator status-loaded';
                statusText.textContent = 'Model loaded and ready';
                modelInfo.style.display = 'grid';
                
                document.getElementById('infoDevice').textContent = data.device || '-';
                document.getElementById('infoVocab').textContent = data.vocab_size?.toLocaleString() || '-';
                document.getElementById('infoDModel').textContent = data.d_model || '-';
                document.getElementById('infoLayers').textContent = data.num_layers || '-';
            } else {
                indicator.className = 'status-indicator status-unloaded';
                statusText.textContent = 'Model not loaded';
                modelInfo.style.display = 'none';
            }
        }
        
        async function loadModel() {
            const errorDiv = document.getElementById('loadError');
            errorDiv.classList.remove('active');
            
            try {
                document.getElementById('modelStatus').textContent = 'Loading model...';
                
                const response = await fetch('/load', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to load model');
                }
                
                const data = await response.json();
                checkStatus();
                
            } catch (error) {
                errorDiv.textContent = error.message;
                errorDiv.classList.add('active');
                document.getElementById('modelStatus').textContent = 'Failed to load model';
            }
        }
        
        async function unloadModel() {
            try {
                await fetch('/unload', { method: 'POST' });
                checkStatus();
            } catch (error) {
                console.error('Error unloading model:', error);
            }
        }
        
        async function generate() {
            const errorDiv = document.getElementById('generateError');
            errorDiv.classList.remove('active');
            
            const prompt = document.getElementById('prompt').value;
            const maxLength = parseInt(document.getElementById('maxLength').value);
            const temperature = parseFloat(document.getElementById('temperature').value);
            const topK = parseInt(document.getElementById('topK').value);
            const topP = parseFloat(document.getElementById('topP').value);
            
            const loading = document.getElementById('loading');
            const output = document.getElementById('output');
            const stats = document.getElementById('stats');
            const generateBtn = document.getElementById('generateBtn');
            
            loading.classList.add('active');
            output.style.display = 'none';
            stats.style.display = 'none';
            generateBtn.disabled = true;
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt: prompt,
                        max_length: maxLength,
                        temperature: temperature,
                        top_k: topK,
                        top_p: topP
                    })
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Generation failed');
                }
                
                const data = await response.json();
                
                output.textContent = data.generated_text;
                output.classList.remove('empty');
                output.style.display = 'block';
                
                document.getElementById('tokensGenerated').textContent = data.tokens_generated;
                stats.style.display = 'flex';
                
            } catch (error) {
                errorDiv.textContent = error.message;
                errorDiv.classList.add('active');
                output.style.display = 'block';
            } finally {
                loading.classList.remove('active');
                generateBtn.disabled = false;
            }
        }
        
        // Allow Ctrl+Enter to generate
        document.getElementById('prompt').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                generate();
            }
        });
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


def main():
    """Entry point for the server."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)
