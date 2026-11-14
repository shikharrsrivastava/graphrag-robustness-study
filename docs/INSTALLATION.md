# Installation and Setup Guide

This guide walks you through setting up and running the **Hybrid GraphRAG Engine** on local hardware.

---

## 1. System & Hardware Requirements

The Hybrid GraphRAG Engine is optimized to run completely offline on local consumer hardware without remote API fees.

### Recommended Specifications
| Component | Minimum | Recommended | Tested Configuration |
| :--- | :--- | :--- | :--- |
| **Operating System** | macOS 13+ (Ventura), Linux (Ubuntu 22.04+), Windows (WSL2) | macOS Sonoma / Sequoia | macOS Sequoia (Darwin 24.x) |
| **Processor (CPU)** | 4-core x86_64 or Apple Silicon (M1) | Apple Silicon (M1/M2/M3 Pro/Max) | Apple M2 (8-core CPU) |
| **RAM (Unified Memory)** | 8 GB | 16 GB unified RAM | 16 GB Unified Memory |
| **Storage (SSD)** | 10 GB free space | 25 GB free NVMe SSD | ~80 GB free SSD |
| **GPU / Acceleration** | CPU fallback supported | Metal (Apple Silicon) or CUDA (NVIDIA) | Apple Metal Performance Shaders |

---

## 2. Ollama & Local Model Setup

The system utilizes [Ollama](https://ollama.com/) as its offline inference engine.

### Step 2.1: Install Ollama
- **macOS**: Download from [ollama.com/download/mac](https://ollama.com/download/mac) or install via Homebrew:
  ```bash
  brew install --cask ollama
  ```
- **Linux**:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
- **Windows**: Download installer from [ollama.com/download/windows](https://ollama.com/download/windows).

### Step 2.2: Start the Ollama Daemon
Ensure Ollama is running and listening on port `11434`:
```bash
ollama serve
```

### Step 2.3: Pull Required Local Models
Pull the primary reasoning LLM, embedding model, and optional high-capacity LLM:
```bash
# Primary Reasoning & Extraction LLM (3.4 GB)
ollama pull gemma3:4b

# Dense 768-dimensional Vector Embeddings (621 MB)
ollama pull embeddinggemma:latest

# High-Capacity Alternative LLM (Optional, 4.7 GB)
ollama pull qwen2.5:7b
```

Verify the downloaded models:
```bash
ollama list
```
Expected output:
```
NAME                     ID              SIZE      MODIFIED
gemma3:4b                68ceab7e468a    3.4 GB    ...
embeddinggemma:latest    85462619ee72    621 MB    ...
qwen2.5:7b               845dbda0ea48    4.7 GB    ...
```

---

## 3. Python Environment & Package Installation

The project requires Python **3.10 through 3.13** (tested extensively on Python 3.13.7).

### Step 3.1: Clone the Repository
```bash
git clone https://github.com/shikhar-srivastava/hybrid-graphrag-system.git
cd hybrid-graphrag-system
```

### Step 3.2: Create and Activate a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3.3: Install the Package in Editable Mode
Install `hybrid-graphrag` and its dependencies:
```bash
pip install -e .
```

To install development and testing dependencies:
```bash
pip install -e ".[dev]"
```

---

## 4. Verification & Testing

Verify that the installation is complete and healthy.

### Step 4.1: Run the Automated Test Suite
Execute pytest across the comprehensive test suite:
```bash
pytest
```
All unit, integration, and regression tests should pass:
```
============================== 101 passed in 4.5s ===============================
```

### Step 4.2: Execute the End-to-End Quickstart
Run the self-contained quickstart script:
```bash
python examples/quickstart.py
```
This script initializes an isolated knowledge base, indexes sample multi-domain documents, performs a hybrid query, and displays structured citations and answers.

---

## 5. Troubleshooting & FAQ

### Issue: `Connection refused on localhost:11434`
- **Cause**: The Ollama service is not running.
- **Resolution**: Run `ollama serve` in a separate terminal window, or launch the Ollama desktop app.

### Issue: `OutOfMemoryError` or High Latency on Apple Silicon
- **Cause**: Running multiple large models concurrently or low unified memory headroom.
- **Resolution**:
  - Use `gemma3:4b` (3.4 GB footprint) instead of larger 7B models.
  - Set `OLLAMA_NUM_PARALLEL=1` to prevent concurrent model allocations.

### Issue: `igraph` C Compilation Failure on Linux
- **Cause**: Missing C compilation toolchain or development headers.
- **Resolution**:
  ```bash
  sudo apt-get update && sudo apt-get install -y build-essential python3-dev
  ```

### Issue: ChromaDB Database Lock or Permission Error
- **Cause**: Multiple processes writing to the same knowledge base directory concurrently.
- **Resolution**: Ensure each knowledge base has a unique `kb_id` and is accessed by one writer at a time.
