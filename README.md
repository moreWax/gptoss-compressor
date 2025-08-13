# gptoss-compressor

A batteries-included CLI to **dequantize GPT‑OSS MXFP4 → FP16/BF16** and **compress** large LLMs with
**GPTQ/AWQ**, supporting **W4A16**, **W8A8**, and **W4A8** (experimental, GPTQ‑only), with optional **2:4 sparsity**.
It also ships a **CUTLASS kernel installer** (QQQ) and a **runtime verifier + microbench**.

> If you’re in a hurry, jump to **[Quickstart](#quickstart)** and **[Recipes](#recipes)**.

---

## Features

- **Dequantize**: MXFP4 → BF16/FP16 one‑time conversion for `openai/gpt-oss-120b` (works for other models too).
- **Quantize** (via [LLM Compressor]):  
  - **GPTQ**: `W4A16`, `W8A8`, **`W4A8`** (needs W4A8 kernels).  
  - **AWQ**: weight‑only **INT4** (`W4A16`).
- **Per‑group controls**: separate algo and scheme for **experts** vs **other model weights**.
- **Optional 2:4 sparsity** (SparseGPT) prior to quantization.
- **W4A8 runtime integration**: detect/install **QQQ** CUTLASS kernels automatically.
- **verify-runtime**: prints GPU/CC, CUDA status, detects QQQ, and can run a tiny **FP16 vs W4A8** microbench.

---

## Requirements

- **Python** ≥ 3.9
- **NVIDIA GPU** with compute capability **≥ 8.0** (Ampere+). RTX **3090 (CC 8.6)** is supported.
- **PyTorch** with CUDA that matches your system drivers.
- Build tools for CUDA extensions (for W4A8 kernels): `nvcc`, C++ compiler, headers.  
  If building on 3090, you may set:

  ```bash
  export TORCH_CUDA_ARCH_LIST="8.6"
  ```

> **Note:** `W4A8` requires CUTLASS kernels. This project can install **QQQ** automatically.

---

## Install

Using the provided requirements (runtime only; toolchain installed separately):

```bash
pip install -U pip
pip install -r requirements.txt
```

Install the **W4A8 kernels** (optional unless you plan to use W4A8):

```bash
python compress_gptoss.py install-kernel --yes
```

Verify environment and try a tiny microbench:

```bash
python compress_gptoss.py verify-runtime --bench --M 16 --K 4096 --N 4096 --iters 50
```

## Resource Requirements

**Default (Sequential Device Mapping)**:

- **RAM**: 32-64GB recommended
- **Disk**: 100GB+ free space
- **GPU**: Optional, any NVIDIA GPU with 8GB+ VRAM helps

**Alternative (Auto Device Mapping)**:

- **RAM**: 200GB+ recommended
- **Disk**: 500GB+ free space
- **GPU**: Multi-GPU setup preferred

💡 **The default sequential device mapping dramatically reduces resource requirements** by loading model layers sequentially through CPU/disk rather than keeping everything in memory simultaneously.

---

## Quickstart

1. **Dequantize** MXFP4 → FP16 (with memory optimization):

```bash
python compress_gptoss.py dequantize \
  --src openai/gpt-oss-120b \
  --dst gpt-oss-120b-fp16 \
  --dtype fp16 \
  --device-map sequential \
  --offload-folder ./offload_cache
```

2. **Quantize** (example: W4A8 experts with fallback + auto kernel install + memory optimization):

```bash
python compress_gptoss.py quantize \
  --model-in gpt-oss-120b-fp16 \
  --experts-algo gptq --experts-type int4 --experts-scheme w4a8 \
  --mw-algo gptq --mw-type int4 --mw-scheme w4a16 \
  --auto-detect-experts \
  --try-install-qqq --yes \
  --allow-fallback \
  --dataset open_platypus --num-calibration-samples 256 --max-seq-length 2048 \
  --group-size 128 --block-size 128 --dampening-frac 0.01 --offload-hessians \
  --device-map sequential \
  --offload-folder ./offload_cache
```

---

## CLI Overview

```

python compress_gptoss.py --help
python compress_gptoss.py dequantize --help
python compress_gptoss.py quantize --help
python compress_gptoss.py install-kernel --help
python compress_gptoss.py verify-runtime --help
```

### Subcommands

- **dequantize** — Upcast MXFP4 to BF16/FP16 and save a local checkpoint.  
  Flags: `--src`, `--dst`, `--dtype {fp16,bf16}`, `--device-map {auto,sequential,cpu-only}` (default: sequential), `--offload-folder`

- **quantize** — Compress with GPTQ/AWQ (and optional 2:4 sparsity).  
  Common flags:
  - Default algo: `--algo {gptq,awq}`
  - Per‑group algos: `--experts-algo {gptq,awq}` `--mw-algo {gptq,awq}`
  - Per‑group type/scheme:
    - `--experts-type {int4,int8}` + `--experts-scheme {w4a16|w4a8|w8a8}`
    - `--mw-type {int4,int8}`      + `--mw-scheme {w4a16|w4a8|w8a8}`
  - (Shortcuts) `--experts int4|w4a16|w4a8|int8|w8a8`, `--mw ...`
  - Expert selection: `--auto-detect-experts` or `--experts-regex 're:...'`
  - 2:4 sparsity: `--with-sparse --sparsity 0.5 --mask 2:4`
  - GPTQ knobs: `--group-size 128 --block-size 128 --dampening-frac 0.01 --offload-hessians`
  - Calibration: `--dataset open_platypus` or `--calib-jsonl /path/texts.jsonl`
  - Memory optimization: `--device-map {auto,sequential,cpu-only}` (default: sequential), `--offload-folder`
  - Housekeeping: `--output-dir ...` `--extra-ignore PATTERN` `--dry-run`

  **W4A8 control** (GPTQ‑only):
  - If you request `w4a8`, the CLI checks for QQQ; add `--try-install-qqq` to install automatically (use `--yes` to skip the prompt).
  - Add `--allow-fallback` to downgrade `w4a8 → w4a16` when kernels aren’t available.

- **install-kernel** — Installs **QQQ** (CUTLASS W4A8 kernels) from GitHub.  
  Flags: `--yes`

- **verify-runtime** — Prints GPU/CC, CUDA status, QQQ detection, and optional microbench.  
  Flags: `--bench --M 16 --K 4096 --N 4096 --iters 50`

---

## Recipes

**W4A16 everywhere (AWQ)** — weight‑only INT4:

```bash
python compress_gptoss.py quantize   --model-in gpt-oss-120b-fp16   --experts-algo awq --experts-type int4 --experts-scheme w4a16   --mw-algo awq      --mw-type int4    --mw-scheme w4a16   --dataset open_platypus --num-calibration-samples 256 --max-seq-length 2048   --group-size 128 --symmetric
```

**W8A8 (GPTQ)** — INT8 weights & activations:

```bash
python compress_gptoss.py quantize   --model-in gpt-oss-120b-fp16   --mw-algo gptq --mw-type int8 --mw-scheme w8a8   --experts-algo gptq --experts-type int8 --experts-scheme w8a8   --dataset open_platypus --num-calibration-samples 256 --max-seq-length 2048
```

**W4A8 experts (GPTQ) + W4A16 rest (GPTQ)**:

```bash
python compress_gptoss.py quantize   --model-in gpt-oss-120b-fp16   --experts-algo gptq --experts-type int4 --experts-scheme w4a8   --mw-algo gptq      --mw-type int4    --mw-scheme w4a16   --auto-detect-experts   --try-install-qqq --yes --allow-fallback   --dataset open_platypus --num-calibration-samples 256 --max-seq-length 2048   --group-size 128 --block-size 128 --dampening-frac 0.01 --offload-hessians
```

**2:4 sparsity + W4A8 experts (GPTQ) + W4A16 rest (GPTQ)**:

```bash
python compress_gptoss.py quantize   --model-in gpt-oss-120b-fp16   --with-sparse --sparsity 0.5 --mask 2:4   --experts-algo gptq --experts-type int4 --experts-scheme w4a8   --mw-algo gptq      --mw-type int4    --mw-scheme w4a16   --auto-detect-experts --try-install-qqq   --dataset open_platypus --num-calibration-samples 256 --max-seq-length 2048   --group-size 128 --block-size 128 --dampening-frac 0.01 --offload-hessians
```

---

## Output & Defaults

- Unless `--output-dir` is set, outputs are named like:  
  `gpt-oss-120b[-2of4][EXPERTALGO-MWALGO]-experts-<spec>-mw-<spec>`
- We ignore by default: `embed_tokens`, `mlp.router`, `lm_head`. Add your own with `--extra-ignore` (repeatable).

---

## Microbench Notes

`verify-runtime --bench` uses synthetic tensors to compare **FP16 `nn.Linear`** vs **QQQ W4A8 `QuantLinear`**.  
This is a **smoke test**, not end‑to‑end throughput. Real serving speed depends on batching, seq lengths, attention backend, and memory bandwidth.

---

## Troubleshooting

- **“W4A8 requested but runtime not available”**  
  Use `--try-install-qqq` (add `--yes` for non‑interactive) or pass `--allow-fallback` to switch to `W4A16`.
- **“ImportError: No module named qqq/QQQ”**  
  Run `python compress_gptoss.py install-kernel --yes`.
- **CUDA arch mismatch / slow kernels**  
  Rebuild with `export TORCH_CUDA_ARCH_LIST="8.6"` for RTX 3090 before installing QQQ.
- **OOM during GPTQ calibration**  
  Reduce `--num-calibration-samples`, lower `--max-seq-length`, and enable `--offload-hessians`.
- **Quality regressions**  
  Try larger `--group-size`, keep sequential updates on (don’t pass `--no-sequential-update`), or use more/better calibration data via `--calib-jsonl`.

---

## Support Matrix (quick reference)

| Algo | Scheme | Supported | Notes |
|------|--------|-----------|-------|
| AWQ  | W4A16  | ✅        | Weight‑only INT4 |
| AWQ  | W8A8   | ❌        | Not supported |
| AWQ  | W4A8   | ❌        | Not supported |
| GPTQ | W4A16  | ✅        | Stable |
| GPTQ | W8A8   | ✅        | Stable (INT8 weights + activations) |
| GPTQ | W4A8   | ⚠️        | Needs CUTLASS kernels (QQQ); experimental |

---

## Acknowledgements

- [LLM Compressor] for quantization/sparsity primitives.  
- [Transformers] for model loading.  
- **QQQ** by HandH1998 et al. for W4A8 CUTLASS kernels.  
- PyTorch / CUDA.

---

## License

MIT — see [LICENSE](LICENSE).

[LLM Compressor]: https://github.com/neuralmagic/llm-compressor
[Transformers]: https://github.com/huggingface/transformers

---

## Docker (CUDA + PyTorch + build tools)

A ready-to-use **Dockerfile** is included for building/running **W4A8** in a clean environment on an RTX **3090 (CC 8.6)**.

**Base image:** `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` (includes **nvcc** for compiling CUDA extensions).  
**Build args:**

- `CUDA_ARCH` — defaults to `8.6` for RTX 3090, override if needed.
- `INSTALL_QQQ` — set `true` to compile/install **QQQ CUTLASS kernels** at build-time; set `false` to install later.

**Note:** When `INSTALL_QQQ=true`, the Docker build will download and compile QQQ CUTLASS kernels from [HandH1998/QQQ](https://github.com/HandH1998/QQQ) during the build process. This may take several minutes and requires sufficient build resources.

### Build

```bash
cd gptoss-compressor

# Build with QQQ CUTLASS kernels (recommended for W4A8 support)
docker build -t gptoss-compressor:cu128   --build-arg CUDA_ARCH=8.6   --build-arg INSTALL_QQQ=true   .

# Build without QQQ (faster build, install kernels later)
docker build -t gptoss-compressor:cu128   --build-arg CUDA_ARCH=8.6   --build-arg INSTALL_QQQ=false   .
```

**Or use docker-compose:**

```bash
# Default build (includes QQQ)
docker-compose build

# Build without QQQ (faster)
INSTALL_QQQ=false docker-compose build
```

**Test the build:**

```bash
# Comprehensive build and functionality test
make build-test

# Or manually
./build_test.sh
```

### Run (GPU required)

Make sure the **NVIDIA Container Toolkit** is installed and pass `--gpus all`.

**The container now automatically runs verification commands on startup:**

```bash
# Interactive shell with automatic verification
docker run --rm --gpus all -it gptoss-compressor:cu128

# Skip automatic verification and go straight to shell
docker run --rm --gpus all -it -e SKIP_STARTUP_COMMANDS=true gptoss-compressor:cu128 bash

# Run specific command with verification first
docker run --rm --gpus all -v "$PWD:/workspace/app" gptoss-compressor:cu128 make verify
```

**Control startup behavior with environment variables:**

```bash
# Enable microbench in verification
docker run --rm --gpus all -e RUN_MICROBENCH=true gptoss-compressor:cu128

# Skip environment testing
docker run --rm --gpus all -e AUTO_TEST_ENV=false gptoss-compressor:cu128

# Install kernels if missing
docker run --rm --gpus all -e AUTO_INSTALL_KERNELS=true gptoss-compressor:cu128
```

**Or with docker-compose (recommended):**

```bash
# Interactive shell with automatic verification
docker-compose run gptoss-compressor

# Run verification with microbench
docker-compose run verify

# Run dequantization (with verification first)
docker-compose run dequantize

# Run quantization with W4A8 (with verification first)
docker-compose run quantize-w4a8

# Run custom make targets
docker-compose run make w4a16_awq_all
docker-compose run make sparse_w4a8exp_w4a16mw

# Debug mode (skip verification)
docker-compose run debug
```

**Environment variable control:**

```bash
# Enable microbench for all services
RUN_MICROBENCH=true docker-compose run gptoss-compressor

# Skip verification for faster startup
SKIP_STARTUP_COMMANDS=true docker-compose run gptoss-compressor

# Install missing kernels at runtime
AUTO_INSTALL_KERNELS=true docker-compose run gptoss-compressor
```

### GPU Configuration

There are several ways to expose GPUs to the container:

#### 1. Docker Run with All GPUs

```bash
# Expose all GPUs
docker run --rm --gpus all -it -v "$(pwd):/workspace/app" gptoss-compressor:cu128

# Test GPU access
docker run --rm --gpus all gptoss-compressor:cu128 python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

#### 2. Docker Run with Specific GPUs

```bash
# Specific GPU devices (0 and 1)
docker run --rm --gpus '"device=0,1"' -it -v "$(pwd):/workspace/app" gptoss-compressor:cu128

# Single GPU (device 0)
docker run --rm --gpus '"device=0"' -it -v "$(pwd):/workspace/app" gptoss-compressor:cu128
```

#### 3. Environment Variable Control

```bash
# All GPUs
NVIDIA_VISIBLE_DEVICES=all docker run --rm --gpus all -it gptoss-compressor:cu128

# Specific GPUs
NVIDIA_VISIBLE_DEVICES=0,1 docker run --rm --gpus all -it gptoss-compressor:cu128

# Single GPU
NVIDIA_VISIBLE_DEVICES=0 docker run --rm --gpus all -it gptoss-compressor:cu128
```

#### 4. Docker Compose

The docker-compose.yml file is configured to use all GPUs by default:

```bash
# All GPUs (default)
docker-compose run --rm gpu-app

# Override with specific GPUs
NVIDIA_VISIBLE_DEVICES=0,1 docker-compose run --rm gpu-app

# Single GPU
NVIDIA_VISIBLE_DEVICES=0 docker-compose run --rm gpu-app
```

#### 5. Makefile Targets

```bash
# Test GPU access
make test-gpu-access              # All GPUs
make test-specific-gpu-access     # Specific GPUs (0,1,2,3,4)

# Interactive shells
make run-shell-all-gpus          # Shell with all GPUs
make run-shell-specific-gpus     # Shell with specific GPUs

# Get help
make gpu-help                    # GPU configuration examples
```

**Prerequisites:**

- NVIDIA Docker runtime installed (`nvidia-docker2` or `nvidia-container-toolkit`)
- NVIDIA drivers compatible with CUDA 12.8
- GPU compute capability 7.0+ (for W4A8 kernels, 8.6+ recommended)

### Inside the container

**Automatic startup commands run by default:**

- `make test-env` - Test Python environment and GPU access
- `python compress_gptoss.py verify-runtime` - Verify CUDA and QQQ kernels  

```bash
# Available commands after startup:
make verify          # Full verification + microbench
make dequant         # Dequantize GPT-OSS
make w4a8exp_w4a16mw # Quantize with W4A8 experts
make help            # See all targets

# Direct commands:
python compress_gptoss.py dequantize --src openai/gpt-oss-120b --dst gpt-oss-120b-fp16 --dtype fp16

# Control startup behavior:
export SKIP_STARTUP_COMMANDS=true     # Skip all automatic commands
export RUN_MICROBENCH=true           # Include microbench in verification
export AUTO_INSTALL_KERNELS=true     # Install QQQ if missing
export AUTO_TEST_ENV=false           # Skip environment testing
export AUTO_VERIFY=false             # Skip runtime verification
```

#### Startup Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTO_VERIFY` | `true` | Run runtime verification on startup |
| `AUTO_TEST_ENV` | `true` | Test environment and GPU access |
| `AUTO_INSTALL_KERNELS` | `false` | Install QQQ kernels if missing |
| `RUN_MICROBENCH` | `false` | Include microbench in verification |
| `SKIP_STARTUP_COMMANDS` | `false` | Skip all automatic startup commands |

### Persistent Model Storage

Your models and cache persist outside the container through volume mounts:

#### Default Volume Mounts (docker-compose)

```bash
# These directories are automatically mounted and persist:
./models                     # → /workspace/app/models (your quantized models)
./offload_cache              # → /workspace/app/offload_cache (memory offload cache)
~/.cache/huggingface         # → /root/.cache/huggingface (HuggingFace model cache)
```

#### Quick Setup Script

Use the provided script to set up persistent storage:

```bash
# Default location (~100GB recommended with sequential device mapping)
# Use 500GB+ for auto device mapping
./setup_storage.sh

# Custom location
./setup_storage.sh /path/to/your/storage

# Example: external drive
./setup_storage.sh /mnt/external/gptoss-models
```

The script creates:

- Persistent storage directories
- Symlinks for easy access
- Environment configuration for docker-compose

#### Custom Storage Locations

You can override the default locations:

**Using docker-compose with custom paths:**

```bash
# Create your custom directories
mkdir -p /path/to/your/models
mkdir -p /path/to/your/cache

# Override with environment variables
MODELS_DIR=/path/to/your/models CACHE_DIR=/path/to/your/cache docker-compose run dequantize
```

**Using docker run with custom mounts:**

```bash
# Custom model storage location
docker run --rm --gpus all \
  -v "/path/to/your/models:/workspace/app/models" \
  -v "/path/to/your/cache:/workspace/app/offload_cache" \
  -v "~/.cache/huggingface:/root/.cache/huggingface" \
  -v "$(pwd):/workspace/app" \
  gptoss-compressor:cu128 \
  python compress_gptoss.py dequantize --src openai/gpt-oss-120b --dst gpt-oss-120b-fp16
```

#### Storage Requirements

- **Source models**: ~240GB for GPT-OSS-120B MXFP4
- **Dequantized models**: ~240GB for FP16 output  
- **Quantized models**: 30-120GB depending on scheme
- **Cache/offload**: 32-64GB during processing
- **HuggingFace cache**: ~240GB for downloaded models

**Recommended**: Use a fast SSD with at least 100GB free space (sequential device mapping) or 500GB+ (auto device mapping).

---

## How `config.json` is Handled

The script ensures that the output model preserves the exact configuration of the source model, which is critical for custom architectures like GPT-OSS. Here's how:

1. **Initial Load**: `transformers` reads `config.json` from the source model when `from_pretrained` is called. `trust_remote_code=True` is used to allow the custom GPT-OSS model code to be loaded.
2. **Error Correction**: If a `rope_scaling` error occurs, the script pre-loads the config, corrects the setting in memory, and retries.
3. **Final Overwrite**: After saving the dequantized model, the script **re-reads the original `config.json` from the source** and **overwrites the `config.json` in the destination directory**. This guarantees that all custom fields and the correct `model_type` ("gpt_oss") are preserved, preventing the model from being misidentified as a different architecture (e.g., Llama).

---

## One-liners (Makefile recipes)

Use **Makefile** targets for popular recipes. Run locally or inside Docker with `DOCKER=1`.

```bash
# Build the image (includes QQQ kernels by default)
make build-image

# Interactive shell in the container
make bash

# Verify runtime (with microbench)
make verify

# Install kernels (if you didn't build them in)
make install-kernels

# Dequantize GPT-OSS (MXFP4 -> FP16)
make dequant

# AWQ W4A16 everywhere
make w4a16_awq_all

# GPTQ W8A8 everywhere
make w8a8_gptq_all

# GPTQ W4A8 experts + W4A16 rest
make w4a8exp_w4a16mw

# 2:4 sparsity + GPTQ W4A8 experts + W4A16 rest
make sparse_w4a8exp_w4a16mw

# Mixed: AWQ (experts) + GPTQ W8A8 (rest)
make mixed_awqexp_gptq_w8a8mw
```

Run the same targets **inside Docker**:

```bash
make verify DOCKER=1
make w4a8exp_w4a16mw DOCKER=1
```
