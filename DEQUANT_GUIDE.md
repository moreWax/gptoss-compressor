# One-Click Dequantization Guide

## 🎯 Complete One-Click Solution

The `dequantize-to-local` service is now a **complete one-click solution** that handles everything automatically:

### ✅ What it does automatically

1. **Creates output directory** if it doesn't exist
2. **Runs environment verification** (GPU, CUDA, QQQ kernels)
3. **Checks disk space** and shows available storage
4. **Downloads GPT-OSS-120B** (~240GB) to HuggingFace cache
5. **Dequantizes MXFP4 → FP16** with memory optimization
6. **Saves to local directory** outside the container
7. **Shows final results** and directory size

## 🚀 Usage

### Default (saves to ./fp16/)

```bash
docker-compose run --rm dequantize-to-local
```

### Custom output directory

```bash
OUTPUT_DIR=/path/to/your/models docker-compose run --rm dequantize-to-local

# Examples:
OUTPUT_DIR=/mnt/external/models docker-compose run --rm dequantize-to-local
OUTPUT_DIR=$HOME/ai-models docker-compose run --rm dequantize-to-local
```

### Quick version (skips verification)

```bash
docker-compose run --rm dequant-quick
```

## 📁 Output Structure

After completion, you'll have:

docker-compose run --rm dequantize-to-local
```
./fp16/                                    # (or your custom OUTPUT_DIR)
└── gpt-oss-120b-fp16/
    ├── config.json
    ├── generation_config.json
    ├── model-00001-of-00048.safetensors  # ~240GB total
    ├── model-00002-of-00048.safetensors
    ├── ...
    └── tokenizer files
```

## 💾 Requirements

- **Disk space**: 100GB+ recommended with sequential device mapping (vs 500GB+ with auto mapping)
- **GPU**: NVIDIA with CUDA support
- **RAM**: 32-64GB (with memory optimization)
- **Time**: 30-60 minutes depending on GPU and storage speed

## 🛠️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OUTPUT_DIR` | `./fp16` | Local directory to save the model |
| `SKIP_STARTUP_COMMANDS` | `false` | Skip verification for faster startup |
| `AUTO_VERIFY` | `true` | Run runtime verification |
| `AUTO_TEST_ENV` | `true` | Test environment and GPU access |

## 🔧 Advanced Usage

### Monitor progress in another terminal

```bash
# While dequantization is running, monitor in another terminal:
watch -n 5 'du -sh ./fp16/ && df -h .'
```

### Custom GPU selection

```bash
NVIDIA_VISIBLE_DEVICES=0,1 docker-compose run --rm dequantize-to-local
```

### Different data types

```bash
# For BF16 instead of FP16, modify the service or run manually:
docker-compose run --rm gptoss-compressor python compress_gptoss.py dequantize \
  --src openai/gpt-oss-120b \
  --dst /workspace/app/output/gpt-oss-120b-bf16 \
  --dtype bf16 \
  --device-map sequential \
  --offload-folder ./offload_cache
```

## ⚡ Summary

**One command does everything:**

```bash
docker-compose run --rm dequantize-to-local
```

That's it! The service handles all the complexity and gives you a ready-to-use FP16 model in your local directory.
