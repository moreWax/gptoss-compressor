#!/usr/bin/env bash
set -euo pipefail

# Verify runtime and run microbenchmark
python compress_gptoss.py verify-runtime --bench --M 16 --K 4096 --N 4096 --iters 50

# Dequantize with memory optimization (sequential device mapping + disk offloading)
# This uses sequential offloading to minimize RAM usage (~32-64GB instead of 500GB+)
python compress_gptoss.py dequantize \
  --src openai/gpt-oss-120b \
  --dst gpt-oss-120b-fp16 \
  --dtype fp16 \
  --device-map sequential \
  --offload-folder ./offload_cache

# Quantize with memory optimization
python compress_gptoss.py quantize \
  --model-in gpt-oss-120b-fp16 \
  --experts-algo gptq --experts-type int4 --experts-scheme w4a8 \
  --mw-algo gptq --mw-type int4 --mw-scheme w4a16 \
  --auto-detect-experts \
  --try-install-qqq --yes \
  --dataset open_platypus --num-calibration-samples 256 --max-seq-length 2048 \
  --group-size 128 --block-size 128 --dampening-frac 0.01 --offload-hessians \
  --device-map sequential \
  --offload-folder ./offload_cache
