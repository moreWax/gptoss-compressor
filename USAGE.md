# GPT-OSS 120B Compression CLI — W4A8, Kernel Install, Mixed Algorithms

This CLI supports **W4A16**, **W8A8**, and **W4A8** (experimental, GPTQ-only), optional **2:4 sparsity**,
per-group algos (**AWQ/GPTQ**) and per-group type/scheme (**experts** vs **other weights**),
plus a **kernel installer** and a **runtime verifier** with a quick microbench.

## ⚡ Memory Optimization Features (NEW)

The tool now includes **sequential offloading** to minimize RAM usage, making it usable on systems with limited memory:

### Device Mapping Options

- `--device-map sequential` (default): Sequential loading through CPU, minimizes memory (~32-64GB)
- `--device-map auto`: Automatically distributes model across available GPUs and CPU
- `--device-map cpu-only`: Forces CPU-only execution (slowest, minimal GPU memory)
- `--device-map sequential`: Places layers sequentially across devices to minimize peak memory
- `--device-map cpu-only`: Forces CPU-only execution (requires large RAM)

### Disk Offloading

- `--offload-folder ./cache`: Temporarily stores weights on disk when needed
- Significantly reduces peak memory usage from ~300GB to as low as 32-64GB
- Essential for consumer hardware

**Memory Benefits**: Before optimization required ~300GB+ RAM. After optimization can work with 32-64GB RAM using proper device mapping and disk offloading.

## Install base deps

```bash
pip install -U pip
pip install "llmcompressor==0.6.0.1" "transformers>=4.44" "torch" "accelerate" "safetensors"
```

## Install W4A8 kernels (optional but required for W4A8)

```bash
python compress_gptoss.py install-kernel --yes
```

## Verify runtime & microbenchmark

```bash
python compress_gptoss.py verify-runtime --bench   --M 16 --K 4096 --N 4096 --iters 50
```

## De-MXFP4 → FP16/BF16

```bash
python compress_gptoss.py dequantize   --src openai/gpt-oss-120b   --dst gpt-oss-120b-fp16   --dtype fp16
```

## Quantization — Key Rules

- AWQ: weight-only INT4 → use `int4 + w4a16` only.
- GPTQ: supports `W4A16`, `W8A8`, and `W4A8` (W4A8 requires kernels).
- If you request `W4A8`:
  - CLI checks for QQQ (`QQQ` or `qqq` pkg). With `--try-install-qqq`, it can install.
  - If install fails and `--allow-fallback` is set, it falls back to `W4A16`.
  - Otherwise, it errors.

### Per-group knobs

- Algorithms: `--experts-algo {gptq,awq}` and `--mw-algo {gptq,awq}`  
- Type/Scheme: `--experts-type {int4,int8}` + `--experts-scheme {w4a16|w4a8|w8a8}`  
               `--mw-type {int4,int8}`      + `--mw-scheme {w4a16|w4a8|w8a8}`  
- Shortcuts: `--experts int4|w4a16|w4a8|int8|w8a8`, `--mw ...`
- Expert selection: `--auto-detect-experts` or `--experts-regex 're:...your path...'`

### Helpful GPTQ flags

`--group-size 128 --block-size 128 --dampening-frac 0.01 --offload-hessians`

## Recipes

### W4A8 experts (GPTQ) + W4A16 MW (GPTQ), auto-install if missing

```bash
python compress_gptoss.py quantize   --model-in gpt-oss-120b-fp16   --experts-algo gptq --experts-type int4 --experts-scheme w4a8   --mw-algo gptq      --mw-type int4    --mw-scheme w4a16   --auto-detect-experts   --try-install-qqq --yes   --dataset open_platypus --num-calibration-samples 256 --max-seq-length 2048   --group-size 128 --block-size 128 --dampening-frac 0.01 --offload-hessians
```

### W4A8 everywhere with fallback to W4A16

```bash
python compress_gptoss.py quantize   --model-in gpt-oss-120b-fp16   --experts-algo gptq --experts-type int4 --experts-scheme w4a8   --mw-algo gptq      --mw-type int4    --mw-scheme w4a8   --allow-fallback   --dataset open_platypus --num-calibration-samples 256 --max-seq-length 2048   --group-size 128 --block-size 128 --dampening-frac 0.01 --offload-hessians
```

### Mixed algorithms: AWQ for experts (INT4), GPTQ W8A8 for MW

```bash
python compress_gptoss.py quantize   --model-in gpt-oss-120b-fp16   --experts-algo awq --experts-type int4 --experts-scheme w4a16   --mw-algo gptq     --mw-type int8 --mw-scheme w8a8   --auto-detect-experts   --dataset open_platypus --num-calibration-samples 256 --max-seq-length 2048   --group-size 128 --block-size 128 --dampening-frac 0.01 --offload-hessians
```

### 2:4 sparsity + W4A8 experts (GPTQ) + W4A16 MW (GPTQ)

```bash
python compress_gptoss.py quantize   --model-in gpt-oss-120b-fp16   --with-sparse --sparsity 0.5 --mask 2:4   --experts-algo gptq --experts-type int4 --experts-scheme w4a8   --mw-algo gptq      --mw-type int4    --mw-scheme w4a16   --auto-detect-experts   --try-install-qqq   --dataset open_platypus --num-calibration-samples 256 --max-seq-length 2048   --group-size 128 --block-size 128 --dampening-frac 0.01 --offload-hessians
```

## Output naming & dry runs

```bash
python compress_gptoss.py quantize --dry-run   --experts-algo awq --experts-type int4 --experts-scheme w4a16   --mw-algo gptq   --mw-type int8 --mw-scheme w8a8
```
