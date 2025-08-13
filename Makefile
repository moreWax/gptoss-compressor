# Makefile for gptoss-compressor
# Supports both local and Docker execution
# Usage: make <target> [DOCKER=1]

# Configuration
DOCKER_IMAGE = gptoss-compressor:cu128
DOCKER_RUN = docker run --rm --gpus all -it -v "$(PWD):/workspace/app" $(DOCKER_IMAGE)
DOCKER_RUN_GPU = docker run --rm --gpus all -it -v "$(PWD):/workspace/app" $(DOCKER_IMAGE)
DOCKER_RUN_SPECIFIC_GPU = docker run --rm --gpus '"device=0,1,2,3,4"' -it -v "$(PWD):/workspace/app" $(DOCKER_IMAGE)
DOCKER_COMPOSE = docker-compose

# Python command (local or Docker)
ifeq ($(DOCKER),1)
    PYTHON_CMD = $(DOCKER_RUN) python
    BASH_CMD = $(DOCKER_RUN) bash
else
    PYTHON_CMD = python
    BASH_CMD = bash
endif

# Default target
.PHONY: help
help:
	@echo "gptoss-compressor Makefile"
	@echo ""
	@echo "Docker targets:"
	@echo "  build-image           Build Docker image with QQQ kernels"
	@echo "  build-test            Run comprehensive build and functionality test"
	@echo "  test-entrypoint       Test automatic entrypoint behavior"
	@echo "  bash                  Interactive shell in container"
	@echo "  compose-up            Start services with docker-compose"
	@echo "  compose-down          Stop docker-compose services"
	@echo ""
	@echo "Verification targets:"
	@echo "  verify                Verify runtime and run microbench"
	@echo "  install-kernels       Install QQQ CUTLASS kernels"
	@echo "  test-env              Test container environment"
	@echo "  test-qqq              Test QQQ CUTLASS kernels specifically"
	@echo ""
	@echo "Processing targets:"
	@echo "  dequant               Dequantize GPT-OSS (MXFP4 -> FP16)"
	@echo "  dequant-to-fp16       Dequantize to ./fp16/ folder"
	@echo "  w4a16_awq_all         AWQ W4A16 everywhere"
	@echo "  w8a8_gptq_all         GPTQ W8A8 everywhere"
	@echo "  w4a8exp_w4a16mw       GPTQ W4A8 experts + W4A16 rest"
	@echo "  sparse_w4a8exp_w4a16mw  2:4 sparsity + GPTQ W4A8 experts + W4A16 rest"
	@echo "  mixed_awqexp_gptq_w8a8mw  Mixed: AWQ (experts) + GPTQ W8A8 (rest)"
	@echo ""
	@echo "Add DOCKER=1 to run targets in Docker container"
	@echo "Example: make verify DOCKER=1"

# Docker targets
.PHONY: build-image
build-image:
	docker build -t $(DOCKER_IMAGE) --build-arg CUDA_ARCH=8.6 --build-arg INSTALL_QQQ=true .

.PHONY: build-test
build-test:
	./build_test.sh

.PHONY: test-entrypoint
test-entrypoint:
	./test_entrypoint.sh

.PHONY: bash
bash:
ifeq ($(DOCKER),1)
	$(BASH_CMD)
else
	@echo "Use 'make bash DOCKER=1' to run bash in container"
	@exit 1
endif

.PHONY: compose-up
compose-up:
	$(DOCKER_COMPOSE) up -d

.PHONY: compose-down
compose-down:
	$(DOCKER_COMPOSE) down

# Verification targets
.PHONY: verify
verify:
	$(PYTHON_CMD) compress_gptoss.py verify-runtime --bench --M 16 --K 4096 --N 4096 --iters 50

.PHONY: install-kernels
install-kernels:
	$(PYTHON_CMD) compress_gptoss.py install-kernel --yes

# Core processing targets
.PHONY: dequant
dequant:
	$(PYTHON_CMD) compress_gptoss.py dequantize \
		--src openai/gpt-oss-120b \
		--dst gpt-oss-120b-fp16 \
		--dtype fp16 \
		--device-map sequential \
		--offload-folder ./offload_cache

.PHONY: dequant-to-fp16
dequant-to-fp16:
	@mkdir -p ./fp16
	$(PYTHON_CMD) compress_gptoss.py dequantize \
		--src openai/gpt-oss-120b \
		--dst ./fp16/gpt-oss-120b-fp16 \
		--dtype fp16 \
		--device-map sequential \
		--offload-folder ./offload_cache

# Quantization recipes
.PHONY: w4a16_awq_all
w4a16_awq_all:
	$(PYTHON_CMD) compress_gptoss.py quantize \
		--model-in gpt-oss-120b-fp16 \
		--experts-algo awq --experts-type int4 --experts-scheme w4a16 \
		--mw-algo awq --mw-type int4 --mw-scheme w4a16 \
		--dataset open_platypus --num-calibration-samples 256 --max-seq-length 2048 \
		--group-size 128 --symmetric \
		--device-map sequential --offload-folder ./offload_cache

.PHONY: w8a8_gptq_all
w8a8_gptq_all:
	$(PYTHON_CMD) compress_gptoss.py quantize \
		--model-in gpt-oss-120b-fp16 \
		--mw-algo gptq --mw-type int8 --mw-scheme w8a8 \
		--experts-algo gptq --experts-type int8 --experts-scheme w8a8 \
		--dataset open_platypus --num-calibration-samples 256 --max-seq-length 2048 \
		--device-map sequential --offload-folder ./offload_cache

.PHONY: w4a8exp_w4a16mw
w4a8exp_w4a16mw:
	$(PYTHON_CMD) compress_gptoss.py quantize \
		--model-in gpt-oss-120b-fp16 \
		--experts-algo gptq --experts-type int4 --experts-scheme w4a8 \
		--mw-algo gptq --mw-type int4 --mw-scheme w4a16 \
		--auto-detect-experts \
		--try-install-qqq --yes --allow-fallback \
		--dataset open_platypus --num-calibration-samples 256 --max-seq-length 2048 \
		--group-size 128 --block-size 128 --dampening-frac 0.01 --offload-hessians \
		--device-map sequential --offload-folder ./offload_cache

.PHONY: sparse_w4a8exp_w4a16mw
sparse_w4a8exp_w4a16mw:
	$(PYTHON_CMD) compress_gptoss.py quantize \
		--model-in gpt-oss-120b-fp16 \
		--with-sparse --sparsity 0.5 --mask 2:4 \
		--experts-algo gptq --experts-type int4 --experts-scheme w4a8 \
		--mw-algo gptq --mw-type int4 --mw-scheme w4a16 \
		--auto-detect-experts --try-install-qqq \
		--dataset open_platypus --num-calibration-samples 256 --max-seq-length 2048 \
		--group-size 128 --block-size 128 --dampening-frac 0.01 --offload-hessians \
		--device-map sequential --offload-folder ./offload_cache

.PHONY: mixed_awqexp_gptq_w8a8mw
mixed_awqexp_gptq_w8a8mw:
	$(PYTHON_CMD) compress_gptoss.py quantize \
		--model-in gpt-oss-120b-fp16 \
		--experts-algo awq --experts-type int4 --experts-scheme w4a16 \
		--mw-algo gptq --mw-type int8 --mw-scheme w8a8 \
		--auto-detect-experts \
		--dataset open_platypus --num-calibration-samples 256 --max-seq-length 2048 \
		--group-size 128 --block-size 128 --dampening-frac 0.01 --offload-hessians \
		--device-map sequential --offload-folder ./offload_cache

# Utility targets
.PHONY: clean
clean:
	rm -rf ./offload_cache/*
	rm -rf ./models/*
	@echo "Cleaned cache and model directories"

.PHONY: setup-storage
setup-storage:
	@echo "Setting up persistent storage with default location..."
	./setup_storage.sh

.PHONY: setup-storage-custom
setup-storage-custom:
	@echo "Usage: make setup-storage-custom STORAGE_PATH=/path/to/storage"
	@if [ -z "$(STORAGE_PATH)" ]; then \
		echo "Error: STORAGE_PATH not specified"; \
		echo "Example: make setup-storage-custom STORAGE_PATH=/mnt/external/gptoss"; \
		exit 1; \
	fi
	./setup_storage.sh "$(STORAGE_PATH)"

.PHONY: test-env
test-env:
	@echo "=== Environment Test ==="
	@echo "Working directory: $(PWD)"
	@echo "Python version: $(shell python --version)"
	@echo "Make version: $(shell make --version | head -n1)"
	@echo "CUDA available: $(shell python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not available')"
	@echo "Available GPUs: $(shell python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'N/A')"
	@echo "=== End Test ==="

.PHONY: test-qqq
test-qqq:
	@echo "=== QQQ CUTLASS Kernels Test ==="
	$(PYTHON_CMD) compress_gptoss.py verify-runtime
	@echo "=== End QQQ Test ==="

.PHONY: clean-docker
clean-docker:
	docker system prune -f
	docker volume prune -f

# Help for common docker commands
.PHONY: docker-help
docker-help:
	@echo "Common Docker usage patterns:"
	@echo ""
	@echo "Build and run verification:"
	@echo "  make build-image"
	@echo "  make verify DOCKER=1"
	@echo ""
	@echo "Full pipeline with Docker:"
	@echo "  make build-image"
	@echo "  make dequant DOCKER=1"
	@echo "  make w4a8exp_w4a16mw DOCKER=1"
	@echo ""
	@echo "Interactive development:"
	@echo "  make bash DOCKER=1"
	@echo "  # Inside container, you can use:"
	@echo "  # make test-env"
	@echo "  # make verify"
	@echo "  # make dequant"
	@echo ""
	@echo "Using docker-compose:"
	@echo "  docker-compose run verify"
	@echo "  docker-compose run dequantize"
	@echo "  docker-compose run quantize-w4a8"
	@echo "  # Or get shell: docker-compose run gptoss-compressor"

# GPU-specific Docker targets
.PHONY: run-shell-all-gpus
run-shell-all-gpus:
	@echo "Starting interactive shell with ALL GPUs..."
	$(DOCKER_RUN_GPU) bash

.PHONY: run-shell-specific-gpus
run-shell-specific-gpus:
	@echo "Starting interactive shell with specific GPUs (0,1,2,3,4)..."
	$(DOCKER_RUN_SPECIFIC_GPU) bash

.PHONY: test-gpu-access
test-gpu-access:
	@echo "Testing GPU access in container..."
	$(DOCKER_RUN_GPU) python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

.PHONY: test-specific-gpu-access
test-specific-gpu-access:
	@echo "Testing specific GPU access in container..."
	$(DOCKER_RUN_SPECIFIC_GPU) python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

.PHONY: run-compose-all-gpus
run-compose-all-gpus:
	@echo "Starting container with all GPUs via docker-compose..."
	NVIDIA_VISIBLE_DEVICES=all $(DOCKER_COMPOSE) run --rm gpu-app $(CMD)

.PHONY: run-compose-gpu0
run-compose-gpu0:
	@echo "Starting container with GPU 0 only via docker-compose..."
	NVIDIA_VISIBLE_DEVICES=0 $(DOCKER_COMPOSE) run --rm gpu-app $(CMD)

.PHONY: gpu-help
gpu-help:
	@echo "GPU Configuration Examples:"
	@echo ""
	@echo "Docker run with all GPUs:"
	@echo "  docker run --rm --gpus all -it $(DOCKER_IMAGE)"
	@echo ""
	@echo "Docker run with specific GPUs:"
	@echo "  docker run --rm --gpus '\"device=0,1\"' -it $(DOCKER_IMAGE)"
	@echo "  docker run --rm --gpus '\"device=0\"' -it $(DOCKER_IMAGE)"
	@echo ""
	@echo "Environment variable control:"
	@echo "  NVIDIA_VISIBLE_DEVICES=all docker run ..."
	@echo "  NVIDIA_VISIBLE_DEVICES=0,1 docker run ..."
	@echo "  NVIDIA_VISIBLE_DEVICES=0 docker run ..."
	@echo ""
	@echo "Docker-compose with GPU control:"
	@echo "  NVIDIA_VISIBLE_DEVICES=all docker-compose run gpu-app"
	@echo "  NVIDIA_VISIBLE_DEVICES=0,1 docker-compose run gpu-app"
	@echo ""
	@echo "Makefile targets:"
	@echo "  make test-gpu-access              # Test all GPUs"
	@echo "  make test-specific-gpu-access     # Test specific GPUs"
	@echo "  make run-shell-all-gpus          # Shell with all GPUs"
	@echo "  make run-shell-specific-gpus     # Shell with specific GPUs"
