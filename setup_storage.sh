#!/bin/bash
# setup_storage.sh - Configure persistent model storage for gptoss-compressor
# Usage: ./setup_storage.sh [STORAGE_PATH]

set -euo pipefail

DEFAULT_STORAGE_PATH="${HOME}/gptoss-models"
STORAGE_PATH="${1:-$DEFAULT_STORAGE_PATH}"

echo "Setting up gptoss-compressor persistent storage..."
echo "Storage location: $STORAGE_PATH"

# Create directory structure
mkdir -p "$STORAGE_PATH"/{models,offload_cache,huggingface_cache}

# Create symlinks in project directory
ln -sf "$STORAGE_PATH/models" ./models
ln -sf "$STORAGE_PATH/offload_cache" ./offload_cache

echo "✅ Created storage directories:"
echo "   Models:           $STORAGE_PATH/models"
echo "   Offload cache:    $STORAGE_PATH/offload_cache" 
echo "   HuggingFace cache: $STORAGE_PATH/huggingface_cache"
echo ""
echo "✅ Created symlinks:"
echo "   ./models -> $STORAGE_PATH/models"
echo "   ./offload_cache -> $STORAGE_PATH/offload_cache"
echo ""

# Create environment file for docker-compose
cat > .env.storage << EOF
# Custom storage paths for gptoss-compressor
MODELS_DIR=$STORAGE_PATH/models
CACHE_DIR=$STORAGE_PATH/offload_cache
HF_CACHE_DIR=$STORAGE_PATH/huggingface_cache
EOF

echo "✅ Created .env.storage file with custom paths"
echo ""
echo "Usage with Docker:"
echo "  # Use default mounts (via symlinks)"
echo "  docker-compose run dequantize"
echo ""
echo "  # Use custom paths explicitly"
echo "  MODELS_DIR=$STORAGE_PATH/models CACHE_DIR=$STORAGE_PATH/offload_cache docker-compose run dequantize"
echo ""
echo "  # Use with docker run"
echo "  docker run --rm --gpus all \\"
echo "    -v '$STORAGE_PATH/models:/workspace/app/models' \\"
echo "    -v '$STORAGE_PATH/offload_cache:/workspace/app/offload_cache' \\"
echo "    -v '$STORAGE_PATH/huggingface_cache:/root/.cache/huggingface' \\"
echo "    -v '\$(pwd):/workspace/app' \\"
echo "    gptoss-compressor:cu128"
echo ""

# Check disk space
echo "💾 Storage Requirements:"
echo "   - Source models: ~240GB (GPT-OSS-120B MXFP4)"
echo "   - Dequantized: ~240GB (FP16 output)"
echo "   - Quantized: 30-120GB (depending on scheme)"
echo "   - Processing cache: 32-64GB (temporary)"
echo "   - Total recommended: 100GB+ free space (sequential device mapping)"
echo "   - Alternative: 500GB+ free space (auto device mapping)"
echo ""

if command -v df >/dev/null 2>&1; then
    available=$(df -h "$STORAGE_PATH" | tail -1 | awk '{print $4}')
    echo "📊 Available space at $STORAGE_PATH: $available"
else
    echo "📊 Check available disk space manually"
fi

echo ""
echo "🚀 Setup complete! Your models will persist in:"
echo "   $STORAGE_PATH"
