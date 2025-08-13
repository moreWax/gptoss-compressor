#!/bin/bash
# demo_one_click.sh - Demonstrate the one-click dequantization service

set -euo pipefail

echo "🚀 GPT-OSS One-Click Dequantization Demo"
echo "========================================"
echo ""

# Check if Docker and docker-compose are available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install docker-compose first."
    exit 1
fi

# Check if the image exists
if ! docker images | grep -q "gptoss-compressor"; then
    echo "🔨 Docker image not found. Building it first..."
    make build-image
else
    echo "✅ Docker image found: gptoss-compressor:cu128"
fi

echo ""
echo "📋 Available One-Click Options:"
echo ""
echo "1. 🎯 Full Service (with verification) → ./fp16/"
echo "   Command: docker-compose run --rm dequantize-to-local"
echo ""
echo "2. ⚡ Quick Service (no verification) → ./fp16/"
echo "   Command: docker-compose run --rm dequant-quick"
echo ""
echo "3. 🗂️  Custom Location → /path/of/your/choice/"
echo "   Command: OUTPUT_DIR=/path/to/models docker-compose run --rm dequantize-to-local"
echo ""

# Check available disk space
echo "💾 Current Disk Space:"
df -h . | head -2
echo ""

# Estimate requirements
echo "📊 Requirements for GPT-OSS-120B dequantization:"
echo "   • Disk space: ~100GB (sequential device mapping) or 500GB+ (auto device mapping)"
echo "   • GPU memory: 24GB+ VRAM (or use CPU offloading)"
echo "   • System RAM: 32-64GB recommended"
echo "   • Time: 30-60 minutes"
echo ""

# Interactive selection
echo "🤔 What would you like to do?"
echo ""
echo "a) Run full dequantization service (default ./fp16/)"
echo "b) Run quick dequantization service (no verification)"
echo "c) Run with custom output directory"
echo "d) Show service logs from last run"
echo "e) Just show the commands without running"
echo "q) Quit"
echo ""

read -p "Choose an option [a/b/c/d/e/q]: " choice

case $choice in
    a|A)
        echo ""
        echo "🚀 Starting full dequantization service..."
        echo "This will save the model to ./fp16/gpt-oss-120b-fp16/"
        echo ""
        read -p "Continue? [y/N]: " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            echo "Running: docker-compose run --rm dequantize-to-local"
            docker-compose run --rm dequantize-to-local
        else
            echo "Cancelled."
        fi
        ;;
    b|B)
        echo ""
        echo "⚡ Starting quick dequantization service..."
        echo "This skips verification for faster startup"
        echo ""
        read -p "Continue? [y/N]: " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            echo "Running: docker-compose run --rm dequant-quick"
            docker-compose run --rm dequant-quick
        else
            echo "Cancelled."
        fi
        ;;
    c|C)
        echo ""
        read -p "Enter output directory path: " output_dir
        if [[ -n "$output_dir" ]]; then
            echo "🗂️  Starting dequantization with custom output: $output_dir"
            echo ""
            read -p "Continue? [y/N]: " confirm
            if [[ $confirm =~ ^[Yy]$ ]]; then
                echo "Running: OUTPUT_DIR='$output_dir' docker-compose run --rm dequantize-to-local"
                OUTPUT_DIR="$output_dir" docker-compose run --rm dequantize-to-local
            else
                echo "Cancelled."
            fi
        else
            echo "No directory specified. Cancelled."
        fi
        ;;
    d|D)
        echo ""
        echo "📋 Recent service logs:"
        docker-compose logs --tail=50 dequantize-to-local || echo "No logs found."
        ;;
    e|E)
        echo ""
        echo "📝 One-Click Commands:"
        echo ""
        echo "# Default location (./fp16/):"
        echo "docker-compose run --rm dequantize-to-local"
        echo ""
        echo "# Quick version (no verification):"
        echo "docker-compose run --rm dequant-quick"
        echo ""
        echo "# Custom location:"
        echo "OUTPUT_DIR=/your/path docker-compose run --rm dequantize-to-local"
        echo ""
        echo "# With specific GPUs:"
        echo "NVIDIA_VISIBLE_DEVICES=0,1 docker-compose run --rm dequantize-to-local"
        echo ""
        ;;
    q|Q)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid option. Please choose a, b, c, d, e, or q."
        exit 1
        ;;
esac

echo ""
echo "✅ Demo completed!"
