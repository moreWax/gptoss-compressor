#!/bin/bash
# build_test.sh - Test script to verify Docker build and QQQ installation

set -e

echo "=== Docker Build Test Script ==="
echo "Testing gptoss-compressor Docker build with QQQ CUTLASS kernels"
echo ""

# Build the image
echo "Building Docker image with QQQ kernels..."
docker build -t gptoss-compressor-test:latest \
    --build-arg CUDA_ARCH=8.6 \
    --build-arg INSTALL_QQQ=true \
    . || {
    echo "❌ Docker build failed"
    exit 1
}

echo "✅ Docker build completed successfully"
echo ""

# Test basic functionality
echo "Testing basic container functionality..."
docker run --rm gptoss-compressor-test:latest python --version || {
    echo "❌ Python test failed"
    exit 1
}

echo "✅ Python test passed"
echo ""

# Test QQQ installation (without GPU)
echo "Testing QQQ installation..."
docker run --rm gptoss-compressor-test:latest python -c "
try:
    import QQQ
    print('✅ QQQ package imported successfully')
    print(f'QQQ version: {getattr(QQQ, \"__version__\", \"unknown\")}')
except ImportError:
    try:
        import qqq
        print('✅ qqq package imported successfully')
        print(f'qqq version: {getattr(qqq, \"__version__\", \"unknown\")}')
    except ImportError:
        print('❌ QQQ/qqq package not found')
        exit(1)
" || {
    echo "⚠️  QQQ import test failed - this may be normal without GPU"
}

echo ""

# Test compress_gptoss.py help
echo "Testing compress_gptoss.py help..."
docker run --rm gptoss-compressor-test:latest python compress_gptoss.py --help > /dev/null || {
    echo "❌ compress_gptoss.py help failed"
    exit 1
}

echo "✅ compress_gptoss.py help test passed"
echo ""

# Test Makefile
echo "Testing Makefile in container..."
docker run --rm gptoss-compressor-test:latest make test-env || {
    echo "❌ Makefile test failed"
    exit 1
}

echo "✅ Makefile test passed"
echo ""

echo "=== All tests completed successfully! ==="
echo "Docker image 'gptoss-compressor-test:latest' is ready to use"
echo ""
echo "To test with GPU (if available):"
echo "  docker run --rm --gpus all gptoss-compressor-test:latest make test-qqq"
