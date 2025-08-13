#!/bin/bash
# entrypoint.sh - Smart entrypoint for gptoss-compressor container
# Automatically runs verification commands on startup, with flexibility for different use cases

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== gptoss-compressor Container Startup ===${NC}"
echo "Working directory: $(pwd)"
echo "Container args: $*"

# Environment variables to control behavior
AUTO_VERIFY="${AUTO_VERIFY:-true}"
AUTO_TEST_ENV="${AUTO_TEST_ENV:-true}"
AUTO_INSTALL_KERNELS="${AUTO_INSTALL_KERNELS:-false}"  # Only if not built-in
RUN_MICROBENCH="${RUN_MICROBENCH:-false}"
SKIP_STARTUP_COMMANDS="${SKIP_STARTUP_COMMANDS:-false}"

# Function to run make command with error handling
run_make_command() {
    local cmd="$1"
    local description="$2"
    
    echo -e "${YELLOW}Running: $description${NC}"
    if make "$cmd"; then
        echo -e "${GREEN}✓ $description completed${NC}"
    else
        echo -e "${RED}✗ $description failed${NC}"
        return 1
    fi
}

# Function to run startup verification commands
run_startup_commands() {
    echo -e "${BLUE}=== Automatic Startup Commands ===${NC}"
    
    # Test environment
    if [ "$AUTO_TEST_ENV" = "true" ]; then
        run_make_command "test-env" "Environment Test"
    fi
    
    # Install kernels if requested and not already present
    if [ "$AUTO_INSTALL_KERNELS" = "true" ]; then
        echo -e "${YELLOW}Checking for QQQ kernels...${NC}"
        if ! python -c "import qqq" 2>/dev/null; then
            echo -e "${YELLOW}QQQ not found, installing...${NC}"
            run_make_command "install-kernels" "QQQ Kernel Installation"
        else
            echo -e "${GREEN}✓ QQQ kernels already available${NC}"
        fi
    fi
    
    # Verify runtime
    if [ "$AUTO_VERIFY" = "true" ]; then
        if [ "$RUN_MICROBENCH" = "true" ]; then
            run_make_command "verify" "Runtime Verification with Microbench"
        else
            echo -e "${YELLOW}Running runtime verification without microbench...${NC}"
            if python compress_gptoss.py verify-runtime; then
                echo -e "${GREEN}✓ Runtime verification completed${NC}"
            else
                echo -e "${RED}✗ Runtime verification failed${NC}"
            fi
        fi
    fi
    
    echo -e "${GREEN}=== Startup Commands Complete ===${NC}"
}

# Main logic
if [ "$SKIP_STARTUP_COMMANDS" = "true" ]; then
    echo -e "${YELLOW}Skipping startup commands (SKIP_STARTUP_COMMANDS=true)${NC}"
elif [ $# -eq 0 ]; then
    # No arguments - run startup commands then drop to shell
    run_startup_commands
    echo -e "${BLUE}=== Dropping to Interactive Shell ===${NC}"
    echo "Available commands:"
    echo "  make verify          # Verify runtime + microbench"
    echo "  make dequant         # Dequantize GPT-OSS"
    echo "  make w4a8exp_w4a16mw # Quantize with W4A8 experts"
    echo "  make help            # See all targets"
    echo ""
    exec bash
elif [ "$1" = "bash" ] || [ "$1" = "sh" ] || [ "$1" = "/bin/bash" ] || [ "$1" = "/bin/sh" ]; then
    # Explicit shell request - run startup commands then shell
    run_startup_commands
    echo -e "${BLUE}=== Starting Shell ===${NC}"
    exec "$@"
elif [ "$1" = "make" ]; then
    # Make command - run startup commands first, then the make command
    if [ "$SKIP_STARTUP_COMMANDS" != "true" ]; then
        run_startup_commands
    fi
    echo -e "${BLUE}=== Running Make Command: ${*:2} ===${NC}"
    exec "$@"
elif [ "$1" = "python" ] && [[ "$2" == *"compress_gptoss.py"* ]]; then
    # Direct python command - optionally run startup, then the command
    if [ "$AUTO_VERIFY" = "true" ] && [ "$SKIP_STARTUP_COMMANDS" != "true" ]; then
        echo -e "${YELLOW}Running quick verification before command...${NC}"
        python compress_gptoss.py verify-runtime || echo -e "${RED}Verification failed, continuing anyway...${NC}"
    fi
    echo -e "${BLUE}=== Running Command: $* ===${NC}"
    exec "$@"
else
    # Any other command - run as-is (but optionally verify first)
    if [ "$AUTO_VERIFY" = "true" ] && [ "$SKIP_STARTUP_COMMANDS" != "true" ]; then
        echo -e "${YELLOW}Running quick verification before command...${NC}"
        python compress_gptoss.py verify-runtime || echo -e "${RED}Verification failed, continuing anyway...${NC}"
    fi
    echo -e "${BLUE}=== Running Command: $* ===${NC}"
    exec "$@"
fi
