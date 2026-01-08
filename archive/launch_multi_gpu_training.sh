#!/bin/bash

# Multi-GPU Training Launcher
# Supports single-machine and multi-node distributed training

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.multi-gpu.yml"
MAX_STEPS=${1:-250000}
BATCH_SIZE=${2:-12}
RESUME_FROM=${3:-""}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker not installed"
    exit 1
fi

# Check docker-compose
if ! docker compose version &> /dev/null; then
    print_error "Docker Compose not installed"
    exit 1
fi

# Check nvidia-docker
if ! command -v nvidia-docker &> /dev/null; then
    print_warning "nvidia-docker not found, using regular docker with GPU support"
    print_info "GPU support requires NVIDIA Container Toolkit"
fi

# Check GPU availability
print_info "Checking GPU availability..."
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")

if [ "$GPU_COUNT" -lt 2 ]; then
    print_error "Multi-GPU training requires at least 2 GPUs"
    print_info "Found $GPU_COUNT GPU(s)"
    exit 1
fi

print_success "Found $GPU_COUNT GPUs"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read -r line; do
    print_info "  $line"
done

# Check disk space
REQUIRED_SPACE=50  # GB
AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    print_warning "Low disk space: ${AVAILABLE_SPACE}GB available (${REQUIRED_SPACE}GB recommended)"
fi

# Check RAM
REQUIRED_RAM=90  # GB
AVAILABLE_RAM=$(free -g | awk 'NR==2 {print $7}')
print_info "System RAM: ~$(free -h | awk 'NR==2 {print $2}') available"

# Prepare data if needed
if [ ! -f "data_clean/train_tokens.bin" ]; then
    print_warning "Training data not found (data_clean/train_tokens.bin)"
    print_info "You may need to run: python scripts/prepare_training_data.py"
    print_info "Continuing anyway with dummy data loader..."
fi

# Create directories
mkdir -p trained_models/runs/french_medium_multi_gpu
mkdir -p logs

# Update docker-compose with parameters
print_info "Configuring training parameters..."
print_info "  Max steps: $MAX_STEPS"
print_info "  Batch size: $BATCH_SIZE"
if [ -n "$RESUME_FROM" ]; then
    print_info "  Resume from: $RESUME_FROM"
fi

# Build images if needed
print_info "Building Docker images..."
docker compose -f "$COMPOSE_FILE" build --quiet

# Start multi-GPU training
print_success "Starting multi-GPU training..."
print_info "Master (RTX 4090): RANK=0"
print_info "Worker (RTX 5080): RANK=1"
print_info ""
print_info "Logs:"
print_info "  Master: docker logs -f french-llm-training-master"
print_info "  Worker: docker logs -f french-llm-training-worker"
print_info ""
print_info "Dashboard: http://localhost:5174"
print_info "API Metrics: http://localhost:8000/api/metrics"
print_info ""

docker compose -f "$COMPOSE_FILE" up --profile multi-gpu

# Cleanup on exit
print_info "Shutting down..."
docker compose -f "$COMPOSE_FILE" down

print_success "Training session completed!"
echo ""
print_info "Checkpoints saved to: trained_models/runs/french_medium_multi_gpu/"
print_info "Logs saved to: trained_models/runs/french_medium_multi_gpu/training_log.jsonl"
