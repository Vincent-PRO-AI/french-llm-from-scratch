#!/bin/bash
# Launch DDP training via Docker with memory flexibility
# Usage: ./launch_ddp_docker.sh [up|down|logs|stop]

set -e

ACTION="${1:-up}"
COMPOSE_FILE="docker-compose.ddp.yml"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

case "$ACTION" in
    up|start)
        echo -e "${BLUE}🚀 Launching DDP training via Docker${NC}"
        echo "Configuration:"
        echo "  - GPU 0 (RTX 4090): Rank 0"
        echo "  - GPU 1 (RTX 5080): Rank 1"
        echo "  - Batch size: 4 per GPU (8 total with DDP)"
        echo "  - Gradient accumulation: 2x"
        echo "  - Max steps: 350,000 (100k new steps from checkpoint 250k)"
        echo "  - Duration: ~8-10 hours"
        echo ""
        echo "Memory flexibility:"
        echo "  ✓ VRAM → RAM fallback enabled"
        echo "  ✓ CPU offload enabled"
        echo "  ✓ Gradient checkpointing enabled"
        echo "  ✓ NVMe cache mounted (/nvme_cache)"
        echo ""
        
        # Build images
        echo -e "${YELLOW}Building Docker images...${NC}"
        docker compose -f "$COMPOSE_FILE" build
        
        # Start containers
        echo -e "${YELLOW}Starting DDP containers...${NC}"
        docker compose -f "$COMPOSE_FILE" up -d
        
        echo -e "${GREEN}✅ DDP training started!${NC}"
        echo ""
        echo "Monitor with:"
        echo "  ./launch_ddp_docker.sh logs       # View live logs"
        echo "  ./launch_ddp_docker.sh status     # Check status"
        echo ""
        ;;
    
    logs)
        echo -e "${BLUE}📊 DDP Training Logs${NC}"
        docker compose -f "$COMPOSE_FILE" logs -f --tail=50
        ;;
    
    status)
        echo -e "${BLUE}📊 DDP Container Status${NC}"
        docker compose -f "$COMPOSE_FILE" ps
        
        echo ""
        echo -e "${BLUE}GPU Memory Status:${NC}"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader
        ;;
    
    stop)
        echo -e "${YELLOW}⏸️  Stopping DDP training...${NC}"
        docker compose -f "$COMPOSE_FILE" stop
        echo -e "${GREEN}✅ Stopped${NC}"
        ;;
    
    down|cleanup)
        echo -e "${YELLOW}🗑️  Cleaning up DDP containers...${NC}"
        docker compose -f "$COMPOSE_FILE" down
        echo -e "${GREEN}✅ Cleaned up${NC}"
        ;;
    
    attach-rank0)
        echo -e "${BLUE}Attaching to Rank 0 container...${NC}"
        docker attach french-llm-ddp-rank0
        ;;
    
    attach-rank1)
        echo -e "${BLUE}Attaching to Rank 1 container...${NC}"
        docker attach french-llm-ddp-rank1
        ;;
    
    exec-rank0)
        echo -e "${BLUE}Executing shell in Rank 0 container...${NC}"
        docker exec -it french-llm-ddp-rank0 bash
        ;;
    
    exec-rank1)
        echo -e "${BLUE}Executing shell in Rank 1 container...${NC}"
        docker exec -it french-llm-ddp-rank1 bash
        ;;
    
    test-50)
        echo -e "${BLUE}🧪 Testing DDP with 50 steps (2 min)...${NC}"
        
        # Create test compose file with 50 steps
        cat > docker-compose.ddp.test.yml << 'EOF'
version: '3.8'

services:
  ddp-rank0:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: french-llm-ddp-test-rank0
    command: >
      python -m torch.distributed.launch
      --nproc_per_node=1
      --nnodes=2
      --node_rank=0
      --master_addr=ddp-master-test
      --master_port=29500
      scripts/train_subtitles_transformer.py
      --arch-preset medium
      --pretokenized-path /workspace/data_clean/conversations_mega_train_mistral_tokenized.pt
      --resume-from /workspace/trained_models/checkpoint_250k_fixed.pt
      --max-steps 250050
      --batch-size 4
      --log-dir /workspace/trained_models/runs
      --gradient-accumulation-steps 2
      --cpu-offload
      --gradient-checkpointing
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - NCCL_DEBUG=INFO
    volumes:
      - ./data_clean:/workspace/data_clean
      - ./trained_models:/workspace/trained_models
      - ./scripts:/workspace/scripts
      - ./utils:/workspace/utils
      - /nvme_cache:/nvme_cache:rw
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    networks:
      - ddp-test-network
    depends_on:
      - ddp-master-test

  ddp-rank1:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: french-llm-ddp-test-rank1
    command: >
      python -m torch.distributed.launch
      --nproc_per_node=1
      --nnodes=2
      --node_rank=1
      --master_addr=ddp-master-test
      --master_port=29500
      scripts/train_subtitles_transformer.py
      --arch-preset medium
      --pretokenized-path /workspace/data_clean/conversations_mega_train_mistral_tokenized.pt
      --resume-from /workspace/trained_models/checkpoint_250k_fixed.pt
      --max-steps 250050
      --batch-size 4
      --log-dir /workspace/trained_models/runs
      --gradient-accumulation-steps 2
      --cpu-offload
      --gradient-checkpointing
    environment:
      - CUDA_VISIBLE_DEVICES=1
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - NCCL_DEBUG=INFO
    volumes:
      - ./data_clean:/workspace/data_clean
      - ./trained_models:/workspace/trained_models
      - ./scripts:/workspace/scripts
      - ./utils:/workspace/utils
      - /nvme_cache:/nvme_cache:rw
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
    networks:
      - ddp-test-network
    depends_on:
      - ddp-master-test

  ddp-master-test:
    image: ubuntu:22.04
    container_name: french-llm-ddp-master-test
    command: /bin/bash -c "while true; do sleep 10; done"
    networks:
      - ddp-test-network

networks:
  ddp-test-network:
    driver: bridge
EOF
        
        docker compose -f docker-compose.ddp.test.yml build
        docker compose -f docker-compose.ddp.test.yml up -d
        
        echo -e "${GREEN}✅ Test started (50 steps = ~2 min)${NC}"
        echo "Monitor: docker compose -f docker-compose.ddp.test.yml logs -f"
        ;;
    
    *)
        echo "Usage: $0 {up|down|logs|status|stop|attach-rank0|attach-rank1|exec-rank0|exec-rank1|test-50}"
        echo ""
        echo "Commands:"
        echo "  up              - Start DDP training (100k steps)"
        echo "  down            - Stop and remove containers"
        echo "  logs            - Show live logs"
        echo "  status          - Show container status and GPU memory"
        echo "  stop            - Stop containers (keep volume)"
        echo "  attach-rank0    - Attach to Rank 0 output"
        echo "  attach-rank1    - Attach to Rank 1 output"
        echo "  exec-rank0      - Open shell in Rank 0 container"
        echo "  exec-rank1      - Open shell in Rank 1 container"
        echo "  test-50         - Quick test with 50 steps"
        exit 1
        ;;
esac
