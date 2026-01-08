#!/bin/bash

# 🎯 QUICK START GUIDE - French LLM Training

set -e

echo "════════════════════════════════════════════════════════════"
echo "🚀 QUICK START - French LLM Training (Mistral Architecture)"
echo "════════════════════════════════════════════════════════════"
echo ""

# Step 1: Activation
echo "Step 1️⃣  : Activate Conda Environment"
echo "─────────────────────────────────────────────────────────"
echo "Command:"
echo "  conda activate french-llm"
echo ""
echo "Or use the full path:"
echo "  source /home/vincent/miniconda3/bin/activate french-llm"
echo ""

# Step 2: Verify
echo "Step 2️⃣  : Verify Setup (Run in activated env)"
echo "─────────────────────────────────────────────────────────"
echo "Command:"
echo "  python -c \"import torch; print(f'GPU: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')\""
echo ""

# Step 3: Launch Training
echo "Step 3️⃣  : Launch Phase 3 (Training - 12-16 hours)"
echo "─────────────────────────────────────────────────────────"
echo "Command:"
echo ""
echo "  cd /home/vincent/code/repo/french-llm-from-scratch"
echo "  python scripts/02_train_mistral.py \\"
echo "    --data-path ./data_clean/conversations_mega_train_mistral_tokenized.pt \\"
echo "    --output-dir ./trained_models/french_medium_v2 \\"
echo "    --max-steps 100000"
echo ""

# Step 4: Monitor
echo "Step 4️⃣  : Monitor Training (In separate terminals)"
echo "─────────────────────────────────────────────────────────"
echo "GPU stats (every 1 sec):"
echo "  nvidia-smi -l 1"
echo ""
echo "Training metrics:"
echo "  tail -f ./trained_models/french_medium_v2/metrics.jsonl"
echo ""
echo "TensorBoard (optional):"
echo "  tensorboard --logdir ./trained_models/french_medium_v2"
echo ""

# Step 5: Export
echo "Step 5️⃣  : After Training - Export to GGUF (10 min)"
echo "─────────────────────────────────────────────────────────"
echo "Command:"
echo ""
echo "  python scripts/03_export_gguf.py \\"
echo "    --model-path ./trained_models/french_medium_v2/checkpoint_final \\"
echo "    --output-dir ./models_gguf"
echo ""

echo "════════════════════════════════════════════════════════════"
echo "📊 System Configuration"
echo "════════════════════════════════════════════════════════════"
echo "GPU: RTX 5080 (17.1GB VRAM)"
echo "Python: 3.10.19"
echo "PyTorch: 2.6.0+cu124"
echo "CUDA: 12.4 (system 13.1)"
echo "Conda env: /home/vincent/miniconda3/envs/french-llm"
echo ""

echo "════════════════════════════════════════════════════════════"
echo "📖 Documentation"
echo "════════════════════════════════════════════════════════════"
echo "• SETUP_COMPLETE.md  → Full setup report"
echo "• STATUS.md          → Current project status"
echo "• LAUNCH_GUIDE.md    → Launch options"
echo "• CONDA_GUIDE.md     → Conda usage guide"
echo ""

echo "════════════════════════════════════════════════════════════"
echo "✅ READY TO START TRAINING!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "1. Close Chrome (free 2-4GB RAM)"
echo "2. Run: conda activate french-llm"
echo "3. Run: python scripts/02_train_mistral.py ..."
echo ""
