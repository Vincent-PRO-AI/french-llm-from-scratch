#!/usr/bin/env python3
"""
Training Script with OpenTelemetry Tracing Integration
Demonstrates how to add tracing to the main training loop
"""

import sys
from pathlib import Path
import json
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Add tracing setup
try:
    from tracing_setup import setup_tracing
    from tracing_module import TracingContext, trace_training_step
    TRACING_ENABLED = True
except ImportError:
    TRACING_ENABLED = False
    print("⚠️  Tracing dependencies not installed. Run: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp")


def trace_training_example():
    """
    Example of how to integrate tracing into your training script.
    
    This demonstrates:
    1. Initializing tracing at startup
    2. Tracing individual training steps
    3. Tracing checkpoints
    4. Tracing evaluations
    """
    
    if TRACING_ENABLED:
        # Initialize tracing
        tracer, meter = setup_tracing(
            service_name="french-llm-training",
            service_version="0.1.0",
            environment="development"
        )
        print("✅ OpenTelemetry tracing initialized\n")
    else:
        print("⚠️  Tracing disabled - install dependencies to enable\n")
    
    # Example: Trace a training loop
    if TRACING_ENABLED:
        ctx = TracingContext("training")
        
        print("🚀 Starting example training simulation with tracing...\n")
        
        # Simulate training steps
        for step in range(1, 6):
            with ctx.trace_operation(
                "training_step",
                {
                    "step": step,
                    "batch_size": 32,
                }
            ):
                # Simulate training
                import time
                time.sleep(0.5)
                
                # Simulate metrics
                train_loss = 5.0 - (step * 0.3)
                val_loss = 5.2 - (step * 0.28)
                
                print(f"Step {step:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                
                # Trace checkpoint save
                if step % 2 == 0:
                    with ctx.trace_checkpoint(step, val_loss):
                        print(f"  └─ Checkpoint saved at step {step}")
                        time.sleep(0.2)
        
        print("\n✅ Training simulation complete")
        print("   Check AI Toolkit trace viewer for detailed traces")
    else:
        print("⚠️  Cannot run tracing example - install OpenTelemetry first\n")
        print("Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp")


def integration_guide():
    """
    Print integration guide for adding tracing to existing scripts.
    """
    print("\n" + "="*70)
    print("HOW TO INTEGRATE TRACING INTO YOUR TRAINING SCRIPTS")
    print("="*70 + "\n")
    
    guide = """
1. AT THE TOP OF YOUR SCRIPT:
   ─────────────────────────────
   from tracing_setup import setup_tracing
   from tracing_module import TracingContext

2. INITIALIZE TRACING (before main training loop):
   ──────────────────────────────────────────────
   if __name__ == "__main__":
       tracer, meter = setup_tracing(
           service_name="french-llm-training",
           service_version="0.1.0"
       )

3. WRAP TRAINING STEPS:
   ──────────────────────
   ctx = TracingContext("training")
   
   for step in range(num_steps):
       with ctx.trace_operation("training_step", {"step": step}):
           # Your training code here
           loss = train_one_step(...)

4. TRACE CHECKPOINTS:
   ───────────────────
   with ctx.trace_checkpoint(step, loss):
       torch.save(checkpoint, path)

5. TRACE EVALUATIONS:
   ──────────────────
   with ctx.trace_evaluation(step, {"val_loss": val_loss, "train_loss": loss}):
       # Your evaluation code

6. USE DECORATORS (optional):
   ──────────────────────────
   @trace_training_step
   def train_step(batch, step):
       # Code automatically traced

7. START TRACE VIEWER:
   ──────────────────
   In VS Code: Use command palette → "AI Toolkit: Open Tracing"
   Or run: ai-mlstudio.tracing.open

AVAILABLE CONTEXT MANAGERS:
  • ctx.trace_operation(name, attributes)    - Generic operation
  • ctx.trace_checkpoint(step, loss)         - Checkpoint save
  • ctx.trace_evaluation(step, metrics)      - Evaluation run

AVAILABLE DECORATORS:
  • @trace_training_step                     - Trace a training step function
  • @trace_data_loading                      - Trace data loading
  • @trace_model_export                      - Trace model export

ENVIRONMENT VARIABLES:
  • OTEL_EXPORTER_OTLP_ENDPOINT              - Custom OTLP endpoint
                                              (default: http://localhost:4318)
"""
    print(guide)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--guide":
        integration_guide()
    else:
        trace_training_example()
