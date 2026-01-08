# OpenTelemetry Tracing Setup Instructions

## ✅ What's Been Added

Three tracing modules have been created:

1. **tracing_setup.py** - Initializes OpenTelemetry OTLP exporter
2. **tracing_module.py** - Reusable context managers and decorators
3. **tracing_example.py** - Example implementation with guide

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
```

### 2. Run Example
```bash
python tracing_example.py
```

### 3. Open Trace Viewer (in VS Code)
Use command palette: **AI Toolkit: Open Tracing**

Or from terminal:
```bash
ai-mlstudio.tracing.open
```

## 📋 Integration into Your Training Script

### Minimal Setup
```python
from tracing_setup import setup_tracing
from tracing_module import TracingContext

# At startup
tracer, meter = setup_tracing("french-llm-training", "0.1.0")

# In training loop
ctx = TracingContext("training")

for step in range(num_steps):
    with ctx.trace_operation("training_step", {"step": step}):
        # Your training code
        loss = train_one_step(batch)
        
        if step % checkpoint_interval == 0:
            with ctx.trace_checkpoint(step, loss):
                torch.save(model, path)
```

### With Decorators
```python
from tracing_module import trace_training_step, trace_data_loading

@trace_training_step
def train_step(batch, step):
    # Automatically traced
    return loss

@trace_data_loading
def load_data():
    # Automatically traced
    return dataset
```

## 📊 What Gets Traced

### Training Steps
- Step number
- Batch size
- Training/validation loss
- Execution time
- Success/error status

### Checkpoints
- Step number
- Loss at checkpoint
- Save duration

### Evaluations
- Step number
- Training loss
- Validation loss
- Evaluation metrics

## 🔍 View Traces

1. Keep training script running or let it complete
2. Open trace viewer in AI Toolkit
3. Traces appear automatically as operations complete
4. Filter by operation type, step, loss, etc.

## ⚙️ Custom Endpoint

Set environment variable to use custom OTLP endpoint:
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://your-server:4318
python your_training_script.py
```

## 📚 Context Managers Available

```python
ctx = TracingContext("training")

# Generic operation
with ctx.trace_operation("my_op", {"key": "value"}):
    ...

# Checkpoint save
with ctx.trace_checkpoint(step, loss):
    ...

# Evaluation
with ctx.trace_evaluation(step, {"val_loss": 2.0, "train_loss": 1.9}):
    ...
```

## 🎯 Next Steps

1. Run `python tracing_example.py` to see it in action
2. Integrate into `scripts/train_subtitles_transformer.py`
3. Integrate into `monitor_training_continuous.py`
4. Use trace viewer to debug training issues

---

**Created**: 2026-01-03  
**OTLP Endpoint**: http://localhost:4318 (AI Toolkit default)  
**Status**: ✅ Ready to integrate
