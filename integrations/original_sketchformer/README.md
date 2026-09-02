# Sketchformer Codebase Fine-Tuning Integration

This folder contains the adapter for fine-tuning with the original
Sketchformer codebase.

Text-to-Sketch remains responsible for generating and preparing stroke data.
After the handoff point, this integration launches the local git-ignored
`sketchformer/` checkout for training, checkpoint loading, loss computation,
and evaluation.

The adapter does not reimplement Sketchformer's model, dataloader, masks,
losses, or checkpoint structure.

## Contents

| Path | Purpose |
|---|---|
| `launcher.py` | Builds Docker commands for original Sketchformer training and evaluation. |
| `docker/Dockerfile.cpu` | CPU TensorFlow 2.1 image used to run the legacy code. |
| `docker/Dockerfile.gpu` | TensorFlow GPU image intended for RTX 3090 server experiments. |

The original Sketchformer release used an old TensorFlow/CUDA stack. Use the
CPU image for local compatibility checks and the GPU image for RTX 3090 server
experiments.

## Common Commands

Build the image:

```bash
python scripts/integrations/sketchformer_codebase_finetune.py --sudo build-image
```

Build the GPU image:

```bash
python scripts/integrations/sketchformer_codebase_finetune.py --sudo build-gpu-image
```

Prepare legacy-compatible stroke3 chunks:

```bash
python scripts/integrations/sketchformer_codebase_finetune.py --sudo prepare-data \
  --source-dir data/processed/stroke5 \
  --target-dir data/processed/sketchformer-ready-data/stroke3 \
  --n-chunks 10 \
  --n-classes 345
```

Evaluate pretrained reconstruction:

```bash
python scripts/integrations/sketchformer_codebase_finetune.py --sudo evaluate-reconstruction
```

Fine-tune on CPU:

```bash
python scripts/integrations/sketchformer_codebase_finetune.py --sudo finetune-continuous
```

Dry-run the GPU command shape:

```bash
python scripts/integrations/sketchformer_codebase_finetune.py \
  --sudo \
  --image sketchformer-tf2-gpu \
  --gpus all \
  --dry-run \
  finetune-continuous
```

Keep `--max-seq-len 200` when resuming the released continuous checkpoint. The
legacy TensorFlow model has sequence-length-dependent layers; larger values are
for from-scratch experiments.
