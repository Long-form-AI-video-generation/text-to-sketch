# Metrics

This folder contains evaluation utilities for two stages of the project.

## Preprocessing Metrics

These modules evaluate the sketch extraction and vectorization pipeline:

- `preprocessing/compare_rdp_epsilon.py` ranks dense sketches and compares RDP
  simplification.
- `preprocessing/evaluate_encoder.py` checks sketch-token encode/decode
  reconstruction error.
- `preprocessing/evaluate_ordering.py` visualizes stroke ordering strategies.
- `preprocessing/evaluate_centerline_preprocessing.py` compares extractor/threshold
  profiles on common source images, reports token-decoded fidelity and drift,
  enforces the centerline gates, and writes manual-review pairs.
- `preprocessing/visualisation.py` contains plotting helpers for
  original-vs-simplified sketches.

## Native Sketchformer Metrics

These modules support the in-repo Sketchformer fine-tuning path:

- `sketchformer/reconstruction.py` converts continuous outputs or codebook-
  decoded tok-dict outputs into stroke3 predictions, collects target/prediction
  examples, and writes JSON metric reports.
- `sketchformer/visualisation.py` saves target-vs-prediction reconstruction
  plots for qualitative evaluation.

Use the native evaluation script to produce artifacts:

```bash
python scripts/sketchformer/evaluate.py \
  --experiment anime_tok_dict_finetune \
  --checkpoint weights/finetuned/sketchformer-tok-dict-anime/last.pt \
  --metrics-output weights/finetuned/sketchformer-tok-dict-anime/eval_metrics.json \
  --plots-output-dir weights/finetuned/sketchformer-tok-dict-anime/reconstruction_plots \
  --num-plots 8
```

Final long-sequence evaluation defaults to free-running cached decoding and reports the
geometry F1 separately for 1-512, 513-1024, 1025-2048, and 2049-4096 token
groups, including per-bucket medians. During early fine-tuning, prioritize:

- token loss
- token accuracy
- token perplexity
- codebook-decoded target-vs-prediction plots
