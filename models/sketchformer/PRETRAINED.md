# Pretrained Sketchformer Assets

This document describes pretrained Sketchformer assets used by the native
PyTorch fine-tuning path.

The original CVPR continuous checkpoint is a TensorFlow checkpoint family, not
a single portable weight file. A usable pretrained source includes:

- `config.json`
- `weights/checkpoint`
- one or more `weights/*.index` files
- matching `weights/*.data-*` shard files
- optional evaluation plots under `plots/`

## Why This Exists Here

Native fine-tuning needs a reliable handoff from the original TensorFlow
checkpoint layout into the in-repo PyTorch model. Before implementing the
variable-by-variable conversion table, we need a stable way to answer:

- Is the pretrained directory present?
- Which checkpoints are available?
- Is the recommended checkpoint complete?
- Which architecture settings came from the original run?
- Are previous evaluation plots available for qualitative comparison?

## Inspect The Assets

```bash
python scripts/sketchformer/inspect_pretrained.py
```

The default root is:

```text
weights/pretrained/sketch-transformer-tf2-cvpr_tform_cont
```

Print JSON instead of a text report:

```bash
python scripts/sketchformer/inspect_pretrained.py --json
```

## Conversion

The tok-dict conversion is implemented in:

```text
scripts/sketchformer/convert_checkpoint.py
models/sketchformer/checkpoint_mapping.py
```

For the long-sequence path, use experiment
`anime_tok_dict_long`. It preserves the released 200-position dense
expander and initializes the separate 4096-position residual to zero, while all
shared encoder, decoder, embedding, pooling, and reconstruction tensors are
mapped from the TensorFlow checkpoint.

After conversion, run the 200-token extension parity gate:

```bash
tts-check-sketchformer-parity \
  --experiment anime_tok_dict_long \
  --checkpoint weights/pretrained/sketchformer_tok_dict_4096_init.safetensors
```

The gate requires at least `0.999` flattened-logit cosine similarity and
`0.99` token argmax agreement against the exact 200-position reference. Pass
`--reference-npz` with `tokens` and `logits` arrays when an external TensorFlow
reference fixture is available.
