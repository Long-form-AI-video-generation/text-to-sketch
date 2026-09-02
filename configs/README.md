# Training Configs

This folder contains the configuration layer for in-repo Sketchformer
fine-tuning.

The configs are intentionally split by responsibility:

- `data/` controls dataset paths, sequence length, batching, and dataloader
  behavior.
- `model/` controls the native PyTorch Sketchformer-style architecture.
- `optimizer/` controls optimizer and learning-rate scheduler settings.
- `trainer/` controls device, precision, logging, checkpointing, and runtime
  behavior.
- `experiment/` selects concrete training runs such as smoke tests and anime
  fine-tuning.

The default entrypoint config is `train.yaml`.

## Research Objective

The in-repo fine-tuning path targets more sophisticated sketches than the
short, simple QuickDraw-style drawings used by the original Sketchformer setup.
For that reason, the base data/model configs are long-sequence capable:

- the default model consumes tok-dict token sequences built from the sketch
  token dictionary using the released TensorFlow token ID layout;
- the model supports sequences up to `2048` tok-dict tokens;
- the anime tok-dict data config trains at `2048` by default for server-side
  runs;
- the smoke-test experiment overrides sequence length down to `256`;
- attention is configured for PyTorch scaled dot-product attention with Flash
  Attention preferred when the hardware supports it;
- SDPA padding masks use broadcast key masks instead of batch-sized square
  tensors, allowing long runs to remain on memory-efficient attention paths;
- sparse attention is represented in config but disabled until the dense
  Flash/SDPA baseline is correct.

For checkpoint-compatible runs, `data/processed/sketch_token/codebook.npy`
must be exported from `sketchformer/prep_data/sketch_token/token_dict.pkl`.
The token IDs are `PAD=0`, motion tokens `1..1000`, `SEP=1001`,
`SOS=1002`, and `EOS=1003`.

`configs/train.yaml` defaults to `anime_tok_dict` plus
`sketchformer_tok_dict`. Continuous stroke3 configs remain in this folder for
legacy compatibility, but they are not the native fine-tuning objective.

The default trainer config uses CUDA `16-mixed` precision and TF32-friendly
runtime settings for an RTX 3090-class server. CPU development should use the
`smoke_test` experiment or explicit CLI overrides.

## 4096-token Long-Sequence Workflow

`experiment/anime_tok_dict_long.yaml` keeps the existing 2048-token run
unchanged and defines the full long-sequence contract: separately converted weights,
complete sequences with truncation disabled, token-budget batches, and the
512/1024/2048/4096 length curriculum. Its dataset and checkpoint paths are
separate from legacy artifacts.

The long-sequence fields are:

- `data.batching.max_tokens_per_batch: 4096`
- `trainer.training.target_tokens_per_step: 32768`
- `model.architecture.latent_expander_base_length: 200`
- `data.sequence.truncate_long_sequences: false`

Curriculum loaders filter by complete sequence length. They never shorten an
example to make it enter an earlier stage.
