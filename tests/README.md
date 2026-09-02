# Tests

This folder contains standard-library `unittest` tests for the native
Sketchformer fine-tuning path.

The tests follow the original flat test layout:

```text
tests/
  test_config_composition.py
  test_stroke_sequence_dataset.py
  test_collate_masks.py
  test_sketchformer_forward.py
  test_losses.py
  test_prepare_sketchformer_tokens.py
  test_checkpoint_mapping.py
  test_train_smoke.py
  test_centerline_preprocessing.py
  test_long_sequence.py
```

They avoid real datasets, Docker, GPU, and large checkpoints. Temporary toy
fixtures are created inside each test.

Run all tests:

```bash
python -B -m unittest discover -s tests
```

Run the deterministic periodic long-sequence eval:

```bash
python -B evals/long_sequence_eval.py
```

Run the serial/parallel preprocessing equivalence eval:

```bash
python -B evals/parallel_preprocessing_eval.py
```

The `-B` flag avoids writing `__pycache__` files into the repository.
