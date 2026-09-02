#!/usr/bin/env python3
"""Prepare Stroke-3 chunks using legacy NumPy-compatible pickles.

The original Sketchformer stack runs on Python 3.6 / NumPy 1.18. Object arrays
saved by newer NumPy versions may not unpickle there, so this script is kept
self-contained and Python 3.6-compatible for running inside the Sketchformer
Docker image.
"""

import argparse
import os

import numpy as np


def load_stroke5_file(path):
    data = np.load(path, allow_pickle=True)
    if len(data.files) != 1:
        raise ValueError("Expected one array in {}, found {}".format(path, data.files))
    return data[data.files[0]]


def stroke5_to_stroke3(stroke5):
    if stroke5.ndim != 2 or stroke5.shape[1] != 5:
        raise ValueError("Expected stroke5 shape (N,5), got {}".format(stroke5.shape))
    if stroke5.shape[0] < 2:
        raise ValueError("Stroke5 sketch is too short to contain sentinel row")

    stroke5 = stroke5[:-1]
    dx_dy = stroke5[:, :2]
    pen_state = stroke5[:, 3]
    return np.concatenate([dx_dy, pen_state[:, None]], axis=1).astype(np.float32)


def build_splits(sketches, seed, train_frac, valid_frac, test_frac):
    if not np.isclose(train_frac + valid_frac + test_frac, 1.0):
        raise ValueError("train/valid/test fractions must sum to 1")

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(sketches))

    train_end = int(len(sketches) * train_frac)
    valid_end = train_end + int(len(sketches) * valid_frac)

    return (
        [sketches[i] for i in indices[:train_end]],
        [sketches[i] for i in indices[train_end:valid_end]],
        [sketches[i] for i in indices[valid_end:]],
    )


def chunk_sketch_data(sketches, labels, n_chunks):
    chunk_size = int(np.ceil(float(len(sketches)) / float(n_chunks)))
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(sketches))
        if start >= end:
            break
        chunks.append((sketches[start:end], labels[start:end]))
    return chunks


def save_chunk(path, sketches, labels):
    np.savez_compressed(
        path,
        x=np.array(sketches, dtype=object),
        y=np.array(labels, dtype=np.int32),
    )


def compute_scale_stats(sketches):
    all_deltas = np.concatenate([sketch[:, :2] for sketch in sketches], axis=0)
    return float(np.std(all_deltas)), float(np.mean(all_deltas))


def repeat_to_min_size(sketches, min_size):
    if min_size <= 0 or len(sketches) >= min_size:
        return sketches
    if not sketches:
        raise ValueError("Cannot repeat an empty split")

    repeated = list(sketches)
    cursor = 0
    while len(repeated) < min_size:
        repeated.append(np.copy(sketches[cursor % len(sketches)]))
        cursor += 1
    return repeated


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare legacy Sketchformer Stroke-3 chunks.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--n-chunks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--n-classes", type=int, default=345)
    parser.add_argument("--min-valid-size", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_chunks <= 0:
        raise ValueError("--n-chunks must be positive")

    if not os.path.isdir(args.target_dir):
        os.makedirs(args.target_dir)

    file_list = [
        os.path.join(args.source_dir, name)
        for name in sorted(os.listdir(args.source_dir))
        if name.endswith(".npz")
    ]
    if not file_list:
        raise IOError("No .npz files found in {}".format(args.source_dir))

    print("Found {} source files in {}".format(len(file_list), args.source_dir))
    sketches = [stroke5_to_stroke3(load_stroke5_file(path)) for path in file_list]

    train_sketches, valid_sketches, test_sketches = build_splits(
        sketches, args.seed, args.train_frac, args.valid_frac, args.test_frac)
    if args.min_valid_size and not valid_sketches:
        valid_sketches = [np.copy(sketches[0])]
    valid_sketches = repeat_to_min_size(valid_sketches, args.min_valid_size)

    train_labels = [0] * len(train_sketches)
    valid_labels = [0] * len(valid_sketches)
    test_labels = [0] * len(test_sketches)

    for chunk_idx, (chunk_sketches, chunk_labels) in enumerate(
            chunk_sketch_data(train_sketches, train_labels, args.n_chunks)):
        chunk_file = os.path.join(args.target_dir, "train_{:03}.npz".format(chunk_idx))
        print("Saving {}".format(chunk_file))
        save_chunk(chunk_file, chunk_sketches, chunk_labels)

    save_chunk(os.path.join(args.target_dir, "valid.npz"), valid_sketches, valid_labels)
    save_chunk(os.path.join(args.target_dir, "test.npz"), test_sketches, test_labels)

    std, mean = compute_scale_stats(train_sketches)
    np.savez(
        os.path.join(args.target_dir, "meta.npz"),
        std=std,
        mean=mean,
        class_names=np.array(["class_{}".format(i) for i in range(args.n_classes)]),
        n_classes=args.n_classes,
        n_samples_train=len(train_sketches),
        n_samples_valid=len(valid_sketches),
        n_samples_test=len(test_sketches),
    )
    print("Saved legacy Sketchformer data to {}".format(args.target_dir))


if __name__ == "__main__":
    main()
