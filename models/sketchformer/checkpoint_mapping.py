"""Checkpoint helpers for the native PyTorch Sketchformer path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class CheckpointLoadReport:
    """Summary of a partial checkpoint load."""

    missing_keys: list[str]
    unexpected_keys: list[str]


@dataclass(frozen=True)
class ConversionReport:
    """Summary of TensorFlow-to-native checkpoint conversion."""

    converted_keys: list[str]
    skipped_keys: list[str]
    missing_target_keys: list[str]
    initialized_keys: list[str]


def load_torch_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = False,
) -> CheckpointLoadReport:
    """Load a PyTorch state dict and return missing/unexpected key details."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict: dict[str, Any]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError("Checkpoint must be a state-dict-like mapping")

    incompatible = model.load_state_dict(state_dict, strict=strict)
    return CheckpointLoadReport(
        missing_keys=list(incompatible.missing_keys),
        unexpected_keys=list(incompatible.unexpected_keys),
    )


def resize_learned_position_embedding(
    source_weight: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """Resize learned position embeddings by linear interpolation."""

    if source_weight.ndim != 2:
        raise ValueError("Expected positional embedding weight with shape (length, dim)")
    if source_weight.shape[0] == target_length:
        return source_weight

    resized = torch.nn.functional.interpolate(
        source_weight.T.unsqueeze(0),
        size=target_length,
        mode="linear",
        align_corners=False,
    )
    return resized.squeeze(0).T.contiguous()


def resize_vector(source: torch.Tensor, target_length: int) -> torch.Tensor:
    """Resize a 1D vector by linear interpolation."""

    if source.ndim != 1:
        raise ValueError("Expected vector with shape (length,)")
    if source.shape[0] == target_length:
        return source
    resized = F.interpolate(
        source.view(1, 1, -1),
        size=target_length,
        mode="linear",
        align_corners=False,
    )
    return resized.view(-1).contiguous()


def tensorflow_kernel_to_linear_weight(value: torch.Tensor) -> torch.Tensor:
    """Convert a TensorFlow dense kernel ``(in, out)`` to PyTorch ``(out, in)``."""

    if value.ndim != 2:
        raise ValueError("Expected TensorFlow dense kernel with shape (in, out)")
    return value.T.contiguous()


def normalize_tensorflow_key(key: str) -> str:
    """Strip TensorFlow checkpoint suffixes and optimizer-slot path fragments."""

    suffix = "/.ATTRIBUTES/VARIABLE_VALUE"
    if key.endswith(suffix):
        key = key[: -len(suffix)]
    return key


def _to_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value).detach().cpu().to(dtype=torch.float32)


def _assign(
    output: dict[str, torch.Tensor],
    report: list[str],
    target_state: dict[str, torch.Tensor],
    target_key: str,
    value: torch.Tensor,
) -> None:
    expected = target_state[target_key]
    if value.shape != expected.shape:
        raise ValueError(
            f"Shape mismatch for {target_key}: converted {tuple(value.shape)} "
            f"!= target {tuple(expected.shape)}"
        )
    output[target_key] = value.to(dtype=expected.dtype)
    report.append(target_key)


def _skip_key(key: str) -> bool:
    return (
        "OPTIMIZER_SLOT" in key
        or key.startswith("optimizer/")
        or key in {"iter", "save_counter", "current_step", "_CHECKPOINTABLE_OBJECT_GRAPH"}
        or key.startswith("transformer/classify_layer/")
    )


def convert_tok_dict_tensorflow_state(
    tensorflow_state: dict[str, Any],
    target_state: dict[str, torch.Tensor],
    *,
    target_seq_len: int,
) -> tuple[dict[str, torch.Tensor], ConversionReport]:
    """Convert released tok-dict TensorFlow variables into native state keys."""

    converted: dict[str, torch.Tensor] = {}
    converted_keys: list[str] = []
    skipped_keys: list[str] = []
    initialized_keys: list[str] = []

    direct: dict[str, str] = {
        "transformer/encoder/embedding/embeddings": "input_embedding.token_embedding.weight",
        "transformer/decoder/embedding/embeddings": "target_embedding.token_embedding.weight",
        "transformer/bottleneck_layer/W_attn": "pool.W_attn",
        "transformer/bottleneck_layer/b_attn": "pool.b_attn",
        "transformer/bottleneck_layer/V_attn": "pool.V_attn",
        "transformer/output_layer/bias": "reconstruction_head.projection.bias",
    }
    kernel: dict[str, str] = {
        "transformer/output_layer/kernel": "reconstruction_head.projection.weight",
    }
    for layer_index in range(4):
        prefix = f"transformer/encoder/enc_layers/{layer_index}"
        target_prefix = f"encoder.layers.{layer_index}"
        kernel.update(
            {
                f"{prefix}/mha/wq/kernel": f"{target_prefix}.self_attn.q_proj.weight",
                f"{prefix}/mha/wk/kernel": f"{target_prefix}.self_attn.k_proj.weight",
                f"{prefix}/mha/wv/kernel": f"{target_prefix}.self_attn.v_proj.weight",
                f"{prefix}/mha/dense/kernel": f"{target_prefix}.self_attn.out_proj.weight",
                f"{prefix}/ffn/layer-0/kernel": f"{target_prefix}.ffn.fc1.weight",
                f"{prefix}/ffn/layer-1/kernel": f"{target_prefix}.ffn.fc2.weight",
            }
        )
        direct.update(
            {
                f"{prefix}/mha/wq/bias": f"{target_prefix}.self_attn.q_proj.bias",
                f"{prefix}/mha/wk/bias": f"{target_prefix}.self_attn.k_proj.bias",
                f"{prefix}/mha/wv/bias": f"{target_prefix}.self_attn.v_proj.bias",
                f"{prefix}/mha/dense/bias": f"{target_prefix}.self_attn.out_proj.bias",
                f"{prefix}/ffn/layer-0/bias": f"{target_prefix}.ffn.fc1.bias",
                f"{prefix}/ffn/layer-1/bias": f"{target_prefix}.ffn.fc2.bias",
                f"{prefix}/layernorm1/gamma": f"{target_prefix}.norm1.weight",
                f"{prefix}/layernorm1/beta": f"{target_prefix}.norm1.bias",
                f"{prefix}/layernorm2/gamma": f"{target_prefix}.norm2.weight",
                f"{prefix}/layernorm2/beta": f"{target_prefix}.norm2.bias",
            }
        )

        prefix = f"transformer/decoder/dec_layers/{layer_index}"
        target_prefix = f"decoder.layers.{layer_index}"
        for tf_name, native_name in (("mha1", "self_attn"), ("mha2", "cross_attn")):
            kernel.update(
                {
                    f"{prefix}/{tf_name}/wq/kernel": f"{target_prefix}.{native_name}.q_proj.weight",
                    f"{prefix}/{tf_name}/wk/kernel": f"{target_prefix}.{native_name}.k_proj.weight",
                    f"{prefix}/{tf_name}/wv/kernel": f"{target_prefix}.{native_name}.v_proj.weight",
                    f"{prefix}/{tf_name}/dense/kernel": f"{target_prefix}.{native_name}.out_proj.weight",
                }
            )
            direct.update(
                {
                    f"{prefix}/{tf_name}/wq/bias": f"{target_prefix}.{native_name}.q_proj.bias",
                    f"{prefix}/{tf_name}/wk/bias": f"{target_prefix}.{native_name}.k_proj.bias",
                    f"{prefix}/{tf_name}/wv/bias": f"{target_prefix}.{native_name}.v_proj.bias",
                    f"{prefix}/{tf_name}/dense/bias": f"{target_prefix}.{native_name}.out_proj.bias",
                }
            )
        kernel.update(
            {
                f"{prefix}/ffn/layer-0/kernel": f"{target_prefix}.ffn.fc1.weight",
                f"{prefix}/ffn/layer-1/kernel": f"{target_prefix}.ffn.fc2.weight",
            }
        )
        direct.update(
            {
                f"{prefix}/ffn/layer-0/bias": f"{target_prefix}.ffn.fc1.bias",
                f"{prefix}/ffn/layer-1/bias": f"{target_prefix}.ffn.fc2.bias",
                f"{prefix}/layernorm1/gamma": f"{target_prefix}.norm1.weight",
                f"{prefix}/layernorm1/beta": f"{target_prefix}.norm1.bias",
                f"{prefix}/layernorm2/gamma": f"{target_prefix}.norm2.weight",
                f"{prefix}/layernorm2/beta": f"{target_prefix}.norm2.bias",
                f"{prefix}/layernorm3/gamma": f"{target_prefix}.norm3.weight",
                f"{prefix}/layernorm3/beta": f"{target_prefix}.norm3.bias",
            }
        )

    for raw_key, raw_value in tensorflow_state.items():
        key = normalize_tensorflow_key(raw_key)
        if _skip_key(key):
            skipped_keys.append(raw_key)
            continue
        value = _to_tensor(raw_value)
        if key in direct:
            target_key = direct[key]
            if target_key in target_state:
                _assign(converted, converted_keys, target_state, target_key, value)
            else:
                skipped_keys.append(raw_key)
            continue
        if key in kernel:
            target_key = kernel[key]
            if target_key in target_state:
                _assign(
                    converted,
                    converted_keys,
                    target_state,
                    target_key,
                    tensorflow_kernel_to_linear_weight(value),
                )
            else:
                skipped_keys.append(raw_key)
            continue
        if key == "transformer/expand_layer/expand_layer/kernel":
            target_key = "latent_expander.expand_layer.weight"
            if target_key in target_state:
                weight = tensorflow_kernel_to_linear_weight(value)
                target_length = int(target_state[target_key].shape[0])
                weight = resize_learned_position_embedding(weight, target_length)
                _assign(converted, converted_keys, target_state, target_key, weight)
            continue
        if key == "transformer/expand_layer/expand_layer/bias":
            target_key = "latent_expander.expand_layer.bias"
            if target_key in target_state:
                target_length = int(target_state[target_key].shape[0])
                _assign(
                    converted,
                    converted_keys,
                    target_state,
                    target_key,
                    resize_vector(value, target_length),
                )
            continue
        skipped_keys.append(raw_key)

    for target_key in ("latent_expander.long_weight", "latent_expander.long_bias"):
        if target_key in target_state and target_key not in converted:
            converted[target_key] = torch.zeros_like(target_state[target_key])
            converted_keys.append(target_key)
            initialized_keys.append(target_key)

    missing_target_keys = sorted(key for key in target_state if key not in converted)
    return converted, ConversionReport(
        converted_keys=sorted(converted_keys),
        skipped_keys=sorted(skipped_keys),
        missing_target_keys=missing_target_keys,
        initialized_keys=sorted(initialized_keys),
    )
