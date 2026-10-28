"""Project CLI execution of the existing FLUX 3 Q8 weights.

No checkpoint is written and no new model is loaded. Preparation replaces only
selected block linears, then evaluates their new weights together. The in-memory
layout is for generation through this project, not checkpoint export. Prepared
streams retain their layout for the model lifetime; shrinking an already
prepared stream is outside this project's fixed-shape CLI execution path.
"""

from __future__ import annotations

import gc
from functools import wraps
import math
import time

import mlx.core as mx
from mlx import nn

from gaon_mflux_kernels import MIN_OPTIMIZED_TOKENS


_CPU_FRACTION = 0.45


class DenseLinear(nn.Linear):
    """An existing dense weight without random initialization or packed copies."""

    def __init__(self, weight: mx.array, bias: mx.array | None = None):
        nn.Module.__init__(self)
        self.weight = weight
        if bias is not None:
            self.bias = bias
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        # Keep the optional bias addition separate, as QuantizedLinear does.
        result = mx.matmul(x, self.weight.T)
        return result + self.bias if "bias" in self else result


class HybridLinear(nn.Module):
    """Disjoint output channels on CPU FP32 and GPU BF16, joined in order.

    Each CPU output computes the full dot product. There is no split reduction
    and no extra sum across devices. CPU weights are FP32 representations of the
    reconstructed BF16 values, not a second quantization of the checkpoint.
    """

    def __init__(
        self,
        cpu_weight: mx.array,
        gpu_weight: mx.array | None,
        bias: mx.array | None = None,
    ):
        super().__init__()
        if cpu_weight.dtype != mx.float32 or cpu_weight.ndim != 2:
            raise ValueError("Hybrid CPU weight must be a float32 matrix")
        if gpu_weight is not None and (
            gpu_weight.dtype != mx.bfloat16
            or gpu_weight.ndim != 2
            or gpu_weight.shape[1] != cpu_weight.shape[1]
        ):
            raise ValueError("Hybrid GPU weight must be a matching bfloat16 matrix")
        self.cpu_weight = cpu_weight
        if gpu_weight is not None:
            self.gpu_weight = gpu_weight
        if bias is not None:
            self.bias = bias
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        if x.dtype != mx.bfloat16:
            raise ValueError("HybridLinear requires bfloat16 input; configure precision before preparation")
        cpu_input = x.astype(mx.float32, stream=mx.cpu)
        cpu_result = mx.matmul(cpu_input, self.cpu_weight.T, stream=mx.cpu)
        cpu_result = cpu_result.astype(mx.bfloat16, stream=mx.cpu)
        if "gpu_weight" in self:
            gpu_result = mx.matmul(x, self.gpu_weight.T, stream=mx.gpu)
            result = mx.concatenate((cpu_result, gpu_result), axis=-1, stream=mx.gpu)
        else:
            result = cpu_result
        return result + self.bias if "bias" in self else result

    def _extra_repr(self) -> str:
        gpu_rows = self.gpu_weight.shape[0] if "gpu_weight" in self else 0
        return (
            f"input_dims={self.cpu_weight.shape[1]}, "
            f"cpu_rows={self.cpu_weight.shape[0]}, gpu_rows={gpu_rows}, "
            f"bias={'bias' in self}"
        )


def _validate_fraction(cpu_fraction: float) -> None:
    if not math.isfinite(cpu_fraction) or not 0.0 <= cpu_fraction <= 1.0:
        raise ValueError("cpu_fraction must be finite and between 0 and 1")


def _cpu_rows(output_dims: int, fraction: float) -> int:
    return output_dims if fraction == 1.0 else int(output_dims * fraction) // 64 * 64


def _unpack(layer: nn.QuantizedLinear, start: int = 0, end: int | None = None) -> mx.array:
    affine_biases = layer.get("biases")
    weight = mx.dequantize(
        layer.weight[start:end],
        scales=layer.scales[start:end],
        biases=affine_biases[start:end] if affine_biases is not None else None,
        group_size=layer.group_size,
        bits=layer.bits,
        mode=layer.mode,
    )
    return weight


def _attention_names(block):
    return ("to_qkv", "to_out") if hasattr(block.attn, "to_qkv") else (
        "to_q", "to_k", "to_v", "to_out"
    )


def _fallback_reason(block):
    for component in (block.attn, block.ff):
        precision = getattr(component, "compute_precision", None)
        if precision is not None and precision.dtype is not None:
            return "compute_precision"
    layers = [getattr(block.attn, name) for name in _attention_names(block)]
    layers.extend((block.ff.linear_in, block.ff.linear_out))
    for layer in layers:
        if isinstance(layer, (DenseLinear, HybridLinear)):
            continue
        if not isinstance(layer, nn.QuantizedLinear):
            return "non_quantized_layout"
        if layer.bits != 8 or layer.mode != "affine":
            return "non_q8_affine_layout"
        if layer.scales.dtype != mx.bfloat16:
            return "non_bfloat16_scales"
        if layer.get("biases") is not None and layer.biases.dtype != mx.bfloat16:
            return "non_bfloat16_quantization_biases"
    return None


def _replacement(layer: nn.Module, fraction: float):
    if isinstance(layer, HybridLinear):
        output_dims = layer.cpu_weight.shape[0]
        if "gpu_weight" in layer:
            output_dims += layer.gpu_weight.shape[0]
        if _cpu_rows(output_dims, fraction) != layer.cpu_weight.shape[0]:
            raise ValueError("Reload the original block before changing its hybrid split")
        return None
    if isinstance(layer, DenseLinear):
        if _cpu_rows(layer.weight.shape[0], fraction):
            raise ValueError("Reload the original block before enabling a hybrid split")
        return None
    if not isinstance(layer, nn.QuantizedLinear):
        # Unquantized layers remain unchanged: this path only reconstructs Q8.
        return None

    output_dims = layer.weight.shape[0]
    cpu_rows = _cpu_rows(output_dims, fraction)
    bias = layer.get("bias")
    packed_bytes = sum(
        value.nbytes for key in ("weight", "scales", "biases")
        if (value := layer.get(key)) is not None
    )
    if cpu_rows:
        # Dequantize the partitions independently. Keeping a slice of a full
        # dense matrix would unnecessarily retain its whole backing buffer.
        cpu_weight = _unpack(layer, 0, cpu_rows).astype(mx.float32)
        gpu_weight = _unpack(layer, cpu_rows) if cpu_rows < output_dims else None
        replacement = HybridLinear(cpu_weight, gpu_weight, bias)
        buffers = [cpu_weight] + ([gpu_weight] if gpu_weight is not None else [])
    else:
        weight = _unpack(layer)
        replacement = DenseLinear(weight, bias)
        buffers = [weight]
    metrics = {
        "linear_count": 1,
        "hybrid_count": int(cpu_rows > 0),
        "cpu_rows": cpu_rows,
        "weight_elements": sum(array.size for array in buffers),
        "packed_bytes_replaced": packed_bytes,
        "prepared_weight_bytes": sum(array.nbytes for array in buffers),
    }
    if bias is not None:
        buffers.append(bias)
    return replacement, buffers, metrics


def _prepare(blocks, cpu_fraction: float) -> dict:
    from gaon_mflux_kernels import _merge_qkv_projections

    _validate_fraction(cpu_fraction)
    started = time.perf_counter()
    active_before = mx.get_active_memory()
    changes = []
    pending = []
    totals = {
        "block_count": len(blocks),
        "linear_count": 0,
        "hybrid_count": 0,
        "cpu_rows": 0,
        "weight_elements": 0,
        "packed_bytes_replaced": 0,
        "prepared_weight_bytes": 0,
        "requested_cpu_fraction": cpu_fraction,
        "fallback_blocks": {},
    }
    for block in blocks:
        reason = _fallback_reason(block)
        if reason is not None:
            totals["fallback_blocks"][reason] = totals["fallback_blocks"].get(reason, 0) + 1
            continue
        # Pack while the original rows are still quantized; dense preparation
        # then allocates the combined Q/K/V weight only once.
        _merge_qkv_projections(block.attn)
        targets = [(block.attn, name, 0.0) for name in _attention_names(block)]
        targets.extend(((block.ff, "linear_in", cpu_fraction), (block.ff, "linear_out", 0.0)))
        for owner, name, fraction in targets:
            converted = _replacement(getattr(owner, name), fraction)
            if converted is None:
                continue
            replacement, buffers, metrics = converted
            changes.append((owner, name, replacement))
            pending.extend(buffers)
            for key, value in metrics.items():
                totals[key] += value

    # Validate/build every replacement before changing the model. Store no old
    # modules or packed arrays beyond the temporary dequantization graphs.
    for owner, name, replacement in changes:
        setattr(owner, name, replacement)
    built = time.perf_counter()
    if pending:
        mx.eval(*pending)
    evaluated = time.perf_counter()
    pending.clear()
    changes.clear()
    gc.collect()
    mx.clear_cache()
    totals.update(
        build_seconds=built - started,
        eval_seconds=evaluated - built,
        prepare_seconds=time.perf_counter() - started,
        active_bytes_before=active_before,
        active_bytes_after=mx.get_active_memory(),
        peak_bytes=mx.get_peak_memory(),
    )
    return totals


def enable() -> None:
    """Prepare sufficiently large streams once after initialization and encoding."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux3.model.flux3_transformer.flux3_transformer import Flux3Transformer

    original = Flux3Transformer.__call__
    if getattr(original, "_gaon_runtime_enabled", False):
        return

    @wraps(original)
    def transformer_call(self, *args, **kwargs):
        prepared = getattr(self, "_gaon_prepared_modes", None)
        if prepared is None:
            prepared = set()
            # Keep bookkeeping outside Module's parameter dictionary.
            object.__setattr__(self, "_gaon_prepared_modes", prepared)
            object.__setattr__(self, "_gaon_preparation_metrics", [])

        condition = kwargs.get("cond_hidden_states", args[6] if len(args) > 6 else None)
        has_condition = condition is not None and condition.shape[1] > 0
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        text = kwargs.get("encoder_hidden_states", args[2] if len(args) > 2 else None)
        image_tokens = hidden.shape[1] if hidden is not None else 0
        text_tokens = text.shape[1] if text is not None else 0
        condition_tokens = condition.shape[1] if has_condition else 0
        modes = []
        blocks = []
        if image_tokens >= MIN_OPTIMIZED_TOKENS and "image" not in prepared:
            modes.append("image")
            blocks.extend(self.mode_blocks["image"])
        if image_tokens + text_tokens + condition_tokens >= MIN_OPTIMIZED_TOKENS and "joint" not in prepared:
            modes.append("joint")
            blocks.extend(self.single_transformer_blocks)
        if condition_tokens >= MIN_OPTIMIZED_TOKENS and "image_cond" not in prepared:
            modes.append("image_cond")
            blocks.extend(self.mode_blocks["image_cond"])

        if blocks:
            if ModelConfig.precision == mx.bfloat16:
                metrics = _prepare(blocks, _CPU_FRACTION)
            else:
                metrics = {
                    "block_count": len(blocks),
                    "linear_count": 0,
                    "fallback_blocks": {"non_bfloat16_stream": len(blocks)},
                }
            metrics["modes"] = modes
            self._gaon_preparation_metrics.append(metrics)
            prepared.update(modes)
        return original(self, *args, **kwargs)

    transformer_call._gaon_runtime_enabled = True
    Flux3Transformer.__call__ = transformer_call
