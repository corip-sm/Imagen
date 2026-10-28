"""Fused FLUX 3 Klein kernels for the project's existing MFLUX checkpoint."""

from __future__ import annotations

import mlx.core as mx


MIN_OPTIMIZED_TOKENS = 1024

_enabled = False


_QK_NORM_ROPE_SOURCE = r"""
        const uint lane = thread_position_in_threadgroup.x % 32;
        const uint row = thread_position_in_grid.x / 32;
        if (row >= BATCH * TOKENS * HEADS) return;
        const uint token = row % TOKENS;
        const uint head = (row / TOKENS) % HEADS;
        const uint batch = row / (TOKENS * HEADS);
        const uint base = (batch * TOKENS + token) * (QKV_PACKED ? 3 : 1) * HEADS * 128 + head * 128;
        const uint key_base = base + (QKV_PACKED ? HEADS * 128 : 0);
        float q[4], k[4];
        float q_square = 0.0f, k_square = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            // MLX 0.32.4 rms_single_row uses four adjacent values per lane.
            const uint d = lane * 4 + i;
            q[i] = float(query_in[base + d]);
            k[i] = float(key_in[key_base + d]);
            q_square += q[i] * q[i];
            k_square += k[i] * k[i];
        }
        const float q_inverse = metal::precise::rsqrt(metal::simd_sum(q_square) / 128.0f + 1e-6f);
        const float k_inverse = metal::precise::rsqrt(metal::simd_sum(k_square) / 128.0f + 1e-6f);
        const uint output_base = row * 128;
        for (uint i = 0; i < 2; ++i) {
            // Upstream RoPE rounds each FP32 product before add/subtract.
            #pragma clang fp contract(off)
            const uint pair = lane * 2 + i;
            const uint d = pair * 2;
            const float qc = cosine[token * 64 + pair];
            const float qs = sine[token * 64 + pair];
            const float q_even = q[2 * i] * q_inverse * float(query_weight[d]);
            const float q_odd = q[2 * i + 1] * q_inverse * float(query_weight[d + 1]);
            const float k_even = k[2 * i] * k_inverse * float(key_weight[d]);
            const float k_odd = k[2 * i + 1] * k_inverse * float(key_weight[d + 1]);
            query[output_base + d] = T(q_even * qc - q_odd * qs);
            query[output_base + d + 1] = T(q_even * qs + q_odd * qc);
            key[output_base + d] = T(k_even * qc - k_odd * qs);
            key[output_base + d + 1] = T(k_even * qs + k_odd * qc);
        }
    """

_qk_norm_rope = mx.fast.metal_kernel(
    name="gaon_flux3_qk_rms_rope",
    input_names=["query_in", "key_in", "query_weight", "key_weight", "cosine", "sine"],
    output_names=["query", "key"],
    source=_QK_NORM_ROPE_SOURCE,
)

_packed_qkv_norm_rope = mx.fast.metal_kernel(
    name="gaon_flux3_packed_qkv_rms_rope",
    input_names=["query_in", "key_in", "query_weight", "key_weight", "cosine", "sine"],
    output_names=["query", "key", "value"],
    source=_QK_NORM_ROPE_SOURCE + r"""
        // Read V from the same packed tensor while writing attention's layout.
        const uint value_base = base + 2 * HEADS * 128;
        for (uint i = 0; i < 4; ++i) {
            const uint d = lane * 4 + i;
            value[output_base + d] = T(float(query_in[value_base + d]) / V_SHRINK);
        }
    """,
)


_silu_mul = mx.fast.metal_kernel(
    name="gaon_flux3_silu_mul",
    input_names=["projected"],
    output_names=["hidden"],
    source=r"""
        const uint i = thread_position_in_grid.x;
        if (i >= ROWS * HIDDEN) return;
        const uint token = i / HIDDEN;
        const uint channel = i % HIDDEN;
        const T gate = projected[token * HIDDEN * 2 + channel];
        const T up = T(float(projected[token * HIDDEN * 2 + HIDDEN + channel]) / SHRINK);
        // Match MLX's typed Sigmoid primitive, including BF16 intermediates.
        const auto y = 1 / (1 + metal::precise::exp(metal::abs(gate)));
        const T sigmoid = (gate < 0) ? y : 1 - y;
        const T activated = gate * sigmoid;
        hidden[i] = activated * up;
    """,
)

_modulate = mx.fast.metal_kernel(
    name="gaon_flux3_modulate",
    input_names=["normalized", "shift", "scale"],
    output_names=["out"],
    source=r"""
        #pragma clang fp contract(off)
        const uint index = thread_position_in_grid.x;
        if (index >= SIZE) return;
        const uint column = index % 3072;
        const uint row = index / 3072;
        const uint batch = row / TOKENS;
        const uint token = row % TOKENS;
        const uint shift_index = ((SHIFT_BATCHES == 1 ? 0 : batch) * SHIFT_TOKENS
            + (SHIFT_TOKENS == 1 ? 0 : token)) * 3072 + column;
        const uint scale_index = ((SCALE_BATCHES == 1 ? 0 : batch) * SCALE_TOKENS
            + (SCALE_TOKENS == 1 ? 0 : token)) * 3072 + column;
        const T factor = T(1.0f + float(scale[scale_index]));
        const T scaled = T(float(factor) * float(normalized[index]));
        out[index] = T(float(scaled) + float(shift[shift_index]));
    """,
)

_residual_gate = mx.fast.metal_kernel(
    name="gaon_flux3_residual_gate",
    input_names=["x", "attention", "feed_forward", "gate"],
    output_names=["out"],
    source=r"""
        const uint i = thread_position_in_grid.x;
        if (i >= SIZE) return;
        const uint column = i % 3072;
        const uint row = i / 3072;
        const uint batch = row / TOKENS;
        const uint token = row % TOKENS;
        const uint gate_index = ((GATE_BATCHES == 1 ? 0 : batch) * GATE_TOKENS
            + (GATE_TOKENS == 1 ? 0 : token)) * 3072 + column;
        const T joined = T(float(attention[i]) + float(feed_forward[i]));
        const T gated = T(float(joined) * float(gate[gate_index]));
        out[i] = T(float(x[i]) + float(gated));
    """,
)


def qk_norm_rope(query, key, query_weight, key_weight, cosine, sine, *, heads: int, head_dim: int):
    batch, tokens, inner = query.shape
    if head_dim != 128 or inner != heads * head_dim or key.shape != query.shape:
        raise ValueError("Fused FLUX 3 Q/K kernel expects matching projections with 128 channels per head")
    shape = (batch, heads, tokens, head_dim)
    outputs = _qk_norm_rope(
        inputs=[query, key, query_weight, key_weight, cosine, sine],
        template=[("T", query.dtype), ("BATCH", batch), ("TOKENS", tokens), ("HEADS", heads), ("QKV_PACKED", False)],
        grid=(batch * tokens * heads * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape, shape],
        output_dtypes=[query.dtype, key.dtype],
    )
    return outputs[0], outputs[1]


def packed_qkv_norm_rope(projected, query_weight, key_weight, cosine, sine, *, heads: int, head_dim: int, value_shrink: int = 1):
    """Normalize and rotate packed Q/K, and permute V without split copies."""
    batch, tokens, packed_inner = projected.shape
    if head_dim != 128 or packed_inner != 3 * heads * head_dim:
        raise ValueError("Packed FLUX 3 Q/K/V kernel expects three projections with 128 channels per head")
    if value_shrink not in (1, 4):
        raise ValueError("Packed FLUX 3 V headroom must be 1 or 4")
    shape = (batch, heads, tokens, head_dim)
    return tuple(_packed_qkv_norm_rope(
        # Both arguments share the original contiguous allocation. Passing the
        # same tensor twice does not create the strided views made by mx.split.
        inputs=[projected, projected, query_weight, key_weight, cosine, sine],
        template=[
            ("T", projected.dtype), ("BATCH", batch), ("TOKENS", tokens),
            ("HEADS", heads), ("QKV_PACKED", True), ("V_SHRINK", value_shrink),
        ],
        grid=(batch * tokens * heads * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape, shape, shape],
        output_dtypes=[projected.dtype, projected.dtype, projected.dtype],
    ))


def silu_mul(projected, *, shrink: float):
    doubled_hidden = projected.shape[-1]
    if doubled_hidden % 2:
        raise ValueError("Fused SwiGLU expects equally sized gate and up projections")
    rows = projected.size // doubled_hidden
    hidden = doubled_hidden // 2
    output_shape = (*projected.shape[:-1], hidden)
    return _silu_mul(
        inputs=[projected],
        template=[("T", projected.dtype), ("ROWS", rows), ("HIDDEN", hidden), ("SHRINK", int(shrink))],
        grid=(rows * hidden, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[projected.dtype],
    )[0]


def _modulation_layout(x, value):
    """Keep broadcast modulation compact without dropping token or batch values."""
    if x.ndim not in (2, 3) or x.shape[-1] != 3072:
        raise ValueError("Fused FLUX 3 modulation expects [tokens,3072] or [batch,tokens,3072]")
    if value.ndim not in (1, 2, 3) or value.shape[-1] != 3072:
        raise ValueError("Fused FLUX 3 modulation expects 3072 modulation channels")
    batch, tokens = (1, x.shape[0]) if x.ndim == 2 else x.shape[:2]
    shape = (1,) * (3 - value.ndim) + value.shape
    if shape[0] not in (1, batch) or shape[1] not in (1, tokens):
        raise ValueError(f"Modulation shape {value.shape} cannot broadcast to {x.shape}")
    return value.reshape(shape), shape[0], shape[1], tokens


def layer_norm_modulate(x, shift, scale):
    shift, shift_batches, shift_tokens, tokens = _modulation_layout(x, shift)
    scale, scale_batches, scale_tokens, _ = _modulation_layout(x, scale)
    # Retain MLX's native reduction order so normalization matches the model.
    normalized = mx.fast.layer_norm(x, weight=None, bias=None, eps=1e-6)
    return _modulate(
        inputs=[normalized, shift, scale],
        template=[
            ("T", x.dtype), ("TOKENS", tokens), ("SIZE", x.size),
            ("SHIFT_BATCHES", shift_batches), ("SHIFT_TOKENS", shift_tokens),
            ("SCALE_BATCHES", scale_batches), ("SCALE_TOKENS", scale_tokens),
        ],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def residual_gate(x, attention, feed_forward, gate):
    gate, gate_batches, gate_tokens, tokens = _modulation_layout(x, gate)
    if attention.shape != x.shape or feed_forward.shape != x.shape:
        raise ValueError("Fused FLUX 3 residual inputs must have matching shapes")
    return _residual_gate(
        inputs=[x, attention, feed_forward, gate],
        template=[
            ("T", x.dtype), ("SIZE", x.size), ("TOKENS", tokens),
            ("GATE_BATCHES", gate_batches), ("GATE_TOKENS", gate_tokens),
        ],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def _merge_qkv_projections(module):
    """Pack Q/K/V projection rows into one exact-layout MLX linear operation."""
    if hasattr(module, "to_qkv"):
        return

    from mlx import nn

    projections = (module.to_q, module.to_k, module.to_v)
    first = projections[0]
    if all(isinstance(layer, nn.QuantizedLinear) for layer in projections):
        if any(
            (layer.group_size, layer.bits, layer.mode) != (first.group_size, first.bits, first.mode)
            for layer in projections[1:]
        ):
            raise ValueError("Q/K/V quantization layouts differ; cannot merge projections safely")
        if any(layer.weight.shape[1] != first.weight.shape[1] for layer in projections[1:]):
            raise ValueError("Q/K/V input dimensions differ; cannot merge projections safely")
        if any((layer.biases is None) != (first.biases is None) for layer in projections[1:]):
            raise ValueError("Q/K/V quantization biases differ; cannot merge projections safely")
        # Construct an existing-weights layer without allocating random weights or
        # consuming the random stream. Concatenations stay lazy until inference.
        merged = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
        nn.Module.__init__(merged)
        merged.group_size, merged.bits, merged.mode = first.group_size, first.bits, first.mode
        merged.weight = mx.concatenate([layer.weight for layer in projections], axis=0)
        merged.scales = mx.concatenate([layer.scales for layer in projections], axis=0)
        merged.biases = (
            None if first.biases is None
            else mx.concatenate([layer.biases for layer in projections], axis=0)
        )
        if any("bias" in layer for layer in projections):
            merged.bias = mx.concatenate([
                layer.bias if "bias" in layer else mx.zeros((layer.weight.shape[0],), dtype=first.scales.dtype)
                for layer in projections
            ])
        merged.freeze()
    elif all(isinstance(layer, nn.Linear) for layer in projections):
        if any(layer.weight.shape[1] != first.weight.shape[1] for layer in projections[1:]):
            raise ValueError("Q/K/V input dimensions differ; cannot merge projections safely")
        merged = nn.Linear.__new__(nn.Linear)
        nn.Module.__init__(merged)
        merged.weight = mx.concatenate([layer.weight for layer in projections], axis=0)
        if any("bias" in layer for layer in projections):
            merged.bias = mx.concatenate([
                layer.bias if "bias" in layer else mx.zeros((layer.weight.shape[0],), dtype=first.weight.dtype)
                for layer in projections
            ])
    else:
        raise TypeError("Unsupported Q/K/V projection layout")

    module.to_qkv = merged
    del module.to_q
    del module.to_k
    del module.to_v


def enable() -> None:
    """Patch only FLUX 3 Klein's repeated activation hot paths."""
    global _enabled
    if _enabled:
        return

    from mflux.models.flux3.model.flux3_transformer.flux3_attention import Flux3Attention
    from mflux.models.flux3.model.flux3_transformer.flux3_block import Flux3Block
    from mflux.models.flux3.model.flux3_transformer.flux3_feed_forward import Flux3FeedForward

    # Capture function objects before mutating the classes. Calling the original
    # block alone would still dispatch its children through the patched classes.
    original_attention = Flux3Attention.__call__
    original_feed_forward = Flux3FeedForward.__call__
    original_block = Flux3Block.__call__

    def attention_call(self, x, rotary_emb):
        precision = self.compute_precision
        stream_dtype = x.dtype
        x_compute = precision.to_compute(x)
        batch, length, _ = x_compute.shape
        _merge_qkv_projections(self)
        projected = self.to_qkv(x_compute)
        cosine, sine = rotary_emb
        query, key, value = packed_qkv_norm_rope(
            projected,
            self.norm_q.weight,
            self.norm_k.weight,
            cosine,
            sine,
            heads=self.heads,
            head_dim=self.dim_head,
            value_shrink=int(Flux3Attention.FLOAT16_HEADROOM) if precision.dtype is not None else 1,
        )
        attended = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=1.0 / (self.dim_head**0.5)
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, self.heads * self.dim_head)
        return precision.to_stream(
            self.to_out(attended), stream_dtype, Flux3Attention.FLOAT16_HEADROOM
        )

    def feed_forward_call(self, x):
        precision = self.compute_precision
        stream_dtype = x.dtype
        projected = self.linear_in(precision.to_compute(x))
        shrink = Flux3FeedForward.FLOAT16_HEADROOM if precision.dtype is not None else 1.0
        hidden = silu_mul(projected, shrink=shrink)
        return precision.to_stream(
            self.linear_out(hidden), stream_dtype, Flux3FeedForward.FLOAT16_HEADROOM
        )

    def block_call(self, x, rotary_emb, mod):
        shift, scale, gate = mod
        if x.shape[-2] < MIN_OPTIMIZED_TOKENS and hasattr(self.attn, "to_q"):
            # Short streams select different upstream GEMM/attention kernels.
            # Preserve their original operations and child dispatch exactly.
            normalized = (1 + scale) * self.norm(x) + shift
            attention = original_attention(self.attn, normalized, rotary_emb)
            feed_forward = original_feed_forward(self.ff, normalized)
            return x + gate * (attention + feed_forward)
        normalized = layer_norm_modulate(x, shift, scale)
        attention = self.attn(normalized, rotary_emb)
        feed_forward = self.ff(normalized)
        return residual_gate(x, attention, feed_forward, gate)

    # Keep explicit immutable source function references for introspection.
    attention_call.__wrapped__ = original_attention
    feed_forward_call.__wrapped__ = original_feed_forward
    block_call.__wrapped__ = original_block
    Flux3Attention.__call__ = attention_call
    Flux3FeedForward.__call__ = feed_forward_call
    Flux3Block.__call__ = block_call
    _enabled = True
