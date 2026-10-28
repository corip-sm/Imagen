#pragma once
// Metal arithmetic ported from the previous Imagen runtime; see THIRD_PARTY_NOTICES.md.
namespace imagen::metal_source {
inline constexpr const char *qkv = R"IMAGEN(
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

        // Read V from the same packed tensor while writing attention's layout.
        const uint value_base = base + 2 * HEADS * 128;
        for (uint i = 0; i < 4; ++i) {
            const uint d = lane * 4 + i;
            value[output_base + d] = T(float(query_in[value_base + d]) / V_SHRINK);
        }
    )IMAGEN";
inline constexpr const char *silu_mul = R"IMAGEN(
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
    )IMAGEN";
inline constexpr const char *modulate = R"IMAGEN(
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
    )IMAGEN";
inline constexpr const char *residual_gate = R"IMAGEN(
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
    )IMAGEN";
} // namespace imagen::metal_source
