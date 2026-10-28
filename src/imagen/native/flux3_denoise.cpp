#include "metal_sources.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mlx/mlx.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mx = mlx::core;
using mx::array;
using Map = std::unordered_map<std::string, array>;

array silu(const array &x) { return x * mx::sigmoid(x); }
array part(const array &a, int axis, int start, int end) {
  auto lo = mx::Shape(a.ndim(), 0), hi = a.shape();
  lo[axis] = start;
  hi[axis] = end;
  return mx::slice(a, lo, hi);
}
// Use the same rounding-preserving fused Metal kernels as the Python runtime.
namespace {
using Kernel = mx::fast::CustomKernelFunction;
using Args = std::vector<std::pair<std::string, mx::fast::TemplateArg>>;

const Kernel &qkv_kernel() {
  static const auto kernel = mx::fast::metal_kernel(
      "imagen_native_packed_qkv_rms_rope",
      {"query_in", "key_in", "query_weight", "key_weight", "cosine", "sine"},
      {"query", "key", "value"}, imagen::metal_source::qkv);
  return kernel;
}
const Kernel &silu_kernel() {
  static const auto kernel =
      mx::fast::metal_kernel("imagen_native_silu_mul", {"projected"},
                             {"hidden"}, imagen::metal_source::silu_mul);
  return kernel;
}
const Kernel &modulate_kernel() {
  static const auto kernel = mx::fast::metal_kernel(
      "imagen_native_modulate", {"normalized", "shift", "scale"}, {"out"},
      imagen::metal_source::modulate);
  return kernel;
}
const Kernel &residual_kernel() {
  static const auto kernel = mx::fast::metal_kernel(
      "imagen_native_residual_gate", {"x", "attention", "feed_forward", "gate"},
      {"out"}, imagen::metal_source::residual_gate);
  return kernel;
}
array fused_modulation(const array &x, const array &shift, const array &scale) {
  const int tokens = x.shape(1), size = static_cast<int>(x.size());
  auto normalized = mx::fast::layer_norm(x, {}, {}, 1e-6f);
  Args args = {{"T", x.dtype()},
               {"TOKENS", tokens},
               {"SIZE", size},
               {"SHIFT_BATCHES", shift.shape(0)},
               {"SHIFT_TOKENS", shift.shape(1)},
               {"SCALE_BATCHES", scale.shape(0)},
               {"SCALE_TOKENS", scale.shape(1)}};
  return modulate_kernel()({normalized, shift, scale}, {x.shape()}, {x.dtype()},
                           {size, 1, 1}, {256, 1, 1}, args, std::nullopt, false,
                           {})[0];
}
std::vector<array> fused_qkv(const array &packed, const array &qw,
                             const array &kw, const array &cosine,
                             const array &sine) {
  const int b = packed.shape(0), tokens = packed.shape(1), heads = 24;
  mx::Shape shape = {b, heads, tokens, 128};
  Args args = {{"T", packed.dtype()}, {"BATCH", b},         {"TOKENS", tokens},
               {"HEADS", heads},      {"QKV_PACKED", true}, {"V_SHRINK", 1}};
  return qkv_kernel()({packed, packed, qw, kw, cosine, sine},
                      {shape, shape, shape},
                      {packed.dtype(), packed.dtype(), packed.dtype()},
                      {b * tokens * heads * 32, 1, 1}, {256, 1, 1}, args,
                      std::nullopt, false, {});
}
array fused_silu(const array &projected) {
  int hidden = projected.shape(-1) / 2,
      rows = static_cast<int>(projected.size()) / (2 * hidden);
  auto shape = projected.shape();
  shape.back() = hidden;
  Args args = {{"T", projected.dtype()},
               {"ROWS", rows},
               {"HIDDEN", hidden},
               {"SHRINK", 1}};
  return silu_kernel()({projected}, {shape}, {projected.dtype()},
                       {rows * hidden, 1, 1}, {256, 1, 1}, args, std::nullopt,
                       false, {})[0];
}
array fused_residual(const array &x, const array &attn, const array &ff,
                     const array &gate) {
  int size = static_cast<int>(x.size()), tokens = x.shape(1);
  Args args = {{"T", x.dtype()},
               {"SIZE", size},
               {"TOKENS", tokens},
               {"GATE_BATCHES", gate.shape(0)},
               {"GATE_TOKENS", gate.shape(1)}};
  return residual_kernel()({x, attn, ff, gate}, {x.shape()}, {x.dtype()},
                           {size, 1, 1}, {256, 1, 1}, args, std::nullopt, false,
                           {})[0];
}
} // namespace

struct Model {
  Map w, dense, cpu_weights, gpu_weights;
  std::optional<array> static_text, static_cond;
  std::optional<std::pair<array, array>> image_rope, cond_rope, text_rope,
      joint_rope;
  explicit Model(const std::string &dir) {
    for (const auto &item : std::filesystem::directory_iterator(dir)) {
      if (item.path().extension() != ".safetensors")
        continue;
      auto loaded = mx::load_safetensors(item.path().string()).first;
      for (auto &[k, v] : loaded) {
        if (!w.emplace(k, v).second)
          throw std::runtime_error("duplicate weight: " + k);
      }
    }
    if (w.empty())
      throw std::runtime_error("no transformer shards");
  }
  void prepare_one(const std::string &name, bool hybrid,
                   std::vector<array> &pending) {
    auto weight = w.at(name + ".weight"), scales = w.at(name + ".scales"),
         biases = w.at(name + ".biases");
    auto unpack = [&](int start, int end) {
      return mx::dequantize(part(weight, 0, start, end),
                            part(scales, 0, start, end),
                            part(biases, 0, start, end), 64, 8, "affine");
    };
    int n = weight.shape(0);
    if (hybrid) {
      int split = (int(n * 0.45) / 64) * 64;
      auto a = mx::astype(unpack(0, split), mx::float32);
      auto b = unpack(split, n);
      cpu_weights.emplace(name, a);
      gpu_weights.emplace(name, b);
      pending.push_back(a);
      pending.push_back(b);
    } else {
      auto a = unpack(0, n);
      dense.emplace(name, a);
      pending.push_back(a);
    }
  }
  void prepare(const Map &in) {
    int image = in.at("latents").shape(1), text = in.at("text_tokens").shape(1);
    int cond = in.count("cond_tokens") ? in.at("cond_tokens").shape(1) : 0;
    std::vector<std::string> blocks;
    if (image >= 1024)
      for (int i = 0; i < 5; i++)
        blocks.push_back("mode_blocks.image." + std::to_string(i));
    if (cond >= 1024)
      for (int i = 0; i < 5; i++)
        blocks.push_back("mode_blocks.image_cond." + std::to_string(i));
    if (image + text + cond >= 1024)
      for (int i = 0; i < 28; i++)
        blocks.push_back("single_transformer_blocks." + std::to_string(i));
    std::vector<array> pending;
    std::vector<std::string> names;
    for (const auto &b : blocks) {
      std::vector<array> weights, scales, biases;
      for (const auto &suffix : {".attn.to_q", ".attn.to_k", ".attn.to_v"}) {
        auto name = b + suffix;
        weights.push_back(w.at(name + ".weight"));
        scales.push_back(w.at(name + ".scales"));
        biases.push_back(w.at(name + ".biases"));
        names.push_back(name);
      }
      auto packed = mx::dequantize(mx::concatenate(weights, 0),
                                   mx::concatenate(scales, 0),
                                   mx::concatenate(biases, 0), 64, 8, "affine");
      dense.emplace(b + ".attn.to_qkv", packed);
      pending.push_back(packed);
      for (const auto &suffix :
           {".attn.to_out", ".ff.linear_in", ".ff.linear_out"}) {
        auto name = b + suffix;
        prepare_one(name, std::string(suffix) == ".ff.linear_in", pending);
        names.push_back(name);
      }
    }
    // Lazy dequantization retains its inputs; discard the redundant map
    // references before evaluation so buffers can be released promptly.
    for (const auto &name : names)
      for (const auto &suffix : {".weight", ".scales", ".biases"})
        w.erase(name + suffix);
    mx::eval(pending);
    pending.clear();
    mx::clear_cache();
    std::cerr << "prepared_blocks=" << blocks.size() << "\n";
  }
  array linear(const std::string &name, const array &x) const {
    if (auto it = cpu_weights.find(name); it != cpu_weights.end()) {
      auto cpu = mx::Device(mx::Device::cpu);
      auto gpu = mx::Device(mx::Device::gpu);
      auto ci = mx::astype(x, mx::float32, cpu);
      auto cy =
          mx::astype(mx::matmul(ci, mx::matrix_transpose(it->second, cpu), cpu),
                     mx::bfloat16, cpu);
      auto gy = mx::matmul(x, mx::matrix_transpose(gpu_weights.at(name)), gpu);
      return mx::concatenate({cy, gy}, -1, gpu);
    }
    if (auto it = dense.find(name); it != dense.end())
      return mx::matmul(x, mx::matrix_transpose(it->second));
    const auto &weight = w.at(name + ".weight");
    auto si = w.find(name + ".scales"), bi = w.find(name + ".biases");
    array y = si != w.end()
                  ? mx::quantized_matmul(x, weight, si->second,
                                         bi == w.end()
                                             ? std::optional<array>{}
                                             : std::optional<array>{bi->second},
                                         true, 64, 8, "affine")
                  : mx::matmul(x, mx::matrix_transpose(weight));
    auto bias = w.find(name + ".bias");
    return bias == w.end() ? y : y + bias->second;
  }
  array time_embedding(const std::string &name, const array &t) const {
    auto tbf = mx::astype(t, mx::bfloat16) * array(1000.0f, mx::bfloat16);
    auto freq =
        mx::exp((-std::log(10000.0f) / 128.0f) * mx::arange(128, mx::float32));
    auto args = mx::expand_dims(mx::astype(tbf, mx::float32), -1) *
                mx::expand_dims(freq, 0);
    auto embed = mx::astype(mx::concatenate({mx::cos(args), mx::sin(args)}, -1),
                            mx::bfloat16);
    return linear(name + ".linear_2", silu(linear(name + ".linear_1", embed)));
  }
  std::vector<array> modulation(const std::string &prefix,
                                const array &vec) const {
    auto x = mx::expand_dims(linear(prefix + ".linear", silu(vec)), 1);
    return mx::split(x, 3, -1);
  }
  using Rope = std::pair<array, array>;
  Rope rope(const array &original) const {
    auto ids = mx::astype(original, mx::float32);
    ids.eval();
    const float *data = ids.data<float>();
    int n = ids.shape(0);
    std::vector<float> cs(n * 64), sn(n * 64);
    for (int i = 0; i < n; i++)
      for (int axis = 0; axis < 4; axis++)
        for (int j = 0; j < 16; j++) {
          double angle = static_cast<double>(data[i * 4 + axis]) /
                         std::pow(10000.0, (2.0 * j) / 32.0);
          cs[i * 64 + axis * 16 + j] = static_cast<float>(std::cos(angle));
          sn[i * 64 + axis * 16 + j] = static_cast<float>(std::sin(angle));
        }
    return {array(cs.data(), {n, 64}, mx::float32),
            array(sn.data(), {n, 64}, mx::float32)};
  }
  array norm_rope(const array &x, const array &weight, const Rope &r) const {
    auto v = mx::fast::rms_norm(mx::astype(x, mx::float32),
                                mx::astype(weight, mx::float32), 1e-6f);
    auto p = mx::reshape(v, {x.shape(0), x.shape(1), x.shape(2), 64, 2});
    auto x0 = mx::squeeze(part(p, 4, 0, 1), 4);
    auto x1 = mx::squeeze(part(p, 4, 1, 2), 4);
    auto out = mx::stack(
        {r.first * x0 - r.second * x1, r.second * x0 + r.first * x1}, -1);
    return mx::astype(mx::reshape(out, x.shape()), x.dtype());
  }
  array block(const std::string &p, const array &x, const Rope &r,
              const std::vector<array> &mod, int output_tokens = 0) const {
    const bool fused = dense.count(p + ".attn.to_qkv") != 0;
    // Weakly typed Python scalars retain BF16 intermediate rounding.
    auto z = fused ? fused_modulation(x, mod[0], mod[1])
                   : (array(1.0f, mod[1].dtype()) + mod[1]) *
                             mx::fast::layer_norm(x, {}, {}, 1e-6f) +
                         mod[0];
    int b = x.shape(0), n = x.shape(1);
    auto split_heads = [&](const array &a) {
      return mx::transpose(mx::reshape(a, {b, n, 24, 128}), {0, 2, 1, 3});
    };
    std::vector<array> qkv;
    if (fused) {
      auto packed = linear(p + ".attn.to_qkv", z);
      qkv = fused_qkv(packed, w.at(p + ".attn.norm_q.weight"),
                      w.at(p + ".attn.norm_k.weight"), r.first, r.second);
    } else {
      auto q = norm_rope(split_heads(linear(p + ".attn.to_q", z)),
                         w.at(p + ".attn.norm_q.weight"), r);
      auto k = norm_rope(split_heads(linear(p + ".attn.to_k", z)),
                         w.at(p + ".attn.norm_k.weight"), r);
      qkv = {q, k, split_heads(linear(p + ".attn.to_v", z))};
    }
    const int used_tokens = output_tokens > 0 ? output_tokens : n;
    auto query = used_tokens < n ? part(qkv[0], 2, 0, used_tokens) : qkv[0];
    auto attn = mx::fast::scaled_dot_product_attention(
        query, qkv[1], qkv[2], 1.0f / std::sqrt(128.0f));
    attn = mx::reshape(mx::transpose(attn, {0, 2, 1, 3}),
                       {b, used_tokens, 3072});
    attn = linear(p + ".attn.to_out", attn);
    auto used_z = used_tokens < n ? part(z, 1, 0, used_tokens) : z;
    auto used_x = used_tokens < n ? part(x, 1, 0, used_tokens) : x;
    auto used_gate = mod[2].shape(1) == n && used_tokens < n
                         ? part(mod[2], 1, 0, used_tokens)
                         : mod[2];
    auto ff_in = linear(p + ".ff.linear_in", used_z);
    // Submit the independent branches together so the CPU projection can
    // overlap GPU attention before the feed-forward result is consumed.
    if (fused)
      mx::async_eval(attn, ff_in);
    array hidden = [&]() {
      if (fused)
        return fused_silu(ff_in);
      auto ff = mx::split(ff_in, 2, -1);
      return silu(ff[0]) * ff[1];
    }();
    auto ffout = linear(p + ".ff.linear_out", hidden);
    return fused ? fused_residual(used_x, attn, ffout, used_gate)
                 : used_x + used_gate * (attn + ffout);
  }
  // These inputs and positions do not change during one denoising request.
  void prepare_constants(const Map &in) {
    static_text = linear("context_embedder",
                         mx::astype(in.at("text_tokens"), mx::bfloat16));
    image_rope = rope(in.at("latent_ids"));
    text_rope = rope(in.at("text_ids"));
    std::vector<array> ids = {in.at("latent_ids")};
    std::vector<array> pending = {*static_text};
    if (in.count("cond_tokens")) {
      static_cond = linear("x_embedders.image_cond",
                           mx::astype(in.at("cond_tokens"), mx::bfloat16));
      cond_rope = rope(in.at("cond_ids"));
      ids.push_back(in.at("cond_ids"));
      pending.push_back(*static_cond);
    }
    ids.push_back(in.at("text_ids"));
    joint_rope = rope(mx::concatenate(ids, 0));
    mx::eval(pending);
  }
  array forward(const array &latent, const array &image_ids, const array &text,
                const array &text_ids, const array &t, const array &target,
                const std::optional<array> &cond,
                const std::optional<array> &cond_ids) const {
    auto vimage = time_embedding("timestep_embedder", t);
    auto vcontext = time_embedding("timestep_embedder", mx::zeros_like(t));
    auto tv = time_embedding("target_timestep_embedder", target);
    vimage = vimage + tv;
    vcontext = vcontext + tv;
    std::vector<std::string> names = {"image"};
    std::vector<array> streams = {
        linear("x_embedders.image", mx::astype(latent, mx::bfloat16))};
    std::vector<array> ids = {image_ids};
    std::vector<array> vectors = {vimage};
    if (cond) {
      names.push_back("image_cond");
      streams.push_back(static_cond ? *static_cond
                                    : linear("x_embedders.image_cond",
                                             mx::astype(*cond, mx::bfloat16)));
      ids.push_back(*cond_ids);
      vectors.push_back(vcontext);
    }
    names.push_back("txt");
    streams.push_back(static_text ? *static_text
                                  : linear("context_embedder",
                                           mx::astype(text, mx::bfloat16)));
    ids.push_back(text_ids);
    vectors.push_back(vcontext);
    std::vector<Rope> ropes;
    std::vector<std::vector<array>> early, single;
    for (size_t i = 0; i < names.size(); i++) {
      ropes.push_back(names[i] == "image" && image_rope       ? *image_rope
                      : names[i] == "image_cond" && cond_rope ? *cond_rope
                      : names[i] == "txt" && text_rope        ? *text_rope
                                                              : rope(ids[i]));
      early.push_back(
          modulation("early_stream_modulations." + names[i], vectors[i]));
      single.push_back(
          modulation("single_stream_modulations." + names[i], vectors[i]));
    }
    for (int i = 0; i < 5; i++)
      for (size_t j = 0; j < names.size(); j++) {
        auto p = names[j] == "txt" ? "txt_mode_blocks."
                                   : "mode_blocks." + names[j] + ".";
        streams[j] =
            block(p + std::to_string(i), streams[j], ropes[j], early[j]);
      }
    auto joint = mx::concatenate(streams, 1);
    auto jr = joint_rope ? *joint_rope : rope(mx::concatenate(ids, 0));
    std::vector<array> jm;
    for (int m = 0; m < 3; m++) {
      std::vector<array> parts;
      for (size_t j = 0; j < streams.size(); j++)
        parts.push_back(mx::broadcast_to(
            single[j][m], {joint.shape(0), streams[j].shape(1), 3072}));
      jm.push_back(mx::concatenate(parts, 1));
    }
    // Only image tokens leave the final joint block. Its keys and values still
    // cover every stream, while unused output queries and MLP rows are omitted.
    for (int i = 0; i < 28; i++)
      joint = block("single_transformer_blocks." + std::to_string(i), joint, jr,
                    jm, i == 27 ? latent.shape(1) : 0);
    auto out = part(joint, 1, 0, latent.shape(1));
    auto n = linear("norm_out.linear", mx::astype(silu(vimage), mx::bfloat16));
    auto scale = mx::expand_dims(part(n, 1, 0, 3072), 1),
         shift = mx::expand_dims(part(n, 1, 3072, 6144), 1);
    out = mx::fast::layer_norm(out, {}, {}, 1e-6f) *
              (array(1.0f, scale.dtype()) + scale) +
          shift;
    return linear("proj_out", out);
  }
};
int main(int argc, char **argv) {
  try {
    if (argc != 5)
      throw std::runtime_error("usage: imagen-native TRANSFORMER_DIR "
                               "INPUT.safetensors OUTPUT.safetensors STEPS");
    int steps = std::stoi(argv[4]);
    if (steps < 1)
      throw std::runtime_error("steps must be positive");
    Model model(argv[1]);
    auto in = mx::load_safetensors(argv[2]).first;
    auto latents = in.at("latents");
    auto schedule = mx::linspace(1.0, 0.0, steps + 1, mx::float32);
    schedule = 6.93f * schedule / (1.0f + (6.93f - 1.0f) * schedule);
    schedule.eval();
    std::optional<array> cond, cond_ids;
    if (in.count("cond_tokens")) {
      cond = in.at("cond_tokens");
      cond_ids = in.at("cond_ids");
    }
    mx::reset_peak_memory();
    auto started = std::chrono::steady_clock::now();
    if (std::getenv("IMAGEN_NATIVE_OPT"))
      model.prepare(in);
    if (std::getenv("IMAGEN_NATIVE_CACHE"))
      model.prepare_constants(in);
    for (int i = 0; i < steps; i++) {
      auto step_started = std::chrono::steady_clock::now();
      auto t = part(schedule, 0, i, i + 1),
           target = part(schedule, 0, i + 1, i + 2);
      auto velocity = model.forward(latents, in.at("latent_ids"),
                                    in.at("text_tokens"), in.at("text_ids"), t,
                                    target, cond, cond_ids);
      auto dt = part(schedule, 0, i + 1, i + 2) - part(schedule, 0, i, i + 1);
      dt.eval();
      latents = latents + array(dt.item<float>(), velocity.dtype()) * velocity;
      latents.eval();
      auto step_seconds = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - step_started)
                              .count();
      std::cerr << "native " << (i + 1) << "/" << steps << " " << step_seconds
                << "s\n";
    }
    auto elapsed = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - started)
                       .count();
    auto peak = mx::get_peak_memory();
    mx::save_safetensors(argv[3], {{"latents", latents}},
                         {{"denoise_seconds", std::to_string(elapsed)},
                          {"peak_bytes", std::to_string(peak)}});
    std::cerr << "denoise_seconds=" << elapsed << " peak_bytes=" << peak
              << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
