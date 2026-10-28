# Contributing

Start with [architecture](docs/architecture.md) and [development](docs/development.md).

Keep model weights, tokens, input photos, output images and latent tensors out of commits. The `outputs/` directory is ignored. Use synthetic temporary images for reference-policy tests.

Changes to runtime arithmetic need a same-input numeric comparison against the pinned stock MFLUX implementation. A speed result must identify hardware, model/quantization, resolution, steps, seed, measurement boundaries and sample count. Report peak memory and whether inputs or outputs changed. Do not combine denoising-only times with complete generation times.

Reference-selection changes should cover orientation, small images, pixel caps, native crop preservation, budgets, duplicate suppression and source-image numbering. macOS native detector checks require macOS; the model-free tests run on Linux CI.

Regenerate figures with `scripts/render_benchmarks.py` when changing benchmark data. Preserve attribution in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
