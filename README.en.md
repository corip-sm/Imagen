# Imagen

**Local FLUX 3 Image Klein generation and editing for Apple Silicon Macs.**

[한국어](README.md) · [Usage](docs/usage.md) · [Architecture](docs/architecture.md) · [Benchmarks](docs/benchmarks.md)

Imagen wraps the installed MFLUX/MLX engine and an existing Q8 checkpoint. It generates images from detailed English captions, edits reference images, and automatically supplies higher-detail crops of faces, text and salient objects. Each model input and the generated output stays within 512 pixels per edge.

The denoising transformer and scheduler run through an MLX C++ executable; Python handles prompt/reference encoding and image decoding.

## Recorded performance

On an Apple M3 Pro / 36 GB, the same 512×352 edit input took 72.81 seconds before the latest C++ optimizations and 64.60 seconds after them for 16 denoising steps. Each was measured once, excluding model loading, encoding, decoding and saving. The final latent matched the corresponding Python baseline bit for bit. The 50-second target remains unmet.

This is one input on one machine. See the [optimization experiment log](docs/optimization-experiments.md) for conditions, applied changes and rejected attempts.

## Install

Tested: M3 Pro, 36 GB unified memory, macOS 27.0.1, Python 3.12, MFLUX 0.24.0 and MLX/MLX Metal 0.32.4. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Xcode Command Line Tools (`xcode-select --install`) first.

```sh
git clone https://github.com/corip-sm/Imagen.git
cd Imagen
./setup.sh
```

Setup pins the recorded MFLUX commit and MLX versions and builds the C++ denoiser. It does not download weights. If commands are missing, run `uv tool update-shell` and reopen your terminal.

Accept access conditions on the [official model page](https://huggingface.co/black-forest-labs/flux-3-image-klein) and use a token with read access to that repository:

```sh
uv run --no-project --python 3.12 --with huggingface-hub python -c \
  'import getpass; from huggingface_hub import login; login(token=getpass.getpass("HF read token: "), add_to_git_credential=False)'
export IMAGEN_MODEL="$HOME/.cache/imagen/flux3-klein-q8"
mflux-save --model flux3-image-klein --quantize 8 --path "$IMAGEN_MODEL"
```

Weights require several GB; the source-weight cache also needs space during quantization. If you already have a compatible Q8 checkpoint, set `IMAGEN_MODEL` to it instead. The previous local cache location is supported as a fallback.

## Generate and edit

Use a full English scene caption. A separate prompt-expansion model is not included.

```sh
./generate.sh outputs/cafe.png \
  "A realistic photograph of a quiet cafe at dawn. Warm windows illuminate a wet stone sidewalk. A small wooden table holds a white ceramic cup. Eye-level framing, natural colors, detailed materials, no text or watermark."

WIDTH=384 HEIGHT=512 ./generate.sh outputs/edited.png \
  "A realistic photograph of the same room with the dog removed. Reconstruct the hidden bedding naturally and preserve the blue fabric chair, furniture, daylight and camera angle." \
  /path/to/photo.png

SIZE=256 ./generate.sh outputs/small.png "A detailed English scene caption ..."
```

Defaults: 512×512, 16 steps, seed 7. Output dimensions must be multiples of 16. Smaller valid requests are honored; oversized requests are proportionally reduced and aligned down to the model grid. Inputs preserve aspect ratio, use padding for alignment, and are never upscaled. Originals are not modified.

Automatic selection uses local macOS Vision geometry. It crops original pixels **before** reducing them to 512px, removes redundant regions, and limits extra references to **3 crops / 1,536 tokens**, with **10 total references**. Already-small inputs skip analysis. Saliency is a visual heuristic, not a semantic product detector. Editing is generative and may alter surrounding pixels.

Prepared references and local metadata are saved beside the output in `NAME.references/`. They are ignored by Git.

## Configuration

- `IMAGEN_MODEL`: local Q8 checkpoint directory.
- `SIZE`, or `WIDTH` and `HEIGHT`: output size, capped at 512.
- `STEPS=16`, `SEED=7`: defaults.
- `REFERENCE_REGIONS=examples/manual-regions.json`: manual original-pixel crops replace automatic selection. An empty list disables detail crops.
- `IMAGEN_RUNTIME=0`: disable dense weight preparation and CPU/GPU splitting in the C++ denoiser.

See [usage](docs/usage.md), [automatic references](docs/automatic-references.md), [runtime experiments](docs/optimization.md) and [development](docs/development.md).

## Design

```text
src/imagen/references/   # shared limits, local detection, crop selection, CLI dispatch
src/imagen/native/       # C++ denoiser and Python encoding/decoding bridge
scripts/                 # setup, benchmark reproduction, figure rendering
benchmarks/data/         # recorded numeric evidence
```

Large stream weights are reconstructed once and MLP input output-channels are split across CPU FP32 and GPU BF16. Fixed text/reference projections and positional embeddings are reused within a request. Q/K/V weights are packed and the previous Metal fusion kernels are used for large streams. Independent attention and MLP work is submitted together, and the final block computes only image-token outputs while retaining all attention keys and values. Short streams retain Q8 computation. Runtime optimization and input selection are separate mechanisms.

## License

Code: [Apache 2.0](LICENSE), with attribution in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and MLX MIT notices. Model weights are not distributed; their separate license and access conditions are linked from the official model card.
