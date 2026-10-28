"""Prepare inputs and decode results for the native FLUX.3 denoiser."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import subprocess
import tempfile

import mlx.core as mx
from PIL import Image
from safetensors import safe_open

from mflux.models.common.config.config import Config
from mflux.models.flux3.latent_creator.flux3_latent_creator import Flux3LatentCreator
from mflux.models.flux3.latent_creator.flux3_reference_images import Flux3ReferenceImages
from mflux.models.flux3.variants import Flux3Klein, Flux3KleinEdit
from mflux.utils.image_util import ImageUtil

ROOT = Path(__file__).resolve().parents[3]
BINARY = ROOT / '.cache/imagen-native/imagen-native'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--steps', required=True, type=int)
    parser.add_argument('--width', required=True, type=int)
    parser.add_argument('--height', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--image-paths', nargs='+', type=Path, default=[])
    args = parser.parse_args()
    if not BINARY.is_file():
        raise SystemExit('Native denoiser is not built. Run ./scripts/build-native.sh')
    model = (Flux3KleinEdit if args.image_paths else Flux3Klein)(model_path=str(args.model))
    if model.bits != 8 or not model.transformer.flow_map:
        raise SystemExit('The native denoiser requires the FLUX.3 Klein Q8 distilled checkpoint.')

    inputs = {}
    canvas_width, canvas_height = args.width, args.height
    crop_box = None
    if args.image_paths:
        with_images = []
        for path in args.image_paths:
            with Image.open(path) as image:
                with_images.append(image.copy())
        references, box = Flux3ReferenceImages.prepare(with_images)
        canvas_width, canvas_height, crop_box = Flux3ReferenceImages.resolve_canvas(
            references, box, args.width, args.height
        )
        inputs['cond_tokens'], inputs['cond_ids'] = model.encode_references(references)
    config = Config(model_config=model.model_config, num_inference_steps=args.steps,
                    height=canvas_height, width=canvas_width, guidance=None)
    inputs['text_tokens'], inputs['text_ids'] = model._encode_prompt(args.prompt)
    height, width = Flux3LatentCreator.latent_shape(canvas_height, canvas_width)
    inputs['latents'] = Flux3LatentCreator.create_noise(args.seed, canvas_height, canvas_width)
    inputs['latent_ids'] = Flux3LatentCreator.prepare_ids(height, width)
    mx.eval(*inputs.values())

    # The C++ worker reads the same checkpoint. Release Python copies before it starts.
    model.transformer = None
    model.text_encoder = None
    gc.collect()
    mx.clear_cache()
    with tempfile.TemporaryDirectory(prefix='imagen-native-') as temp_dir:
        temp = Path(temp_dir)
        input_path, output_path = temp / 'input.safetensors', temp / 'output.safetensors'
        mx.save_safetensors(str(input_path), inputs)
        environment = os.environ.copy()
        if environment.get('IMAGEN_RUNTIME', '1') == '1':
            environment['IMAGEN_NATIVE_OPT'] = '1'
        else:
            environment.pop('IMAGEN_NATIVE_OPT', None)
        environment['IMAGEN_NATIVE_CACHE'] = '1'
        subprocess.run([str(BINARY), str(args.model / 'transformer'), str(input_path),
                        str(output_path), str(args.steps)], env=environment, check=True)
        with safe_open(str(output_path), framework='np') as result:
            seconds = float(result.metadata()['denoise_seconds'])
        latents = mx.load(str(output_path))['latents']
        mx.eval(latents)

    decoded = model._decode_latents(latents, height, width)
    generated = ImageUtil.to_image(decoded_latents=decoded, config=config, seed=args.seed,
                                   prompt=args.prompt, quantization=model.bits,
                                   generation_time=seconds, image_paths=args.image_paths or None,
                                   generation_parameters=model._generation_parameters())
    if crop_box is not None:
        generated.image = generated.image.crop(crop_box)
        generated.width, generated.height = generated.image.size
    generated.save(path=args.output)


if __name__ == '__main__':
    main()
