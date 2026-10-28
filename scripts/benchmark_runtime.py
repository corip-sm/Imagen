#!/usr/bin/env python3
"""Compare native and project runtimes using identical numeric inputs, without decoding images.

Run with ./scripts/python.sh scripts/benchmark_runtime.py --model /path/to/q8.
This historical 1024px numeric diagnostic bypasses the normal 512px image policy.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = (
    'A cinematic documentary photograph of a rain-soaked city hall at dusk, '
    'empty steps reflecting red and blue emergency lights, realistic news '
    'photography, detailed stone architecture, subdued colors, natural film grain.'
)


def worker(args):
    from importlib.metadata import version
    import mlx.core as mx
    from mflux.models.flux3.latent_creator.flux3_latent_creator import Flux3LatentCreator
    from mflux.models.flux3.model.flux3_scheduler.flux3_flow_map_scheduler import Flux3FlowMapScheduler
    from mflux.models.flux3.variants import Flux3Klein
    if args.worker == 'optimized':
        sys.path.insert(0, str(ROOT / 'src/imagen/runtime'))
        from gaon_mflux_kernels import enable as enable_kernels
        from gaon_mflux_runtime import enable as enable_runtime
        enable_kernels()
        enable_runtime()
    model = Flux3Klein(model_path=str(args.model))
    input_file = args.output_dir / 'inputs.npz'
    if args.worker == 'prepare':
        text, text_ids = model._encode_prompt(args.prompt)
        latents = Flux3LatentCreator.create_noise(args.seed, args.size, args.size)
        height, width = Flux3LatentCreator.latent_shape(args.size, args.size)
        ids = Flux3LatentCreator.prepare_ids(height, width)
        mx.eval(text, text_ids, latents, ids)
        mx.savez(str(input_file), latents=latents, latent_ids=ids, text_tokens=text, text_ids=text_ids)
        return
    model.text_encoder = None
    model.vae = None
    gc.collect()
    mx.set_cache_limit(1_000_000_000)
    mx.clear_cache()
    inputs = mx.load(str(input_file))
    text = inputs['text_tokens'].astype(mx.bfloat16)
    latents = inputs['latents']
    scheduler = Flux3FlowMapScheduler(num_inference_steps=args.steps)
    mx.eval(inputs, text, scheduler.timesteps)
    mx.reset_peak_memory()
    records = []
    started = time.perf_counter()
    for step in range(args.steps):
        step_start = time.perf_counter()
        velocity = model.transformer(
            hidden_states=latents, img_ids=inputs['latent_ids'],
            encoder_hidden_states=text, txt_ids=inputs['text_ids'],
            timestep=scheduler.timesteps[step:step + 1],
            target_timestep=scheduler.timesteps[step + 1:step + 2],
        )
        latents = scheduler.step(latents=latents, velocity=velocity, step=step)
        mx.eval(latents)
        records.append(time.perf_counter() - step_start)
        print(f'{args.worker}: {step + 1}/{args.steps} ({records[-1]:.2f}s)', flush=True)
    report = {'mode': args.worker, 'size': args.size, 'steps': args.steps, 'seed': args.seed,
              'device': mx.device_info(), 'software': {'mflux': version('mflux'), 'mlx': version('mlx')},
              'denoise_seconds': time.perf_counter() - started, 'step_seconds': records,
              'peak_bytes': mx.get_peak_memory(), 'preparation': getattr(model.transformer, '_gaon_preparation_metrics', [])}
    mx.savez(str(args.output_dir / f'{args.worker}-latents.npz'), latents=latents)
    (args.output_dir / f'{args.worker}.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/runtime-benchmark')
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--prompt', default=DEFAULT_PROMPT)
    parser.add_argument('--worker', choices=('prepare', 'baseline', 'optimized'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.model.is_dir() or args.size < 16 or args.size % 16 or args.steps < 1:
        parser.error('A local checkpoint, grid-aligned size >= 16, and positive step count are required')
    args.model = args.model.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment.update(GAON_MFLUX_FUSED_KERNELS='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for mode in ('prepare', 'baseline', 'optimized'):
        command = [sys.executable, '-I', str(Path(__file__).resolve()), '--model', str(args.model),
                   '--output-dir', str(args.output_dir), '--size', str(args.size), '--steps', str(args.steps),
                   '--seed', str(args.seed), '--prompt', args.prompt, '--worker', mode]
        subprocess.run(command, env=environment, check=True)
    import numpy as np
    def read_latents(mode):
        with np.load(args.output_dir / f'{mode}-latents.npz', allow_pickle=False) as data:
            # MLX saves the evolving state in FP32; do not reinterpret BF16 payloads.
            return data['latents'].astype(np.float32)
    baseline, optimized = read_latents('baseline'), read_latents('optimized')
    result = {'baseline': json.loads((args.output_dir / 'baseline.json').read_text()),
              'optimized': json.loads((args.output_dir / 'optimized.json').read_text()),
              'final_latents_bitwise_equal': bool(np.array_equal(baseline.view(np.uint32), optimized.view(np.uint32))),
              'images_decoded': False,
              'scope': 'Denoising only; first lazy weight preparation included. Excludes model loading, encoding, decoding, saving.'}
    (args.output_dir / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
