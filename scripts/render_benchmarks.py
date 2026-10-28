#!/usr/bin/env python3
"""Regenerate README charts from recorded JSON data; never run a model."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs/assets'
DATA = ROOT / 'benchmarks/data'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.titleweight': 'bold', 'svg.fonttype': 'none',
    'svg.hashsalt': 'imagen-recorded-benchmarks-v1', 'axes.axisbelow': True,
})
GRAY, TEAL, RED = '#64748b', '#087f8c', '#b45309'


def save(fig, name):
    ASSETS.mkdir(parents=True, exist_ok=True)
    destination = ASSETS / f'{name}.svg'
    fig.savefig(destination, bbox_inches='tight', facecolor='white',
                metadata={'Date': None, 'Creator': 'Imagen recorded benchmark renderer'})
    plt.close(fig)
    # Keep generated path-data lines clean in text diffs; newlines still
    # separate SVG path tokens, so this does not alter the figure geometry.
    destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines()) + '\n')


def runtime_chart():
    report = json.loads((DATA / 'runtime-2026-10-28.json').read_text())
    baseline, optimized = report['baseline'], report['optimized']
    duration = [baseline['denoise_seconds'], optimized['denoise_seconds']]
    memory = [baseline['peak_bytes'] / 1e9, optimized['peak_bytes'] / 1e9]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    fig.suptitle('Same-input runtime comparison', fontsize=18, fontweight='bold', y=1.02)
    for axis, values, title, unit, cap in [
        (axes[0], duration, '16-step denoising time', 'seconds', 220),
        (axes[1], memory, 'Peak MLX memory', 'GB (decimal)', 15),
    ]:
        bars = axis.bar(['Stock MFLUX', 'Imagen runtime'], values, color=[GRAY, TEAL], width=0.58)
        axis.set_title(title, pad=15)
        axis.set_ylabel(unit)
        axis.set_ylim(0, cap)
        axis.grid(axis='y', alpha=0.18)
        axis.bar_label(bars, labels=[f'{value:.1f}' if unit == 'seconds' else f'{value:.2f}' for value in values], padding=6, fontsize=13)
    reduction = 100 * (1 - duration[1] / duration[0])
    fig.text(0.5, -0.01, f'{reduction:.1f}% less denoising time | more memory | final latent bitwise identical in this test',
             ha='center', fontsize=11, color='#0f172a')
    fig.text(0.5, -0.065, 'M3 Pro / 36 GB · Q8 · 1024×1024 · seed 7 · n=1 paired run · excludes load, encode, decode and save',
             ha='center', fontsize=9, color=GRAY)
    fig.tight_layout(pad=2)
    save(fig, 'runtime-comparison')


def reference_chart():
    report = json.loads((DATA / 'reference-edit-2026-10-28.json').read_text())
    cases = report['cases']
    fig, axis = plt.subplots(figsize=(10, 4.4))
    labels = ['Full-resolution reference', '512 overview only\nFacial quality rejected by user',
              '512 overview + native detail\nDifferent inputs; quality not quantified']
    bars = axis.barh(labels, [case['seconds'] for case in cases], color=[GRAY, RED, TEAL], height=0.58)
    axis.invert_yaxis()
    axis.set_xlim(0, 670)
    axis.set_xlabel('End-to-end seconds, including encoding, decoding and saving')
    axis.set_title('Reference preparation case study — not an identical-input benchmark', pad=20, fontsize=14)
    axis.grid(axis='x', alpha=0.18)
    axis.bar_label(bars, labels=[f"{case['seconds']:.1f} s" for case in cases], padding=6, fontsize=12)
    fig.text(0.5, -0.025, 'Same optimized runtime in all rows · 512×512 output · 16 steps · seed 7 · one run per strategy',
             ha='center', fontsize=10, color=GRAY)
    fig.text(0.5, -0.075, 'Historical manual-detail experiment. Current automatic-selection generation time is not measured here.',
             ha='center', fontsize=9, color=GRAY)
    fig.tight_layout(pad=1.5)
    save(fig, 'reference-preparation')


if __name__ == '__main__':
    runtime_chart()
    reference_chart()
    print('Updated docs/assets/runtime-comparison.svg and reference-preparation.svg')
