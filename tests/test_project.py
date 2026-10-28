"""Model-free regressions for public size, reference, and launcher behavior."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/imagen/references'))
from auto_regions import DETAIL_TOKEN_BUDGET, select_regions
import launch_generation
from prepare_references import output_size, prepare


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / 'input.png'
        image = Image.new('RGB', (1600, 900), '#437082')
        draw = ImageDraw.Draw(image)
        for x in range(0, 1600, 20):
            draw.rectangle((x, 0, x + 9, 899), fill=(x % 251, 80, 150))
        image.save(self.source)

    def tearDown(self):
        self.temporary.cleanup()

    def test_output_cap_and_small_request(self):
        self.assertEqual(output_size(256, 256), (256, 256))
        self.assertEqual(output_size(384, 256), (384, 256))
        self.assertEqual(output_size(1024, 768), (512, 384))
        for size in ((300, 300), (0, 256), (16, -1)):
            with self.assertRaises(ValueError):
                output_size(*size)

    def test_large_input_resize_and_original_unchanged(self):
        before = self.source.read_bytes()
        report = prepare([self.source], self.root / 'prepared')
        self.assertEqual(report['records'][0]['content_size'], [512, 288])
        self.assertEqual(self.source.read_bytes(), before)

    def test_small_input_is_padded_without_resampling(self):
        source = self.root / 'small.png'
        with Image.open(self.source) as image:
            small = image.crop((50, 50, 223, 151))
            small.save(source)
        report = prepare([source], self.root / 'small-prepared')
        self.assertEqual(report['records'][0]['content_size'], [173, 101])
        with Image.open(report['references'][0]) as result:
            self.assertEqual(result.size, (176, 112))
            self.assertEqual(result.crop((0, 0, 173, 101)).tobytes(), small.tobytes())

    def test_detail_is_selected_before_downscaling(self):
        box = [415, 73, 774, 474]
        report = prepare([self.source], self.root / 'detail', [{'source': 1, 'box': box}])
        record = report['records'][1]
        self.assertEqual(record['source_reference'], 1)
        self.assertFalse(record['resized'])
        with Image.open(self.source) as original, Image.open(record['path']) as result:
            expected = original.crop(tuple(box))
            self.assertEqual(result.crop((0, 0, expected.width, expected.height)).tobytes(), expected.tobytes())
        self.assertTrue(all(max(item['prepared_size']) <= 512 for item in report['records']))

    def test_invalid_reference_and_region_rejected(self):
        for sources, regions in [([], None), ([self.source] * 11, None),
                                 ([self.source], [{'source': 2, 'box': [0, 0, 20, 20]}]),
                                 ([self.source], [{'source': 1, 'box': [0, 0, 2000, 100]}])]:
            with self.assertRaises(ValueError):
                prepare(sources, self.root / 'invalid', regions)

    def test_exif_orientation_applied(self):
        source = self.root / 'oriented.jpg'
        exif = Image.Exif()
        exif[274] = 6
        Image.new('RGB', (59, 27), 'blue').save(source, exif=exif)
        report = prepare([source], self.root / 'oriented')
        self.assertEqual(report['records'][0]['content_size'], [27, 59])

    def test_automatic_selection_budget_and_deduplication(self):
        image = {'source': 1, 'size': [4000, 4000], 'candidates': [
            {'kind': 'object', 'confidence': 0.9, 'box': [0.05, 0.05, 0.30, 0.30]},
            {'kind': 'object', 'confidence': 0.8, 'box': [0.05, 0.05, 0.30, 0.30]},
            {'kind': 'object', 'confidence': 0.9, 'box': [0.60, 0.60, 0.85, 0.85]},
        ]}
        selected, limits = select_regions([image], 9)
        self.assertEqual(len(selected), 1)
        self.assertLessEqual(limits['selected_estimated_tokens'], DETAIL_TOKEN_BUDGET)
        self.assertEqual(select_regions([image], 0)[0], [])

    def test_automatic_source_indices_and_pixel_limit(self):
        small = self.root / 'small.png'
        Image.new('RGB', (128, 128)).save(small)
        native = {'images': [{'source': 1, 'size': [1600, 900], 'errors': [], 'candidates': [
            {'kind': 'text', 'confidence': 1.0, 'box': [0.1, 0.2, 0.3, 0.3]},
        ]}]}
        completed = subprocess.CompletedProcess([], 0, json.dumps(native), '')
        with patch('auto_regions.subprocess.run', return_value=completed):
            report = prepare([small, self.source], self.root / 'automatic', automatic=True)
        details = [item for item in report['records'] if item['role'] == 'detail']
        self.assertTrue(details)
        self.assertTrue(all(item['source_reference'] == 2 for item in details))
        self.assertTrue(all(max(item['prepared_size']) <= 512 for item in report['records']))

    def test_total_detector_failure_stops(self):
        native = {'images': [{'size': [1600, 900], 'candidates': [], 'errors': [
            {'detector': name, 'error': 'injected'} for name in ('face', 'text', 'objectness', 'attention')]}]}
        with patch('auto_regions.subprocess.run', return_value=subprocess.CompletedProcess([], 0, json.dumps(native), '')):
            with self.assertRaises(ValueError):
                prepare([self.source], self.root / 'failure', automatic=True)

    def test_launcher_preserves_edit_instruction_and_runtime_toggle(self):
        regions = self.root / 'regions.json'
        regions.write_text('[{"source":1,"box":[100,100,400,400]}]')
        args = ['launch', '--output', str(self.root / 'result.png'), '--prompt', 'Remove the label.',
                '--model', str(self.root), '--width', '256', '--height', '256', str(self.source)]
        with patch.dict(os.environ, {'REFERENCE_REGIONS': str(regions), 'IMAGEN_RUNTIME': '0'}), \
             patch.object(sys, 'argv', args), patch('launch_generation.os.execvpe') as execute, \
             contextlib.redirect_stderr(io.StringIO()):
            launch_generation.main()
        cli, command, environment = execute.call_args.args
        self.assertEqual(cli, 'mflux-generate-flux3-edit')
        self.assertTrue(command[command.index('--prompt') + 1].endswith('Remove the label.'))
        self.assertEqual(command[command.index('--width') + 1], '256')
        self.assertEqual(environment['GAON_MFLUX_FUSED_KERNELS'], '0')
        self.assertTrue(environment['PYTHONPATH'].split(os.pathsep)[0].endswith('src/imagen/runtime'))
        self.assertFalse((self.root / 'result.png').exists())


class PublicationTests(unittest.TestCase):
    def test_measured_runtime_sources_are_unchanged(self):
        report = json.loads((ROOT / 'benchmarks/data/runtime-2026-10-28.json').read_text())
        for old_path, expected in report['source_sha256'].items():
            if old_path.startswith('runtime_optimization/') and old_path.endswith('.py'):
                current = ROOT / 'src/imagen/runtime' / Path(old_path).name
                self.assertEqual(hashlib.sha256(current.read_bytes()).hexdigest(), expected)

    def test_documentation_links_exist(self):
        files = list(ROOT.glob('README*.md')) + list((ROOT / 'docs').glob('*.md'))
        for document in files:
            for target in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)\)', document.read_text()):
                if '://' in target or target.startswith(('#', 'mailto:')):
                    continue
                target = target.split('#')[0]
                self.assertTrue((document.parent / target).exists(), f'{document.name}: {target}')


if __name__ == '__main__':
    unittest.main()
