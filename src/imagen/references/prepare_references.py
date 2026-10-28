"""Normalize all reference images to a 512-pixel maximum without upscaling.

Input pixels are processed automatically; no image is displayed or visually
inspected. Original files are never modified. The model receives RGB PNG copies
with EXIF orientation applied and padding to its 16-pixel latent grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from PIL import Image, ImageOps

from reference_policy import GRID, MAX_EDGE, MAX_REFERENCES


def output_size(width: int, height: int) -> tuple[int, int]:
    """Honor smaller grid-aligned sizes; proportionally cap oversized requests."""
    if width < GRID or height < GRID:
        raise ValueError(f"Output width and height must be at least {GRID}")
    if max(width, height) <= MAX_EDGE:
        if width % GRID or height % GRID:
            raise ValueError(f"Output width and height must be multiples of {GRID}")
        return width, height
    scale = MAX_EDGE / max(width, height)
    return tuple(max(GRID, int(side * scale) // GRID * GRID) for side in (width, height))


def load_regions(path: Path | None) -> list[dict]:
    if path is None:
        return []
    regions = json.loads(path.read_text())
    if not isinstance(regions, list):
        raise ValueError("The regions file must contain a JSON list")
    return regions


def prepare(sources: list[Path], output_dir: Path, regions: list[dict] | None = None,
            *, automatic: bool = False) -> dict:
    started = time.perf_counter()
    if automatic and regions:
        raise ValueError("Choose automatic selection or manual regions, not both")
    regions = regions or []
    if not 1 <= len(sources) <= MAX_REFERENCES:
        raise ValueError(f"Supply between 1 and {MAX_REFERENCES} reference images")
    sources = [source.resolve(strict=True) for source in sources]
    automatic_report = {"enabled": False}
    if automatic:
        from auto_regions import find_regions
        regions, automatic_report = find_regions(sources, MAX_REFERENCES - len(sources))
    if len(sources) + len(regions) > MAX_REFERENCES:
        raise ValueError(f"Overviews and detail crops together must not exceed {MAX_REFERENCES} references")
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError("Each detail region must be an object with source and box")
        source_index, box = region.get("source"), region.get("box")
        if type(source_index) is not int or not 1 <= source_index <= len(sources):
            raise ValueError("Region source must be a 1-based input image index")
        if not isinstance(box, list) or len(box) != 4 or any(type(value) is not int for value in box):
            raise ValueError("Region box must contain four integer pixel coordinates")
        if not (0 <= box[0] < box[2] and 0 <= box[1] < box[3]):
            raise ValueError("Region box must be [left, top, right, bottom] with positive area")
    output_dir.mkdir(parents=True, exist_ok=True)
    items = [(index, None, {}) for index in range(1, len(sources) + 1)]
    items.extend((region["source"], region["box"], region) for region in regions)
    destinations = [output_dir / f"reference-{index:02d}.png" for index in range(1, len(items) + 1)]
    # Refuse before any write if a prepared image would overwrite any source.
    if any(destination.resolve() in sources for destination in [*destinations, output_dir / "references.json"]):
        raise ValueError("Prepared output would overwrite an input image; choose a different output path")
    records = []
    for (source_index, crop_box, metadata), destination in zip(items, destinations):
        source = sources[source_index - 1]
        with Image.open(source) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        original_size = image.size
        if crop_box is not None:
            if crop_box[2] > image.width or crop_box[3] > image.height:
                raise ValueError(f"Region box exceeds oriented input size {image.size}")
            # Select the original detail before applying the shared 512px cap.
            image = image.crop(tuple(crop_box))
        selected_size = image.size
        image.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
        content_size = image.size
        padded_size = tuple((side + GRID - 1) // GRID * GRID for side in content_size)
        if padded_size != content_size:
            padded = Image.new("RGB", padded_size)
            padded.paste(image, (0, 0))
            image = padded
        assert max(image.size) <= MAX_EDGE
        image.save(destination)
        records.append({
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_size": list(original_size),
            "source_reference": source_index,
            "role": "detail" if crop_box is not None else "overview",
            "selection": ("automatic" if automatic else "manual") if crop_box is not None else "overview",
            "detected_kind": metadata.get("kind") if automatic else None,
            "crop_box": crop_box,
            "selected_size": list(selected_size),
            "content_size": list(content_size),
            "prepared_size": list(image.size),
            "resized": content_size != selected_size,
            "reference_tokens": image.width * image.height // (GRID * GRID),
            "path": str(destination.resolve()),
        })
    report = {
        "schema_version": 4,
        "max_input_edge": MAX_EDGE,
        "upscale_inputs": False,
        "records": records,
        "references": [record["path"] for record in records],
        "prepared_reference_tokens": sum(record["reference_tokens"] for record in records),
        "automatic_selection": automatic_report,
        "preparation_seconds": time.perf_counter() - started,
    }
    if automatic:
        detail_tokens = sum(record["reference_tokens"] for record in records if record["role"] == "detail")
        automatic_report["prepared_detail_tokens"] = detail_tokens
        if detail_tokens > automatic_report.get("detail_token_budget", 0) and regions:
            raise ValueError("Automatic detail references exceeded their computation budget")
    (output_dir / "references.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--regions-file", type=Path)
    args = parser.parse_args()
    try:
        report = prepare(args.sources, args.output_dir, load_regions(args.regions_file),
                         automatic=args.regions_file is None)
        print(json.dumps(report, indent=2))
    except (ValueError, OSError) as error:
        parser.exit(2, f"Reference preparation failed: {error}\n")


if __name__ == "__main__":
    main()
