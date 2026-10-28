"""Select native-detail crops using local Vision geometry and a fixed budget."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import time

from PIL import Image

from reference_policy import DETAIL_TOKEN_BUDGET, GRID, MAX_AUTO_DETAILS, MAX_EDGE


def _area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _overlap(first: list[int], second: list[int]) -> tuple[float, float]:
    intersection = _area([max(first[0], second[0]), max(first[1], second[1]),
                          min(first[2], second[2]), min(first[3], second[3])])
    union = _area(first) + _area(second) - intersection
    smaller = min(_area(first), _area(second))
    return intersection / max(1, union), intersection / max(1, smaller)


def _pixel_box(values, width: int, height: int) -> list[int] | None:
    if not isinstance(values, list) or len(values) != 4:
        return None
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        return None
    left, top, right, bottom = values
    box = [max(0, math.floor(left * width)), max(0, math.floor(top * height)),
           min(width, math.ceil(right * width)), min(height, math.ceil(bottom * height))]
    return box if box[2] - box[0] >= 4 and box[3] - box[1] >= 4 else None


def _merge_text(candidates: list[dict]) -> list[dict]:
    """Join nearby lines/words into compact label regions, not whole pages."""
    others = [item for item in candidates if item["kind"] != "text"]
    groups = []
    for item in sorted((item for item in candidates if item["kind"] == "text"),
                       key=lambda item: (item["pixel_box"][1], item["pixel_box"][0])):
        box = item["pixel_box"]
        for group in groups:
            previous = group["pixel_box"]
            height = min(box[3] - box[1], previous[3] - previous[1])
            horizontal = min(box[2], previous[2]) - max(box[0], previous[0])
            vertical = min(box[3], previous[3]) - max(box[1], previous[1])
            horizontal_gap = max(0, max(box[0], previous[0]) - min(box[2], previous[2]))
            vertical_gap = max(0, max(box[1], previous[1]) - min(box[3], previous[3]))
            same_line = vertical >= 0.5 * height and horizontal_gap <= 2 * height
            nearby_line = horizontal >= 0.55 * min(box[2] - box[0], previous[2] - previous[0]) and vertical_gap <= 0.65 * height
            merged = [min(box[0], previous[0]), min(box[1], previous[1]),
                      max(box[2], previous[2]), max(box[3], previous[3])]
            if (same_line or nearby_line) and merged[2] - merged[0] <= 1024 and merged[3] - merged[1] <= 512:
                group["pixel_box"] = merged
                group["confidence"] = max(group["confidence"], item["confidence"])
                break
        else:
            groups.append(dict(item))
    return others + groups


def select_regions(images: list[dict], slots: int) -> tuple[list[dict], dict]:
    """Rank information gain, remove duplicates, and enforce one global budget."""
    ranked = []
    for image in images:
        width, height = image["size"]
        if max(width, height) <= MAX_EDGE:
            continue
        candidates = []
        for item in image["candidates"]:
            kind = item.get("kind")
            if kind not in ("face", "text", "object"):
                continue
            confidence = float(item.get("confidence", 0))
            if not math.isfinite(confidence) or confidence < {"face": 0.5, "text": 0.35, "object": 0.1}[kind]:
                continue
            box = _pixel_box(item.get("box"), width, height)
            if box is not None:
                candidates.append({**item, "confidence": min(1, confidence), "pixel_box": box})
        for item in _merge_text(candidates):
            kind, box = item["kind"], item["pixel_box"]
            crop_width, crop_height = box[2] - box[0], box[3] - box[1]
            margins = {"face": (0.35, 0.60, 0.35, 0.30),
                       "text": (0.10, 0.35, 0.10, 0.35),
                       "object": (0.06, 0.06, 0.06, 0.06)}[kind]
            box = [max(0, math.floor(box[0] - margins[0] * crop_width)),
                   max(0, math.floor(box[1] - margins[1] * crop_height)),
                   min(width, math.ceil(box[2] + margins[2] * crop_width)),
                   min(height, math.ceil(box[3] + margins[3] * crop_height))]
            crop_width, crop_height = box[2] - box[0], box[3] - box[1]
            scale = min(1.0, MAX_EDGE / max(crop_width, crop_height))
            gain = scale / (MAX_EDGE / max(width, height))
            if gain < 1.25:
                continue
            # Ceil is conservative relative to Pillow's thumbnail rounding.
            tokens = math.ceil(crop_width * scale / GRID) * math.ceil(crop_height * scale / GRID)
            priority = {"face": 3.0, "text": 2.4, "object": 1.5}[kind]
            score = priority * item["confidence"] * math.log2(gain) / (1 + tokens / 1024)
            ranked.append({"source": image["source"], "box": box, "kind": kind,
                           "confidence": item["confidence"], "score": score,
                           "zoom_gain": gain, "estimated_tokens": tokens})

    selected = []
    used_tokens = 0
    while ranked and len(selected) < min(MAX_AUTO_DETAILS, max(0, slots)):
        def diversity_score(item):
            same_kind = sum(other["kind"] == item["kind"] for other in selected)
            same_source = sum(other["source"] == item["source"] for other in selected)
            return item["score"] / (1 + 0.8 * same_kind + 0.5 * same_source)
        ranked.sort(key=lambda item: (-diversity_score(item), item["source"], item["box"]))
        item = ranked.pop(0)
        if used_tokens + item["estimated_tokens"] > DETAIL_TOKEN_BUDGET:
            continue
        duplicate = False
        for other in selected:
            if item["source"] != other["source"]:
                continue
            iou, containment = _overlap(item["box"], other["box"])
            area_ratio = max(_area(item["box"]), _area(other["box"])) / min(_area(item["box"]), _area(other["box"]))
            if iou >= 0.5 or (containment >= 0.85 and area_ratio < 1.8):
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(item)
        used_tokens += item["estimated_tokens"]
    return selected, {"max_details": MAX_AUTO_DETAILS, "detail_token_budget": DETAIL_TOKEN_BUDGET,
                      "selected_estimated_tokens": used_tokens}


def find_regions(sources: list[Path], slots: int) -> tuple[list[dict], dict]:
    started = time.perf_counter()
    targets = []
    for index, source in enumerate(sources, start=1):
        with Image.open(source) as image:
            if max(image.size) > MAX_EDGE:
                targets.append((index, source))
    report = {"enabled": True, "max_details": MAX_AUTO_DETAILS,
              "detail_token_budget": DETAIL_TOKEN_BUDGET, "selected": [],
              "selected_estimated_tokens": 0, "analysis": [], "warnings": []}
    if not targets or slots <= 0:
        report.update(reason="inputs_already_small_or_no_free_slots", seconds=time.perf_counter() - started)
        return [], report
    helper = Path(__file__).with_name("detect_regions.swift")
    try:
        result = subprocess.run(["xcrun", "swift", str(helper), *[str(path) for _, path in targets]],
                                check=True, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        detail = error.stderr[-2000:] if isinstance(error, subprocess.CalledProcessError) else str(error)
        raise ValueError(f"Automatic region detection failed before generation: {detail}") from error
    detected = json.loads(result.stdout)
    if len(detected.get("images", [])) != len(targets):
        raise ValueError("Automatic detector returned an unexpected image count")
    for image, (source_index, _) in zip(detected["images"], targets):
        image["source"] = source_index
        errors = image.get("errors", [])
        if len(errors) == 4:
            raise ValueError(f"All automatic detectors failed for reference {source_index}: {errors}")
        report["warnings"].extend({"source": source_index, **error} for error in errors)
    regions, selection = select_regions(detected["images"], slots)
    report.update(selection, analysis=detected["images"], selected=regions,
                  seconds=time.perf_counter() - started)
    return regions, report
