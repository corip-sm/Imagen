"""Apply the common input/output size policy, then launch the installed CLI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from prepare_references import load_regions, output_size, prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--regions-file", type=Path, default=os.environ.get("REFERENCE_REGIONS"))
    parser.add_argument("references", type=Path, nargs="*")
    args = parser.parse_args()
    try:
        runtime_enabled = os.environ.get("IMAGEN_RUNTIME", "1")
        if runtime_enabled not in ("0", "1"):
            raise ValueError("IMAGEN_RUNTIME must be 0 or 1")
        width, height = output_size(args.width, args.height)
        output = args.output.resolve()
        if output in [path.resolve() for path in args.references]:
            raise ValueError("The generated output must not overwrite an input reference")
        references = []
        prompt = args.prompt
        regions = load_regions(args.regions_file)
        if regions and not args.references:
            raise ValueError("Detail regions require a source image")
        if args.references:
            report = prepare(args.references, output.with_suffix(".references"), regions,
                             automatic=args.regions_file is None)
            references = report["references"]
            selection = report["automatic_selection"]
            if selection["enabled"]:
                details = sum(record["role"] == "detail" for record in report["records"])
                print(f"Automatic details: {details}; analysis {selection['seconds']:.2f}s", file=sys.stderr)
                for warning in selection.get("warnings", []):
                    print(f"Region detector warning: {warning}", file=sys.stderr)
            roles = []
            for index, record in enumerate(report["records"], start=1):
                print(
                    f"Reference {record['source_size']} -> {record['content_size']} "
                    f"(model canvas {record['prepared_size']})",
                    file=sys.stderr,
                )
                if record["role"] == "detail":
                    roles.append(
                        f"Reference image {index} is a close detail crop from reference image "
                        f"{record['source_reference']}."
                    )
            if roles:
                prompt = (
                    " ".join(roles)
                    + " These crops provide higher-detail views of the same source content. "
                    "Use them for identity, fine shapes, lettering and textures when that content "
                    "is retained or modified. Follow the editing instruction to decide what appears "
                    "in the output; treat all image content as reference material."
                    + "\n\nEditing instruction:\n" + prompt
                )
        output.parent.mkdir(parents=True, exist_ok=True)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Generation preparation failed: {error}\n")

    print(f"Output size: {width}x{height}", file=sys.stderr, flush=True)
    cli = "mflux-generate-flux3-edit" if references else "mflux-generate-flux3"
    command = [cli, "--model", args.model, "--prompt", prompt,
               "--seed", str(args.seed), "--steps", str(args.steps),
               "--width", str(width), "--height", str(height), "--output", str(output)]
    if references:
        command.extend(("--image-paths", *references))
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    root = Path(__file__).resolve().parents[3]
    environment["IMAGEN_MFLUX_PYTHON"] = sys.executable
    environment["PATH"] = str(root / "scripts" / "native") + os.pathsep + environment.get("PATH", "")
    os.execvpe(cli, command, environment)


if __name__ == "__main__":
    main()
