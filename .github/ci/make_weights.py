"""Convert a pytest timing TSV into file and test weights."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys


def test_key(nodeid, source_file=None):
    path, separator, remainder = nodeid.partition("::")
    if not separator or not remainder:
        raise ValueError(f"node ID has no test name: {nodeid!r}")
    filename = Path(path).name if path else source_file
    if not filename:
        raise ValueError(f"node ID has no file name; use --source-file: {nodeid!r}")
    if source_file and path and filename != source_file:
        raise ValueError(f"node ID file differs from --source-file: {nodeid!r}")

    depth = 0
    last_separator = -1
    for index, char in enumerate(remainder):
        if char == "[":
            depth += 1
        elif char == "]" and depth:
            depth -= 1
        elif char == ":" and remainder[index:index + 2] == "::" and depth == 0:
            last_separator = index
    name = remainder[last_separator + 2:] if last_separator >= 0 else remainder
    if not name:
        raise ValueError(f"node ID has no test name: {nodeid!r}")
    return filename, f"{filename}::{name}"


def convert(source, source_file=None):
    files = defaultdict(float)
    tests = defaultdict(float)
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"nodeid", "wall_s", "cpu_s"} <= set(reader.fieldnames):
            raise ValueError("TSV header must contain nodeid, wall_s, and cpu_s")
        for line, row in enumerate(reader, start=2):
            try:
                filename, key = test_key(row["nodeid"], source_file)
                value = float(row["wall_s"])
                if not math.isfinite(value) or value < 0:
                    raise ValueError("wall_s must be finite and non-negative")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"line {line}: {exc}") from exc
            files[filename] += value
            tests[key] += value
    if not tests:
        raise ValueError("TSV has no test timings")
    return {"files": dict(sorted(files.items())), "tests": dict(sorted(tests.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("shard_weights.json"))
    parser.add_argument("--source-file", type=lambda value: Path(value).name, help="file for node IDs without a path")
    args = parser.parse_args()
    try:
        weights = convert(args.tsv, args.source_file)
        args.output.write_text(json.dumps(weights, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"cannot make weights: {exc}", file=sys.stderr)
        return 1
    print(f'wrote {len(weights["files"])} file weights and {len(weights["tests"])} test weights to {args.output}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
