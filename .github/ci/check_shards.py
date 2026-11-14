"""Check that pytest shards cover the installed package's collected items."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

from ci_shard import assign, group_key, group_weights, load_weights


def self_test():
    records = [
        {"file": f"test_{name}.py", "name": "test_one", "nodeid": f"test_{name}.py::test_one"}
        for name in "abcd"
    ]
    weights = {"files": dict(zip((record["file"] for record in records), (8, 7, 6, 5))), "tests": {}}
    assignments, loads = assign(records, 2, "file", weights)
    assert loads == [13.0, 13.0], loads
    shuffled = records[:]
    rng = random.Random(42)
    for _ in range(30):
        rng.shuffle(shuffled)
        assert assign(shuffled, 2, "file", weights) == (assignments, loads)

    added = {"file": "test_new.py", "name": "test_new", "nodeid": "test_new.py::test_new"}
    extended = records + [added, records[0]]
    assert group_weights(extended, "file", weights)["test_new.py"] == 6.5
    new_assignments, _ = assign(extended, 3, "file", weights)
    assert sum(new_assignments[group_key(added, "file")] == i for i in range(3)) == 1
    selected = [[r["nodeid"] for r in extended if new_assignments[group_key(r, "file")] == i] for i in range(3)]
    assert sum((Counter(shard) for shard in selected), Counter()) == Counter(r["nodeid"] for r in extended)
    assert all(not (set(a) & set(b)) for i, a in enumerate(selected) for b in selected[i + 1:])

    test_weights = {"files": {}, "tests": {"test_a.py::test_one": 3, "test_b.py::test_one": 1}}
    assert group_weights(records, "test", test_weights)["test_c.py::test_one"] == 2
    assert assign(records, 2, "test", test_weights) == assign(list(reversed(records)), 2, "test", test_weights)
    zero_weights = {"files": {r["file"]: 0 for r in records}, "tests": {}}
    zero_assignments, _ = assign(records, 4, "file", zero_weights)
    assert set(zero_assignments.values()) == set(range(4))
    print("self-test passed: LPT, new and unknown files, coverage, order, test grouping, zero weights")


def counts_arg(value):
    try:
        counts = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("counts must be comma-separated positive integers") from exc
    if not counts or any(count < 1 for count in counts) or len(set(counts)) != len(counts):
        raise argparse.ArgumentTypeError("counts must be distinct positive integers")
    return counts


def collect(python, root, report, weights, shard=None):
    env = os.environ.copy()
    env.pop("SOSFUSE2_SKY_SOS_ROOT", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    plugin_dir = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (plugin_dir, env.get("PYTHONPATH"))))
    command = [
        python, "-B", "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "-p", "ci_shard",
        "--pyargs", "sosfuse2", "--ci-shard-report", str(report),
    ]
    if weights is not None:
        command += ["--ci-shard-weights", str(weights)]
    if shard is not None:
        command += ["--ci-shard", shard]
    result = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, check=False)
    if not report.is_file():
        raise RuntimeError(f"collection failed (rc {result.returncode}):\n{(result.stdout + result.stderr)[-4000:]}")
    data = json.loads(report.read_text(encoding="utf-8"))
    if result.returncode != 0 and not (result.returncode == 5 and not data["items"] and shard is not None):
        raise RuntimeError(f"collection failed (rc {result.returncode}):\n{(result.stdout + result.stderr)[-4000:]}")
    return data


def check(python, counts, weights):
    failed = False
    with tempfile.TemporaryDirectory(prefix="ci-shard-check-") as scratch:
        root = Path(scratch)
        baseline = collect(python, root, root / "baseline.json", weights)
        expected = Counter(item["nodeid"] for item in baseline["items"])
        if not expected:
            raise RuntimeError("baseline collection is empty")
        groups = baseline["group_weights"]
        total_weight = sum(groups.values())
        max_file = max(groups.values(), default=0.0)
        print(f"baseline: {sum(expected.values())} tests, {len(groups)} files, weight={total_weight:.3f}")
        for count in counts:
            ideal = total_weight / count
            ratio = f"{max_file / ideal:.3f}" if ideal else "n/a"
            print(f"\nshards={count}: max_file={max_file:.3f}, ideal={ideal:.3f}, max/ideal={ratio}")
            print("shard  files  tests  weight")
            found = Counter()
            seen = set()
            overlap = set()
            empty = []
            for index in range(count):
                data = collect(python, root, root / f"shard-{count}-{index}.json", weights, f"{index}/{count}")
                summary = data["shard"]
                ids = Counter(item["nodeid"] for item in data["items"])
                overlap.update(seen.intersection(ids))
                seen.update(ids)
                found.update(ids)
                if not ids:
                    empty.append(index)
                if data["group_weights"] != groups:
                    raise RuntimeError(f"collection or weights changed in shard {index}/{count}")
                print(f'{index:5d}  {summary["files"]:5d}  {summary["tests"]:5d}  {summary["weight"]:7.3f}')
            missing = expected - found
            extra = found - expected
            ok = not (missing or extra or overlap or empty)
            failed |= not ok
            print(
                f"coverage={'OK' if ok else 'FAIL'} missing={sum(missing.values())} "
                f"extra={sum(extra.values())} overlap={len(overlap)} empty={empty}"
            )
            if missing or extra or overlap:
                print(f"missing IDs: {list(missing.items())[:10]}")
                print(f"extra IDs: {list(extra.items())[:10]}")
                print(f"overlapping IDs: {sorted(overlap)[:10]}")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--python", help="Python executable with the installed package and pytest")
    parser.add_argument("--counts", type=counts_arg)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.python or not args.counts:
        parser.error("--python and --counts are required unless --self-test is used")
    weights = args.weights.resolve() if args.weights is not None else None
    try:
        load_weights(weights)
        return check(args.python, args.counts, weights)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"shard check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
