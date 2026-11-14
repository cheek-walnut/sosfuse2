"""Assign collected pytest items to repeatable CI shards."""

import json
import math
from pathlib import Path
import re
import statistics


def load_weights(path):
    if path is None:
        return {"files": {}, "tests": {}}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"files", "tests"}:
        raise ValueError("weights must contain 'files' and 'tests' objects")
    for section in ("files", "tests"):
        if not isinstance(data[section], dict):
            raise ValueError(f"weights {section!r} must be an object")
        for key, value in data[section].items():
            if (
                not isinstance(key, str)
                or not key
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"invalid weight in {section}: {key!r}")
    return data


def group_key(record, mode):
    if mode == "file":
        return record["file"]
    if mode == "test":
        return f'{record["file"]}::{record["name"]}'
    raise ValueError(f"unknown grouping: {mode}")


def group_weights(records, mode, weights):
    known = weights["files" if mode == "file" else "tests"]
    fallback = statistics.median(known.values()) if known else 1.0
    keys = {group_key(record, mode) for record in records}
    return {key: float(known.get(key, fallback)) for key in sorted(keys)}


def assign(records, count, mode, weights):
    """Return group assignments and estimated loads for all collected items."""
    if count < 1:
        raise ValueError("shard count must be positive")
    groups = group_weights(records, mode, weights)
    loads = [0.0] * count
    sizes = [0] * count
    assignments = {}
    for key in sorted(groups, key=lambda key: (-groups[key], key)):
        shard = min(range(count), key=lambda index: (loads[index], sizes[index], index))
        assignments[key] = shard
        loads[shard] += groups[key]
        sizes[shard] += 1
    return assignments, loads


def parse_shard(value):
    if not re.fullmatch(r"[0-9]+/[0-9]+", value):
        raise ValueError("--ci-shard must be INDEX/COUNT")
    index, count = (int(part) for part in value.split("/"))
    if count < 1 or index >= count:
        raise ValueError("--ci-shard requires 0 <= INDEX < COUNT")
    return index, count


def pytest_addoption(parser):
    group = parser.getgroup("CI sharding")
    group.addoption("--ci-shard", metavar="INDEX/COUNT")
    group.addoption("--ci-shard-weights", metavar="PATH")
    group.addoption("--ci-shard-by", choices=("file", "test"), default="file")
    group.addoption(
        "--ci-shard-order",
        choices=("collected", "weight"),
        default="collected",
        help="order of the shard's items: as collected, or the heaviest groups first (each group kept in collected order)",
    )
    group.addoption("--ci-shard-report", metavar="PATH", help="write collected item data as JSON")


def pytest_configure(config):
    import pytest

    try:
        option = config.getoption("--ci-shard")
        shard = parse_shard(option) if option is not None else None
        weights = load_weights(config.getoption("--ci-shard-weights"))
    except (OSError, ValueError) as exc:
        raise pytest.UsageError(f"ci-shard: {exc}") from exc
    config._ci_shard_state = {"shard": shard, "weights": weights, "summary": None}


def pytest_collection_modifyitems(config, items):
    state = config._ci_shard_state
    mode = config.getoption("--ci-shard-by")
    records = [{"nodeid": item.nodeid, "file": item.path.name, "name": item.name} for item in items]
    groups = group_weights(records, mode, state["weights"])
    shard = state["shard"]
    if shard is not None:
        index, count = shard
        assignments, loads = assign(records, count, mode, state["weights"])
        chosen = [assignments[group_key(record, mode)] == index for record in records]
        selected = [item for item, keep in zip(items, chosen) if keep]
        deselected = [item for item, keep in zip(items, chosen) if not keep]
        records = [record for record, keep in zip(records, chosen) if keep]
        if deselected:
            config.hook.pytest_deselected(items=deselected)
        if config.getoption("--ci-shard-order") == "weight":
            # Longest groups first, each group in collected order: a parallel run then starts the file that bounds the
            # shard's time at once instead of when its turn in the collected order comes.
            first = {}
            for position, record in enumerate(records):
                first.setdefault(group_key(record, mode), position)
            order = sorted(
                range(len(selected)),
                key=lambda i: (-groups[group_key(records[i], mode)], first[group_key(records[i], mode)], i),
            )
            selected = [selected[i] for i in order]
            records = [records[i] for i in order]
        items[:] = selected
        state["summary"] = {
            "index": index,
            "count": count,
            "files": len({record["file"] for record in records}),
            "tests": len(records),
            "weight": loads[index],
        }
    report = config.getoption("--ci-shard-report")
    if report:
        Path(report).write_text(
            json.dumps({"items": records, "group_weights": groups, "shard": state["summary"]}, ensure_ascii=False),
            encoding="utf-8",
        )


def pytest_report_collectionfinish(config):
    summary = config._ci_shard_state["summary"]
    if summary is not None:
        return (
            f'ci-shard {summary["index"]}/{summary["count"]}: files={summary["files"]}, '
            f'tests={summary["tests"]}, weight={summary["weight"]:.3f}'
        )
