"""Numerical and optional task/slice evaluation for supplied representative data."""

import argparse
import importlib
import json
from pathlib import Path
import numpy as np
from .metrics import compare


def evaluate(reference, candidate, inputs, labels=None, groups=None):
    inputs = np.asarray(inputs)
    if inputs.shape[0] < 1:
        raise ValueError("at least one example required")
    if labels is not None and len(labels) != len(inputs):
        raise ValueError("label count mismatch")
    if groups is not None and len(groups) != len(inputs):
        raise ValueError("group count mismatch")
    records = []
    slices = {}
    for index, value in enumerate(inputs):
        expected = np.asarray(reference(value))
        actual = np.asarray(candidate(value))
        check = compare(actual, expected)
        record = {"case": index, "numerical": check}
        if labels is not None:
            if expected.ndim != 1 or actual.ndim != 1:
                raise ValueError("classification requires one-dimensional logits")
            label = int(labels[index])
            if not 0 <= label < expected.size:
                raise ValueError("label outside class range")
            record["reference_correct"] = bool(
                np.isfinite(expected).all() and int(np.argmax(expected)) == label
            )
            record["candidate_correct"] = bool(
                np.isfinite(actual).all() and int(np.argmax(actual)) == label
            )
        group = str(groups[index]) if groups is not None else "all"
        record["slice"] = group
        records.append(record)
        slices.setdefault(group, []).append(record)

    def aggregate(items):
        result = {
            "examples": len(items),
            "numerical_passes": sum(r["numerical"]["passed"] for r in items),
        }
        if labels is not None:
            result.update(
                reference_accuracy=sum(r["reference_correct"] for r in items)
                / len(items),
                candidate_accuracy=sum(r["candidate_correct"] for r in items)
                / len(items),
            )
        return result

    return {
        "summary": aggregate(records),
        "slices": {g: aggregate(v) for g, v in slices.items()},
        "cases": records,
        "limits": "Task accuracy requires supplied valid labels and representative data; slice definitions do not establish demographic coverage.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--factory",
        required=True,
        help="module:function returning (reference,candidate)",
    )
    p.add_argument(
        "--data",
        type=Path,
        required=True,
        help="NPZ with inputs and optional labels/groups",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    module, name = args.factory.split(":", 1)
    reference, candidate = getattr(importlib.import_module(module), name)()
    with np.load(args.data, allow_pickle=False) as data:
        report = evaluate(
            reference,
            candidate,
            data["inputs"],
            data["labels"] if "labels" in data else None,
            data["groups"] if "groups" in data else None,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
