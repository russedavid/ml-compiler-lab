"""Positive and semantic-negative tests for the built scheduling plugin."""

import argparse
from pathlib import Path
import subprocess
import tempfile

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iree-opt", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="lab-pass-") as tmp:
        for name, strategy in [("small", "persistent"), ("fallback", "conventional")]:
            output = Path(tmp) / (name + ".mlir")
            result = subprocess.run(
                [
                    str(args.iree_opt),
                    "--iree-plugin=lab",
                    "--lab-schedule-row-mlp",
                    str(root / "compiler/tests" / f"{name}.mlir"),
                    "-o",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
            text = output.read_text()
            assert '"lab.schedule"' in text and f'strategy = "{strategy}"' in text
            assert '"lab.graph"' not in text
        bad = subprocess.run(
            [
                str(args.iree_opt),
                "--iree-plugin=lab",
                "--lab-schedule-row-mlp",
                str(root / "compiler/tests/invalid.mlir"),
            ],
            capture_output=True,
            text=True,
        )
        assert bad.returncode != 0 and "proven independent rows" in bad.stderr
    print("COMPILER_PASS_CHECKS")


if __name__ == "__main__":
    main()
