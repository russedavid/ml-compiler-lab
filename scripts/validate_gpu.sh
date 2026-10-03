#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
san=${COMPUTE_SANITIZER:-compute-sanitizer}
"$san" --tool memcheck --error-exitcode 99 .venv/bin/python check_kernels.py
"$san" --tool racecheck --error-exitcode 99 .venv/bin/python check_kernels.py
"$san" --tool synccheck --error-exitcode 99 .venv/bin/python check_kernels.py
"$san" --tool memcheck --error-exitcode 99 .venv/bin/python check_tensorcore.py
"$san" --tool memcheck --error-exitcode 99 .venv/bin/python check_tensorcore_mlp.py
"$san" --tool racecheck --error-exitcode 99 .venv/bin/python check_tensorcore_mlp.py
"$san" --tool synccheck --error-exitcode 99 .venv/bin/python check_tensorcore_mlp.py
