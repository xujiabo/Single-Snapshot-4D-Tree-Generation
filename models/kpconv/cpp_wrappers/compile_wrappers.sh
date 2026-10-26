#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python}"

"$python_bin" "$script_dir/cpp_subsampling/setup.py" build_ext --inplace
"$python_bin" "$script_dir/cpp_neighbors/setup.py" build_ext --inplace
