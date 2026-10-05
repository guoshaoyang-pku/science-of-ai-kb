#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
kb_python="${KB_PYTHON:-python3}"
command -v "$kb_python" >/dev/null 2>&1 || { echo "Python 3.10+ is required." >&2; exit 1; }
"$kb_python" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ is required"'
if [ ! -x .venv/bin/python ]; then
  "$kb_python" -m venv .venv
fi
.venv/bin/python -m pip install -e './aiq_bench_repo[dev]' -r requirements.txt
echo "Installed. Activate with: . .venv/bin/activate"
echo "Verify saved studies with: python tools/verify_research.py"
