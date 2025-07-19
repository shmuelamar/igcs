#!/usr/bin/env bash
# A simple script to install and test the igcs python package.

set -euo pipefail  # abort on error, unset var, or failed pipe

# ---- 1) Build wheel + sdist -------------------------------------------------
# Build already done in github action.
#python -m pip install --upgrade pip build wheel > /dev/null
#python -m build --wheel --sdist               # outputs to dist/

# ---- 2) Create an isolated venv & install the fresh wheel -------------------
tmpdir="$(mktemp -d)"
python -m venv "$tmpdir/venv"
source "$tmpdir/venv/bin/activate"
python -m pip install --upgrade pip > /dev/null
python -m pip install twine

twine check dist/*

python -m pip install dist/*.whl              # installs igcs package

# ---- 3) Verify package version ----------------------------------------------

python - <<'PY'
import importlib.metadata
print("IGCS version", importlib.metadata.version("igcs"))
PY

# ---- 4) Run the smoke test --------------------------------------------------

python - <<'PY'
from igcs import grounding
from igcs.entities import Doc

selections = grounding.ground_selections(
    ["the cat on the mat"],
    docs=[Doc(id=0, text="the cat sat on the mat")],
    max_dist_rel=0.5,
)
assert len(selections) == 1 and selections[0].doc_id == 0
print("✅  Smoke test passed.")
PY

echo "All done. Wheel lives in ./dist/, venv lives in $tmpdir"
