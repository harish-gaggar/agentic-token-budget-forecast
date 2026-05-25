#!/usr/bin/env bash
# End-to-end reproducibility check (no git). Run from repo root:
#   ./scripts/verify.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY=""
for cand in python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    ver="$("$cand" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    major="${ver%%.*}"
    minor="${ver#*.}"
    if [[ "$major" -eq 3 && "$minor" -ge 10 && "$minor" -le 12 ]]; then
      PY="$cand"
      break
    fi
  fi
done

if [[ -z "$PY" ]]; then
  echo "ERROR: Need Python 3.10, 3.11, or 3.12 (3.14+ lacks numpy wheels on many indexes)."
  exit 1
fi

echo "==> Using $PY ($("$PY" --version))"

if [[ ! -x .venv/bin/python ]]; then
  echo "==> Creating .venv"
  "$PY" -m venv .venv
fi

PIP_INDEX="${PIP_INDEX_URL:-https://pypi.org/simple}"
echo "==> pip install (-r requirements.txt) via $PIP_INDEX"
.venv/bin/pip install -q --index-url "$PIP_INDEX" -r requirements.txt

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplcache}"
PYBIN=".venv/bin/python"

step() { echo ""; echo "==> $1"; }

step "TABF smoke (README quick example)"
$PYBIN -c "
from tabf import TABF
fc = TABF().forecast('Refactor the user-auth module to use JWT tokens.')
assert fc.total_tokens > 0
print('  task_type=%s complexity=%s total_tokens=%s' % (fc.task_type, fc.complexity, fc.total_tokens))
"

step "Synthetic benchmark (experiments.run_all)"
$PYBIN -m experiments.run_all --out results
$PYBIN -c "
import json
from pathlib import Path
m = json.loads(Path('results/manifest.json').read_text())
assert m['n_corpus'] == 600
w = m['main']['TABF_gbt']['w20r']
assert abs(w - 74.05405405405405) < 1e-4, w
print('  manifest OK: TABF_gbt W20R=%.2f%%' % w)
"

step "Production figures (plot_production)"
$PYBIN -m experiments.plot_production --out results
for f in fig_production_savings.pdf fig_production_costs.pdf fig_layer_attribution.pdf; do
  test -s "results/$f" || { echo "MISSING results/$f"; exit 1; }
done
echo "  wrote 3 production PDFs"

step "Generic-agent figure (plot_generic_agents)"
$PYBIN -m experiments.plot_generic_agents --out results
test -s results/fig_generic_agents.pdf

step "Reviewer-evidence scripts"
make -s multiseed stagewise calibration stress tabf-replay \
  stagewise-gbt real-trace quality-outcomes compression-baseline oracle-labels

test -s paper.pdf || { echo "MISSING paper.pdf (bundled manuscript)"; exit 1; }

echo ""
echo "All checks passed."
echo "Key outputs:"
echo "  results/manifest.json          synthetic benchmark summary"
echo "  results/fig_w20r.pdf           main W20R figure"
echo "  results/fig_production_*.pdf   production case study"
echo "  results/multiseed_summary.csv  robustness (5 seeds)"
echo "  paper.pdf                      manuscript (read bundled PDF; compare tables to manifest)"
