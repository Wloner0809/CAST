#!/usr/bin/env bash
#
# Launch the Rush Hour trajectory visualizer (.pt variant).
#
# Serves rLLM EVAL outputs serialized as torch .pt files (list[Trajectory]),
# i.e. rush_hour_trajectories.pt produced by the closed-model n4 evals. This is
# the counterpart to rush_hour.sh, which instead serves training-checkpoint
# chat_completions/*.jsonl. Correctness here is success-based (info['success']).
#
# Data live under:
#   OUT_ROOT/<model>/<run_tag>/rush_hour/<difficulty>/<model>/rush_hour_trajectories.pt
#
# Usage:
#   ./rush_hour_pt.sh                          # default target (first .pt found, see below)
#   ./rush_hour_pt.sh <relative-path>          # a path under OUT_ROOT (dir or .pt file)
#   ./rush_hour_pt.sh /abs/path/to/x.pt        # an explicit .pt file
#   ./rush_hour_pt.sh /abs/path/dir            # an explicit dir containing .pt files
#   ./rush_hour_pt.sh <target> --port 8800 --theme light
#   ./rush_hour_pt.sh --list                   # list available .pt trajectory files
#
# Any extra flags (--port, --theme, --no-browser) are forwarded to the visualizer.
set -euo pipefail

ENV_NAME="rush_hour"

# Resolve repo paths relative to THIS script so it works from any cwd.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VISUAL_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${VISUAL_DIR}/../.." && pwd)"
VISUALIZER="${VISUAL_DIR}/${ENV_NAME}_visual_pt.py"

# Root of eval .pt outputs for this env.
OUT_ROOT="${REPO_ROOT}/outputs/${ENV_NAME}"

# Default target: set TRAJ_PT to a .pt path, else falls back to the first
# discovered .pt. Kept as a relative path under OUT_ROOT.
DEFAULT_TARGET="${TRAJ_PT:-<PATH_TO_TRAJECTORIES_PT>}"

list_pt() {
  echo "Available ${ENV_NAME} .pt trajectory files under:"
  echo "  ${OUT_ROOT}"
  echo
  # Skip pruned run subtrees; only list dumps matching the eval naming pattern.
  while IFS= read -r p; do
    [ -f "${p}" ] || continue
    rel="${p#${OUT_ROOT}/}"
    printf "  %s\n" "${rel}"
  done < <(find "${OUT_ROOT}" -path '*n1-run1*' -prune -o \
             -name "${ENV_NAME}_trajectories.pt" -print 2>/dev/null | grep '_n4/' | sort)
}

# --- Parse the first positional arg (target / --list) ---
TARGET="${DEFAULT_TARGET}"
EXTRA_ARGS=()

if [[ $# -gt 0 ]]; then
  case "$1" in
    --list|-l)
      list_pt
      exit 0
      ;;
    --help|-h)
      sed -n '2,20p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    -*)
      # No target given; first token is already a flag for the visualizer.
      EXTRA_ARGS=("$@")
      ;;
    *)
      TARGET="$1"
      shift
      EXTRA_ARGS=("$@")
      ;;
  esac
fi

# --- Resolve TARGET into a .pt file or directory the visualizer accepts ---
if [[ -e "${TARGET}" ]]; then
  DATA_PATH="${TARGET}"                 # absolute or cwd-relative path as given
elif [[ -e "${OUT_ROOT}/${TARGET}" ]]; then
  DATA_PATH="${OUT_ROOT}/${TARGET}"     # relative to the env output root
else
  # Default target missing (e.g. that model not yet evaluated): fall back to the
  # first .pt discovered so the script still launches something useful.
  DATA_PATH="$(find "${OUT_ROOT}" -path '*n1-run1*' -prune -o \
                 -name "${ENV_NAME}_trajectories.pt" -print 2>/dev/null | grep '_n4/' | sort | head -1)"
  if [[ -n "${DATA_PATH}" ]]; then
    echo "Note: target '${TARGET}' not found; falling back to first .pt:" >&2
    echo "  ${DATA_PATH}" >&2
  fi
fi

# --- Validate ---
if [[ ! -f "${VISUALIZER}" ]]; then
  echo "Error: visualizer not found at ${VISUALIZER}" >&2
  exit 1
fi
if [[ -z "${DATA_PATH:-}" || ! -e "${DATA_PATH}" ]]; then
  echo "Error: no .pt data found for target: ${TARGET}" >&2
  echo "Run '$0 --list' to see available .pt files." >&2
  exit 1
fi
# If it's a directory, make sure it actually holds .pt files.
if [[ -d "${DATA_PATH}" ]]; then
  n_pt="$(find "${DATA_PATH}" -maxdepth 1 -name '*.pt' | wc -l | tr -d ' ')"
  if [[ "${n_pt}" -eq 0 ]]; then
    echo "Error: no .pt files found in directory ${DATA_PATH}" >&2
    exit 1
  fi
fi

echo "Visualizer: ${VISUALIZER}"
echo "Data path:  ${DATA_PATH}"
echo

exec python3 "${VISUALIZER}" --path "${DATA_PATH}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
