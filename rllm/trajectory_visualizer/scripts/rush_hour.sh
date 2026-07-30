#!/usr/bin/env bash
#
# Launch the Rush Hour trajectory visualizer.
#
# Serves the chat_completions/ directory of a training/eval checkpoint:
#   - N.jsonl       -> training rollouts (16 prompts x 8 rollouts, auto-detected)
#   - val_N.jsonl   -> validation rollouts
#
# Usage:
#   ./rush_hour.sh                         # list available experiments
#   ./rush_hour.sh <experiment-name>       # a ckpt dir name under CKPT_ROOT
#   ./rush_hour.sh /abs/path/chat_completions   # an explicit directory
#   ./rush_hour.sh <name> --port 8800 --theme light
#   ./rush_hour.sh --list                  # list available experiments
#
# Any extra flags (--port, --theme, --no-browser) are forwarded to the visualizer.
set -euo pipefail

# Resolve repo paths relative to THIS script so it works from any cwd.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VISUAL_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VISUALIZER="${VISUAL_DIR}/rush_hour_visual.py"

CKPT_ROOT="${CKPT_ROOT:-<PATH_TO_YOUR_CHECKPOINT>}"
# No baked-in default: checkpoint directory names are specific to whoever ran the
# training. Set DEFAULT_EXPERIMENT, or pass the experiment name as the first
# argument. With neither, the script lists what is available under CKPT_ROOT.
DEFAULT_EXPERIMENT="${DEFAULT_EXPERIMENT:-}"

list_experiments() {
  echo "Available experiments under:"
  echo "  ${CKPT_ROOT}"
  echo
  # Some experiments nest chat_completions one level deeper, so search up to
  # 3 levels and report the path relative to CKPT_ROOT (which is what you pass
  # back as the experiment name).
  while IFS= read -r d; do
    [ -d "${d}" ] || continue
    exp="$(dirname "${d}")"
    exp="${exp#${CKPT_ROOT}/}"
    n_train="$(find "${d}" -maxdepth 1 -name '[0-9]*.jsonl' 2>/dev/null | wc -l | tr -d ' ')"
    n_val="$(find "${d}" -maxdepth 1 -name 'val_*.jsonl' 2>/dev/null | wc -l | tr -d ' ')"
    printf "  %-110s (train=%s val=%s)\n" "${exp}" "${n_train}" "${n_val}"
  done < <(find "${CKPT_ROOT}" -maxdepth 3 -type d -name chat_completions 2>/dev/null | sort)
}

# --- Parse the first positional arg (experiment / path / --list) ---
TARGET="${DEFAULT_EXPERIMENT}"
EXTRA_ARGS=()

if [[ $# -gt 0 ]]; then
  case "$1" in
    --list|-l)
      list_experiments
      exit 0
      ;;
    --help|-h)
      sed -n '2,18p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    -*)
      # No experiment given; first token is already a flag for the visualizer.
      EXTRA_ARGS=("$@")
      ;;
    *)
      TARGET="$1"
      shift
      EXTRA_ARGS=("$@")
      ;;
  esac
fi

# No experiment named and no DEFAULT_EXPERIMENT set: show what is available
# rather than failing on an empty path.
if [[ -z "${TARGET}" ]]; then
  echo "No experiment specified. Pass one as the first argument, or set DEFAULT_EXPERIMENT."
  echo
  list_experiments
  exit 1
fi

# --- Resolve TARGET into a chat_completions directory ---
if [[ -d "${TARGET}" ]]; then
  # Explicit path. Accept either the ckpt dir or the chat_completions dir.
  if [[ -d "${TARGET}/chat_completions" ]]; then
    DATA_DIR="${TARGET}/chat_completions"
  else
    DATA_DIR="${TARGET}"
  fi
else
  # Treat as an experiment name under CKPT_ROOT.
  if [[ -d "${CKPT_ROOT}/${TARGET}/chat_completions" ]]; then
    DATA_DIR="${CKPT_ROOT}/${TARGET}/chat_completions"
  else
    DATA_DIR="${CKPT_ROOT}/${TARGET}"
  fi
fi

# --- Validate ---
if [[ ! -f "${VISUALIZER}" ]]; then
  echo "Error: visualizer not found at ${VISUALIZER}" >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Error: data directory not found: ${DATA_DIR}" >&2
  echo "Run '$0 --list' to see available experiments." >&2
  exit 1
fi
n_jsonl="$(find "${DATA_DIR}" -maxdepth 1 -name '*.jsonl' | wc -l | tr -d ' ')"
if [[ "${n_jsonl}" -eq 0 ]]; then
  echo "Error: no .jsonl files found in ${DATA_DIR}" >&2
  exit 1
fi

echo "Visualizer: ${VISUALIZER}"
echo "Data dir:   ${DATA_DIR}"
echo

exec python3 "${VISUALIZER}" --path "${DATA_DIR}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
