#!/usr/bin/env bash
# =============================================================================
# key_pool.sh -- closed-source API key pool + failover (shared by closed evals)
# =============================================================================
# After sourcing, call:
#     run_eval_with_failover <logical_model> <config_yaml> <output_root> [output_subtag]
#
# For a logical model name it expands the interchangeable variants (MODEL_VARIANTS),
# collects every key that supports a variant (KEY_SUPPORTS), injects them via
# --set model.api_keys=k1,k2,... and runs the eval. The engine rotates keys per
# request; the shell only moves to the next variant if a whole run fails.
#
# Callers pass these through as env vars: DIFFICULTY / EXTRA_OVERRIDES /
# N_ROLLOUTS / N_PARALLEL / LIMIT. Do not put model.api_keys in EXTRA_OVERRIDES.

# -----------------------------------------------------------------------------
# 1) API key pool (priority order; earlier entries are preferred)
#    Replace the placeholders with your own keys, or source them from the
#    environment, e.g. CLOSED_API_KEYS=("${CLOSED_API_KEY_0:-}" ...).
# -----------------------------------------------------------------------------
CLOSED_API_KEYS=(
    "<YOUR_API_KEY_0>"   # key0
    "<YOUR_API_KEY_1>"   # key1
)

# -----------------------------------------------------------------------------
# 2) variants each key may serve (indices align with CLOSED_API_KEYS). Edit here on quota changes.
# -----------------------------------------------------------------------------
declare -A KEY_SUPPORTS=(
    [0]="claude-opus-4.5 claude-sonnet-4.5 claude-sonnet-4.6 gemini-2.5-pro gemini-2.5-flash"
    [1]="claude-opus-4.6"
)

# -----------------------------------------------------------------------------
# 3) logical model name -> interchangeable variants, in preference order
# -----------------------------------------------------------------------------
declare -A MODEL_VARIANTS=(
    # Claude
    ["opus-4.6"]="claude-opus-4.6"
    ["opus-4.5"]="claude-opus-4.5"
    ["sonnet-4.6"]="claude-sonnet-4.6"
    ["sonnet-4.5"]="claude-sonnet-4.5"
    # Gemini
    ["gemini-2.5-pro"]="gemini-2.5-pro"
    ["gemini-2.5-flash"]="gemini-2.5-flash"
)

# -----------------------------------------------------------------------------
# internal: check whether key#idx supports a variant
# -----------------------------------------------------------------------------
_key_supports_variant() {
    local idx="$1" variant="$2" v
    for v in ${KEY_SUPPORTS[$idx]:-}; do
        [[ "$v" == "$variant" ]] && return 0
    done
    return 1
}

# -----------------------------------------------------------------------------
# $1 logical_model  $2 config_yaml  $3 output_root  $4 output_subtag (optional)
# Output lands in {output_root}/closed/{variant}/{output_subtag}.
# Returns 0 if any variant succeeded, 1 otherwise.
# -----------------------------------------------------------------------------
run_eval_with_failover() {
    local logical="$1" config="$2" out_root="$3" subtag="${4:-}"
    local script_dir variants idx key variant tried=0 out_dir

    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"  # -> eval_scripts/

    variants="${MODEL_VARIANTS[$logical]:-}"
    if [[ -z "${variants}" ]]; then
        echo "[failover] ERROR: unknown logical model name '${logical}' (not in MODEL_VARIANTS)" >&2
        echo "[failover] available logical names: ${!MODEL_VARIANTS[*]}" >&2
        return 1
    fi

    echo "[failover] logical model=${logical}  candidate variants=[${variants}]"
    for variant in ${variants}; do
        # collect all keys supporting this variant (in CLOSED_API_KEYS priority order)
        local keys_csv="" key_count=0
        for idx in "${!CLOSED_API_KEYS[@]}"; do
            if _key_supports_variant "${idx}" "${variant}"; then
                keys_csv="${keys_csv:+${keys_csv},}${CLOSED_API_KEYS[$idx]}"
                key_count=$((key_count + 1))
            fi
        done
        if [[ "${key_count}" -eq 0 ]]; then
            continue   # no key supports this variant, skip
        fi
        tried=$((tried + 1))
        out_dir="${out_root}/${variant}"
        [[ -n "${subtag}" ]] && out_dir="${out_dir}/${subtag}"
        echo "[failover] attempt #${tried}: variant=${variant} (${key_count} keys for per-request failover)"

        # overridable via EVAL_UNIFIED_BIN; the concurrent variant is safe for single runs too
        local eval_bin="${EVAL_UNIFIED_BIN:-${script_dir}/lib/eval_unified_concurrent.sh}"
        # export, not a command prefix: a ${LIMIT:+..} prefix would parse as a command name
        [[ -n "${LIMIT:-}" ]] && export LIMIT
        if EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-} model.api_keys=${keys_csv}" \
           DIFFICULTY="${DIFFICULTY:-}" \
           N_ROLLOUTS="${N_ROLLOUTS:-1}" \
           N_PARALLEL="${N_PARALLEL:-10}" \
           bash "${eval_bin}" "closed/${variant}" "${config}" "${out_dir}"; then
            echo "[failover] success: variant=${variant}"
            return 0
        fi
        echo "[failover] failed(exit≠0): variant=${variant}, moving to next variant" >&2
    done

    if [[ "${tried}" -eq 0 ]]; then
        echo "[failover] ERROR: no supported key for any variant of logical name '${logical}' (check KEY_SUPPORTS/MODEL_VARIANTS)" >&2
    else
        echo "[failover] ERROR: logical name '${logical}': all ${tried} variants tried and failed" >&2
    fi
    return 1
}
