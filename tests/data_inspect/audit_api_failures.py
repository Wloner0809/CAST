#!/usr/bin/env python3
"""Audit *_progress.jsonl eval files for API-failure trajectories that are
silently counted as pass@1=0. Streams each line; does NOT modify any file.

Run: python3 tests/data_inspect/audit_api_failures.py
"""
import glob
import json
import re
from collections import defaultdict

ERR_RE = re.compile(r"error|exception|timeout|rate.?limit|overloaded", re.IGNORECASE)


def is_empty(x):
    return x is None or (isinstance(x, str) and x.strip() == "")


def find_files():
    files = []
    for env in ("sokoban", "minesweeper", "rush_hour"):
        files += glob.glob(f"outputs/{env}/*/*_n4/*/*/*/{env}_progress.jsonl")
    return sorted(files)


def analyze(path):
    total = 0
    success = 0
    reward_pos = 0
    task_rollouts = defaultdict(int)  # task_key(json rollout-independent) -> count
    # API-failure signals (mutually independent counters)
    sig_a_empty_steps = 0          # steps list empty
    sig_b_last_empty = 0           # last step model_response empty/None
    sig_c_mid_drop = 0             # any step empty response but done=False
    sig_d_single_bad = 0           # 1 step, fail, empty resp or invalid action
    sig_e_error_marker = 0         # error/timeout/overloaded substring
    sig_e_finish_length = 0        # any step finish_reason=='length' (truncation)
    # union of "suspected API failure AND currently counted as failure"
    suspect_failures = 0

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tr = rec["trajectory"]
            total += 1
            tk = rec.get("task_key")
            # task_key format (compute_task_key): "{json}::rollout=N".
            # Strip the ::rollout=N suffix so we group the N rollouts of one
            # problem together and can count rollouts/task.
            base = str(tk)
            if "::rollout=" in base:
                base = base.rsplit("::rollout=", 1)[0]
            task_rollouts[base] += 1

            info = tr.get("info") or {}
            is_succ = info.get("success") is True
            if is_succ:
                success += 1
            if (tr.get("reward") or 0) > 0:
                reward_pos += 1

            steps = tr.get("steps") or []
            a = b = c = d_sig = e = e_len = False

            if len(steps) == 0:
                a = True
            else:
                last = steps[-1]
                if is_empty(last.get("model_response")):
                    b = True
                for s in steps:
                    if is_empty(s.get("model_response")) and not s.get("done", False):
                        c = True
                    mo = s.get("model_output") or {}
                    if mo.get("finish_reason") == "length":
                        e_len = True
                    # error markers in step.info + model_output finish_reason
                    if ERR_RE.search(json.dumps(s.get("info") or {})):
                        e = True
                    fr = mo.get("finish_reason")
                    if isinstance(fr, str) and ERR_RE.search(fr):
                        e = True
                if len(steps) == 1:
                    s0 = steps[0]
                    if (not is_succ) and (
                        is_empty(s0.get("model_response"))
                        or (s0.get("info") or {}).get("action_is_valid") is False
                    ):
                        d_sig = True
            if ERR_RE.search(json.dumps(info)):
                e = True

            if a:
                sig_a_empty_steps += 1
            if b:
                sig_b_last_empty += 1
            if c:
                sig_c_mid_drop += 1
            if d_sig:
                sig_d_single_bad += 1
            if e:
                sig_e_error_marker += 1
            if e_len:
                sig_e_finish_length += 1

            # "suspected API failure" = strong signals a/b/c/d (not truncation,
            # which is a real generation, and not error-marker alone which here
            # is noise-free but weak). Count only if currently a failure.
            if (a or b or c or d_sig) and not is_succ:
                suspect_failures += 1

    n_tasks = len(task_rollouts)
    from collections import Counter as _C
    roll_dist = dict(_C(task_rollouts.values()))
    uneven = any(v != 4 for v in task_rollouts.values())
    return {
        "path": path,
        "total": total,
        "n_tasks": n_tasks,
        "ratio": round(total / n_tasks, 3) if n_tasks else 0,
        "rollout_counts": roll_dist,
        "uneven": uneven,
        "success": success,
        "reward_pos": reward_pos,
        "a_empty_steps": sig_a_empty_steps,
        "b_last_empty": sig_b_last_empty,
        "c_mid_drop": sig_c_mid_drop,
        "d_single_bad": sig_d_single_bad,
        "e_error_marker": sig_e_error_marker,
        "e_finish_length": sig_e_finish_length,
        "suspect_failures": suspect_failures,
    }


def short(path):
    # env / model / difficulty
    p = path.split("/")
    return f"{p[1]}/{p[2].split('_n4')[0]}"


def main():
    files = find_files()
    rows = [analyze(f) for f in files]

    hdr = ("env/model/run", "tot", "tsk", "r/t", "succ", "rw>0",
           "A_empty", "B_last", "C_mid", "D_1bad", "E_err", "E_len", "SUSPECT")
    print("\n=== PER-FILE AUDIT ===")
    print(f"{hdr[0]:<52} " + " ".join(f"{h:>7}" for h in hdr[1:]))
    for r in rows:
        flag = "*" if r["uneven"] else " "
        print(f"{short(r['path']):<52} "
              f"{r['total']:>7}{flag}{r['n_tasks']:>6} {r['ratio']:>7} "
              f"{r['success']:>7} {r['reward_pos']:>7} "
              f"{r['a_empty_steps']:>7} {r['b_last_empty']:>7} {r['c_mid_drop']:>7} "
              f"{r['d_single_bad']:>7} {r['e_error_marker']:>7} {r['e_finish_length']:>7} "
              f"{r['suspect_failures']:>7}")

    print("\n=== GRAND SUMMARY ===")
    keys = ["total", "success", "reward_pos", "a_empty_steps", "b_last_empty",
            "c_mid_drop", "d_single_bad", "e_error_marker", "e_finish_length",
            "suspect_failures"]
    tot = {k: sum(r[k] for r in rows) for k in keys}
    for k in keys:
        print(f"  {k:<18}: {tot[k]}")
    print(f"  files                : {len(rows)}")
    print("  files w/ uneven rollouts (!=4 per task):")
    any_uneven = False
    for r in rows:
        if r["uneven"]:
            any_uneven = True
            print(f"    {short(r['path']):<52} tasks={r['n_tasks']} "
                  f"rollout_dist={r['rollout_counts']}")
    if not any_uneven:
        print("    (none — every task has exactly 4 rollouts)")

    print("\n=== TOP SUSPECT FILES ===")
    for r in sorted(rows, key=lambda x: -x["suspect_failures"])[:8]:
        print(f"  {r['suspect_failures']:>4} suspect  {short(r['path'])}")


if __name__ == "__main__":
    main()
