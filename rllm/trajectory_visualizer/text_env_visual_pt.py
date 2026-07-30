#!/usr/bin/env python3
"""
Text-Env Trajectory Visualizer (.pt variant)
============================================
A web-based viewer for text-based RL evaluation trajectories serialized as
torch `.pt` files (rLLM `list[Trajectory]` format). Built for environments whose
observations are natural-language text rather than a spatial board, e.g.
ALFWorld (alfworld_trajectories.pt) and WebShop (webshop_trajectories.pt).

Correctness is success-based: each trajectory is judged correct/incorrect by
`info['success']`, NOT by the (cumulative, possibly fractional) env reward.
This mirrors the backend of sokoban_visual_pt.py; only the frontend differs:
instead of rendering a Sokoban board grid, this viewer renders the text
observations continuously and highlights the model's action commands
(the ```...``` blocks the agent emits, e.g. ```go to desk 1```,
```search[...]```, ```click[Buy Now]```).

Usage:
    python3 text_env_visual_pt.py --path <file.pt | dir_of_pt_files> [--port 8767]

Examples:
    python3 text_env_visual_pt.py --path .../alfworld/eval_id/global_step_200_hf_model/alfworld_trajectories.pt
    python3 text_env_visual_pt.py --path .../webshop/default/global_step_200_hf_model/webshop_trajectories.pt
"""

import argparse
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse


# =============================================================================
# .pt adapter
# -----------------------------------------------------------------------------
# Load evaluation outputs serialized as a ``list[Trajectory]`` and rebuild each
# trajectory into the record shape expected by the existing frontend:
#     {chat_completions, reward, solved, success, env_reward, uid}
#
# Use the environment's success flag for the frontend's binary result; keep the
# original cumulative reward in ``env_reward``.
# =============================================================================


def _traj_to_record(t):
    """Convert one rLLM Trajectory object into a frontend record dict.

    success source priority: trajectory.info['success'] -> last step.info['success'].
    The frontend-facing `reward` is success-based (1.0/0.0). The original
    cumulative env reward is preserved in `env_reward` (not overwritten).
    Returns (record, used_reward_fallback: bool).
    """
    info = getattr(t, 'info', None)
    info = info if isinstance(info, dict) else {}
    steps = list(getattr(t, 'steps', None) or [])

    # Full dialogue = the last (accumulated) step's chat_completions.
    chat = []
    for s in reversed(steps):
        cc = getattr(s, 'chat_completions', None)
        if cc:
            chat = cc
            break

    # success: trajectory-level first, else last step info.
    success = info.get('success', None)
    if success is None and steps:
        last_info = getattr(steps[-1], 'info', None)
        if isinstance(last_info, dict):
            success = last_info.get('success', None)

    env_reward = None
    try:
        if getattr(t, 'reward', None) is not None:
            env_reward = float(t.reward)
    except (TypeError, ValueError):
        env_reward = None

    used_fallback = False
    if success is not None:
        reward = 1.0 if success is True else 0.0
        solved = bool(success)
    else:
        # No success signal (should not happen for alfworld/webshop). Mirror
        # metrics.py: fall back to reward>0, and flag it for the operator.
        used_fallback = True
        reward = 1.0 if (env_reward is not None and env_reward > 0) else 0.0
        solved = reward == 1.0

    record = {
        'chat_completions': chat,
        'reward': reward,          # success-based; drives the frontend success/fail UI
        'solved': solved,
        'success': success,        # raw success flag (may be None)
        'env_reward': env_reward,  # original cumulative env reward, preserved
        'uid': getattr(t, 'uid', None),
    }
    return record, used_fallback


def load_pt(filepath):
    """Load a .pt trajectory file -> list of frontend record dicts.

    Accepts a `list[Trajectory]` (the rLLM eval output format). torch is
    imported lazily so the script still parses without torch installed.
    """
    import torch  # lazy import
    obj = torch.load(filepath, map_location='cpu', weights_only=False)
    if isinstance(obj, dict):
        # Be liberal: some dumps wrap the list under a key.
        for key in ('trajectories', 'data', 'results'):
            if key in obj and isinstance(obj[key], list):
                obj = obj[key]
                break
    if not isinstance(obj, (list, tuple)):
        raise ValueError(
            f"Unexpected .pt structure: top-level type is {type(obj).__name__}, "
            f"expected a list[Trajectory]."
        )
    records = []
    fallback_count = 0
    for t in obj:
        rec, fb = _traj_to_record(t)
        records.append(rec)
        if fb:
            fallback_count += 1
    if fallback_count:
        print(
            f"[warn] {fallback_count}/{len(records)} trajectories in "
            f"{os.path.basename(filepath)} lacked info['success']; used reward>0 fallback."
        )
    return records


def classify_file(fname):
    """Return 'train', 'val', or None for a given .pt filename.

    Validation outputs are named with a 'val' / 'val_' prefix; everything
    else is treated as an eval/train rollout dump. Only .pt files are accepted.
    Uses a prefix check (matching the frontend's isValFile startsWith('val_'))
    rather than a loose substring, so names like 'eval_*.pt' aren't misfiled.
    """
    if not fname.endswith('.pt'):
        return None
    stem = fname[:-len('.pt')].lower()
    return 'val' if (stem == 'val' or stem.startswith('val_')) else 'train'


def get_pt_files(directory):
    """List .pt files in a directory as [{'name', 'type', 'num'}].

    'train' (non-val) files first, then 'val' files; each group sorted by name.
    """
    train_files = []
    val_files = []
    for f in os.listdir(directory):
        kind = classify_file(f)
        if kind == 'train':
            train_files.append({'name': f, 'type': 'train', 'num': 0})
        elif kind == 'val':
            val_files.append({'name': f, 'type': 'val', 'num': 0})
    train_files.sort(key=lambda x: x['name'])
    val_files.sort(key=lambda x: x['name'])
    return train_files + val_files


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Text-Env Trajectory Visualizer</title>
<style>
:root, [data-theme="dark"] {
    --bg-primary: #0f1419;
    --bg-secondary: #1a2332;
    --bg-tertiary: #243447;
    --bg-card: #1e2d3d;
    --text-primary: #e8eaed;
    --text-secondary: #9aa0a6;
    --text-muted: #5f6368;
    --accent-blue: #4fc3f7;
    --accent-green: #81c784;
    --accent-orange: #ffb74d;
    --accent-red: #e57373;
    --accent-purple: #ce93d8;
    --accent-cyan: #4dd0e1;
    --border-color: #2d4156;
    --shadow: 0 4px 24px rgba(0,0,0,0.4);
    --radius: 12px;
    --radius-sm: 8px;
}

[data-theme="light"] {
    --bg-primary: #f5f7fa;
    --bg-secondary: #ffffff;
    --bg-tertiary: #edf0f5;
    --bg-card: #ffffff;
    --text-primary: #1a202c;
    --text-secondary: #4a5568;
    --text-muted: #a0aec0;
    --accent-blue: #2b6cb0;
    --accent-green: #2f855a;
    --accent-orange: #c05621;
    --accent-red: #c53030;
    --accent-purple: #805ad5;
    --accent-cyan: #0987a0;
    --border-color: #e2e8f0;
    --shadow: 0 4px 24px rgba(0,0,0,0.08);
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
    min-height: 100vh;
}

.header {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
    border-bottom: 1px solid var(--border-color);
    padding: 20px 32px;
    position: sticky;
    top: 0;
    z-index: 100;
}

.header-content {
    max-width: 1600px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
}

.header h1 {
    font-size: 1.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.badge {
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.badge-type { background: rgba(79, 195, 247, 0.15); color: var(--accent-cyan); border: 1px solid rgba(79, 195, 247, 0.3); }
.badge-count { background: rgba(129, 199, 132, 0.15); color: var(--accent-green); border: 1px solid rgba(129, 199, 132, 0.3); }
.badge-val { background: rgba(206, 147, 216, 0.15); color: var(--accent-purple); border: 1px solid rgba(206, 147, 216, 0.3); }

.controls {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-color);
    padding: 14px 32px;
    position: sticky;
    top: 72px;
    z-index: 99;
}

.controls-inner {
    max-width: 1600px;
    margin: 0 auto;
    display: flex;
    gap: 16px;
    align-items: center;
    flex-wrap: wrap;
}

.control-group { display: flex; align-items: center; gap: 8px; }
.control-group label {
    font-size: 0.82rem; font-weight: 600; color: var(--text-secondary);
    text-transform: uppercase; letter-spacing: 0.5px; white-space: nowrap;
}

select {
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    padding: 7px 12px;
    border-radius: var(--radius-sm);
    font-size: 0.88rem;
    cursor: pointer;
    transition: border-color 0.2s;
}

select:focus { outline: none; border-color: var(--accent-blue); box-shadow: 0 0 0 3px rgba(79, 195, 247, 0.15); }

.btn {
    padding: 7px 14px; border-radius: var(--radius-sm); font-size: 0.82rem;
    font-weight: 600; border: none; cursor: pointer; transition: all 0.2s;
    display: inline-flex; align-items: center; gap: 5px;
}

.btn-secondary { background: var(--bg-tertiary); color: var(--text-primary); border: 1px solid var(--border-color); }
.btn-secondary:hover { border-color: var(--accent-blue); background: rgba(79, 195, 247, 0.1); }

.btn-page { background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)); color: var(--bg-primary); font-weight: 700; }
.btn-page:hover { box-shadow: 0 2px 8px rgba(79,195,247,0.3); }
.btn-page:disabled { opacity: 0.4; cursor: not-allowed; box-shadow: none; }

.main { max-width: 1600px; margin: 0 auto; padding: 24px 32px; }

.stats-bar {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}

.stat-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: var(--radius); padding: 14px 16px; text-align: center; }
.stat-value { font-size: 1.6rem; font-weight: 700; color: var(--accent-blue); }
.stat-label { font-size: 0.72rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }

.pagination {
    display: flex; align-items: center; justify-content: center; gap: 12px;
    padding: 16px; margin-bottom: 16px; background: var(--bg-card);
    border: 1px solid var(--border-color); border-radius: var(--radius);
}
.page-info { font-size: 0.85rem; color: var(--text-secondary); }

.traj-card {
    background: var(--bg-card); border: 1px solid var(--border-color);
    border-radius: var(--radius); overflow: hidden; margin-bottom: 12px;
    transition: border-color 0.2s;
}
.traj-card:hover { border-color: rgba(79, 195, 247, 0.4); }

.traj-card-header {
    background: var(--bg-tertiary); padding: 12px 18px; display: flex;
    justify-content: space-between; align-items: center; cursor: pointer;
    border-bottom: 1px solid var(--border-color); user-select: none; flex-wrap: wrap; gap: 8px;
}
.traj-card-header:hover { background: rgba(79, 195, 247, 0.05); }

.traj-title { font-weight: 600; font-size: 0.9rem; display: flex; align-items: center; gap: 8px; }
.traj-meta { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }

.score-badge { padding: 3px 9px; border-radius: 12px; font-size: 0.73rem; font-weight: 700; }
.score-success { background: rgba(129, 199, 132, 0.2); color: var(--accent-green); }
.score-fail { background: rgba(229, 115, 115, 0.2); color: var(--accent-red); }
.score-partial { background: rgba(255, 183, 77, 0.2); color: var(--accent-orange); }

.solved-badge { padding: 3px 9px; border-radius: 12px; font-size: 0.7rem; font-weight: 700; }
.solved-yes { background: rgba(129, 199, 132, 0.2); color: var(--accent-green); }
.solved-no { background: rgba(95, 99, 104, 0.2); color: var(--text-muted); }

.turn-count { font-size: 0.78rem; color: var(--text-secondary); }
.env-reward { font-size: 0.72rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }

.traj-card-body { display: none; }
.traj-card-body.expanded { display: block; }

.turn-item { border-bottom: 1px solid var(--border-color); padding: 14px 18px; }
.turn-item:last-child { border-bottom: none; }

.turn-role-line { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }

.turn-role {
    display: inline-flex; align-items: center; gap: 5px; padding: 2px 8px;
    border-radius: 6px; font-size: 0.72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.role-system { background: rgba(206, 147, 216, 0.15); color: var(--accent-purple); }
.role-user { background: rgba(79, 195, 247, 0.15); color: var(--accent-blue); }
.role-assistant { background: rgba(129, 199, 132, 0.15); color: var(--accent-green); }

/* Content box: a bordered, scrollable container holding the message text. */
.content-box {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.8rem; line-height: 1.7; color: var(--text-primary);
    padding: 10px 14px; background: var(--bg-primary);
    border-radius: var(--radius-sm); border: 1px solid var(--border-color);
    max-height: 560px; overflow-y: auto;
}
.content-seg {
    white-space: pre-wrap; word-wrap: break-word; overflow-wrap: anywhere; margin: 0;
    font-family: inherit; font-size: inherit; line-height: inherit; color: inherit;
}

/* The action command the agent emitted (```...``` block). Rendered as a
   highlighted inline-block so it stands out from the surrounding reasoning. */
.action-block {
    display: inline-block; margin: 4px 0; padding: 4px 12px;
    background: rgba(255, 183, 77, 0.15); color: var(--accent-orange);
    border: 1px solid rgba(255, 183, 77, 0.45); border-radius: var(--radius-sm);
    font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.85rem;
    white-space: pre-wrap; word-wrap: break-word; overflow-wrap: anywhere;
}
/* Inline single/double-backtick spans (lighter than a full action block). */
.action-inline {
    background: rgba(79, 195, 247, 0.14); color: var(--accent-cyan);
    border-radius: 4px; padding: 0 4px;
    font-family: 'JetBrains Mono', monospace; font-weight: 600;
}
/* WebShop observations delimit fields with [SEP]; dim them for readability. */
.sep-tag { color: var(--text-muted); font-weight: 700; opacity: 0.7; }

/* Per-turn header badges */
.turn-tag {
    font-size: 0.68rem; font-weight: 700; color: var(--text-muted);
    background: rgba(95, 99, 104, 0.18); padding: 1px 7px; border-radius: 8px;
}
.action-badge {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 700;
    color: var(--accent-orange); background: rgba(255, 183, 77, 0.18);
    border: 1px solid rgba(255, 183, 77, 0.4); padding: 1px 8px; border-radius: 8px;
    max-width: 520px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

.prompt-group { margin-bottom: 14px; border: 1px solid var(--border-color); border-radius: var(--radius); overflow: hidden; }
.prompt-group-header {
    background: linear-gradient(135deg, var(--bg-tertiary), var(--bg-secondary));
    padding: 12px 18px; display: flex; justify-content: space-between; align-items: center;
    cursor: pointer; border-bottom: 1px solid var(--border-color); user-select: none; gap: 12px;
}
.prompt-group-header:hover { background: linear-gradient(135deg, rgba(79,195,247,0.08), var(--bg-secondary)); }
.prompt-group-body { display: none; padding: 10px; }
.prompt-group-body.expanded { display: block; }
.prompt-preview {
    color: var(--text-muted); font-size: 0.74rem; font-family: 'JetBrains Mono', monospace;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 620px; flex: 1;
}

.collapse-arrow { transition: transform 0.3s; color: var(--text-secondary); font-size: 0.8rem; }
.collapse-arrow.rotated { transform: rotate(90deg); }

.loading { display: flex; align-items: center; justify-content: center; padding: 60px; color: var(--text-secondary); font-size: 1rem; }
.spinner { width: 20px; height: 20px; border: 3px solid var(--border-color); border-top-color: var(--accent-blue); border-radius: 50%; animation: spin 0.8s linear infinite; margin-right: 10px; }
@keyframes spin { to { transform: rotate(360deg); } }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--bg-tertiary); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-blue); }

.system-toggle { display: flex; align-items: center; gap: 5px; font-size: 0.78rem; color: var(--text-secondary); cursor: pointer; }
.system-toggle input { cursor: pointer; }
</style>
</head>
<body>

<div class="header">
    <div class="header-content">
        <h1>&#128172; Text-Env Trajectory Visualizer</h1>
        <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
            <span class="badge badge-type" id="file-type-badge">__TYPE_LABEL__</span>
            <span class="badge badge-count">__TRAIN_COUNT__ train</span>
            <span class="badge badge-val">__VAL_COUNT__ val</span>
            <button class="btn btn-secondary" onclick="toggleTheme()" id="theme-btn" title="Toggle theme">&#127769; Dark</button>
        </div>
    </div>
</div>

<div class="controls">
    <div class="controls-inner">
        <div class="control-group">
            <label>File:</label>
            <select id="file-select" onchange="loadFile()">__FILE_OPTIONS__</select>
        </div>
        <div class="control-group">
            <label>Score:</label>
            <select id="score-filter" onchange="onFilterChange()">
                <option value="all">All</option>
                <option value="success">Success (1.0)</option>
                <option value="fail">Fail (0.0)</option>
            </select>
        </div>
        <div class="control-group">
            <label>Min Turns:</label>
            <input type="number" id="min-turns" min="1" value="1" style="width:60px;background:var(--bg-tertiary);border:1px solid var(--border-color);color:var(--text-primary);padding:7px 8px;border-radius:var(--radius-sm);font-size:0.88rem;" onchange="onFilterChange()">
        </div>
        <div class="control-group">
            <label>View:</label>
            <select id="group-mode" onchange="onFilterChange()">
                <option value="prompt">Group by Prompt</option>
                <option value="flat">Flat List</option>
            </select>
        </div>
        <div class="control-group">
            <label>Per Page:</label>
            <select id="page-size" onchange="onFilterChange()">
                <option value="10">10</option>
                <option value="20" selected>20</option>
                <option value="50">50</option>
                <option value="100">100</option>
            </select>
        </div>
        <div class="control-group">
            <label class="system-toggle">
                <input type="checkbox" id="hide-system" onchange="onFilterChange()"> Hide System
            </label>
        </div>
        <div class="control-group" style="margin-left:auto;">
            <button class="btn btn-secondary" onclick="expandAll()">Expand All</button>
            <button class="btn btn-secondary" onclick="collapseAll()">Collapse All</button>
        </div>
    </div>
</div>

<div class="main" id="main-content">
    <div class="loading"><div class="spinner"></div>Loading...</div>
</div>

<script>
let currentData = [];
let filteredData = [];
let currentPage = 0;
const fileCache = {};

function isValFile(name) { return name.startsWith('val_'); }

async function loadFile() {
    const file = document.getElementById('file-select').value;
    const mainEl = document.getElementById('main-content');

    // Update file-type badge
    const badge = document.getElementById('file-type-badge');
    badge.textContent = isValFile(file) ? 'Validation' : 'Training';

    if (fileCache[file]) {
        currentData = fileCache[file];
        onFilterChange();
        return;
    }

    mainEl.innerHTML = '<div class="loading"><div class="spinner"></div>Loading ' + file + '...</div>';

    try {
        const resp = await fetch('/api/load?file=' + encodeURIComponent(file));
        if (!resp.ok) {
            const msg = await resp.text();
            mainEl.innerHTML = '<div class="loading">Error loading ' + escapeHtml(file) + ': ' + escapeHtml(msg) + '</div>';
            return;
        }
        const data = await resp.json();
        fileCache[file] = data;
        currentData = data;
        onFilterChange();
    } catch(e) {
        mainEl.innerHTML = '<div class="loading">Error loading: ' + escapeHtml(e.message) + '</div>';
    }
}

function getReward(record) {
    const r = record.reward;
    return (typeof r === 'number') ? r : 0.0;
}

function getChat(record) {
    return Array.isArray(record.chat_completions) ? record.chat_completions : [];
}

function getTurnCount(record) {
    return getChat(record).filter(m => m.role === 'assistant').length;
}

// Extract the action command an assistant turn took. Agents emit their final
// action inside a ```...``` block (e.g. ```go to desk 1```, ```search[shoes]```,
// ```click[Buy Now]```). Prefer the LAST triple-backtick block (the final
// decision); fall back to the last single/double-backtick inline span.
function extractAction(content) {
    const text = String(content);
    const fenced = /`{3}\s*([\s\S]*?)\s*`{3}/g;
    let m, last = null;
    while ((m = fenced.exec(text)) !== null) last = m[1];
    if (last !== null) return last.trim();
    // Fallback: last single/double-backtick inline span on one line.
    const inline = /`{1,2}\s*([^`\n]+?)\s*`{1,2}/g;
    while ((m = inline.exec(text)) !== null) last = m[1];
    return last !== null ? last.trim() : null;
}

function getPromptKey(record) {
    // Group key = first user message content (initial observation / instruction).
    const cc = getChat(record);
    for (const m of cc) {
        if (m.role === 'user') return m.content || '';
    }
    return '';
}

function onFilterChange() {
    const scoreFilter = document.getElementById('score-filter').value;
    const minTurns = parseInt(document.getElementById('min-turns').value) || 1;
    filteredData = currentData;
    if (scoreFilter === 'success') filteredData = filteredData.filter(d => getReward(d) === 1.0);
    else if (scoreFilter === 'fail') filteredData = filteredData.filter(d => getReward(d) === 0.0);
    if (minTurns > 1) filteredData = filteredData.filter(d => getTurnCount(d) >= minTurns);
    currentPage = 0;
    renderPage();
}

function renderPage() {
    const groupMode = document.getElementById('group-mode').value;
    const pageSize = parseInt(document.getElementById('page-size').value);
    const mainEl = document.getElementById('main-content');

    // Stats are always computed over the full filtered set.
    const totalItems = filteredData.length;
    const successCount = filteredData.filter(d => getReward(d) === 1.0).length;
    const failCount = filteredData.filter(d => getReward(d) === 0.0).length;
    const avgReward = filteredData.length > 0 ? (filteredData.reduce((s, d) => s + getReward(d), 0) / filteredData.length).toFixed(3) : '0';

    // Build the paginated unit list. In group mode we group the ENTIRE
    // filtered set first (so a prompt's rollouts never split across a page),
    // then paginate over groups. In flat mode we paginate over records.
    let totalUnits, pageLabel;
    let allGroups = [];
    if (groupMode === 'prompt') {
        const groups = {};
        const groupOrder = [];
        filteredData.forEach((record, idx) => {
            const key = getPromptKey(record);
            if (!groups[key]) { groups[key] = []; groupOrder.push(key); }
            groups[key].push({rec: record, gidx: idx});
        });
        allGroups = groupOrder.map(k => groups[k]);
        totalUnits = allGroups.length;
        pageLabel = 'prompts';
    } else {
        totalUnits = totalItems;
        pageLabel = 'trajectories';
    }

    const totalPages = Math.max(1, Math.ceil(totalUnits / pageSize));
    if (currentPage >= totalPages) currentPage = totalPages - 1;
    if (currentPage < 0) currentPage = 0;
    const start = currentPage * pageSize;
    const end = Math.min(start + pageSize, totalUnits);

    let html = `
    <div class="stats-bar">
        <div class="stat-card"><div class="stat-value">${totalItems}</div><div class="stat-label">Trajectories</div></div>
        <div class="stat-card"><div class="stat-value" style="color:var(--accent-green)">${successCount}</div><div class="stat-label">Success</div></div>
        <div class="stat-card"><div class="stat-value" style="color:var(--accent-red)">${failCount}</div><div class="stat-label">Failed</div></div>
        <div class="stat-card"><div class="stat-value" style="color:var(--accent-orange)">${avgReward}</div><div class="stat-label">Success Rate</div></div>
    </div>
    <div class="pagination">
        <button class="btn btn-page" onclick="goPage(-1)" ${currentPage===0?'disabled':''}>&#9664; Prev</button>
        <span class="page-info">Page ${currentPage+1} / ${totalPages} (showing ${totalUnits?start+1:0}-${end} of ${totalUnits} ${pageLabel})</span>
        <button class="btn btn-page" onclick="goPage(1)" ${currentPage>=totalPages-1?'disabled':''}>Next &#9654;</button>
    </div>`;

    if (groupMode === 'prompt') {
        const pageGroups = allGroups.slice(start, end);
        pageGroups.forEach((entries, pi) => {
            const gi = start + pi;
            const rewards = entries.map(e => getReward(e.rec));
            const avg = (rewards.reduce((a,b)=>a+b,0)/rewards.length).toFixed(2);
            const succ = rewards.filter(s=>s===1.0).length;
            const cls = succ===entries.length?'score-success':succ>0?'score-partial':'score-fail';
            const preview = escapeHtml(getPromptKey(entries[0].rec).replace(/\s+/g, ' ').slice(0, 160));

            html += `<div class="prompt-group">
                <div class="prompt-group-header" onclick="toggleGroup(this)">
                    <div style="display:flex;align-items:center;gap:8px;min-width:0;flex:1;">
                        <span class="collapse-arrow">&#9654;</span>
                        <span style="font-weight:600;white-space:nowrap;">Prompt #${gi+1}</span>
                        <span style="color:var(--text-muted);font-size:0.78rem;white-space:nowrap;">(${entries.length} rollout${entries.length>1?'s':''})</span>
                        <span class="prompt-preview">${preview}</span>
                    </div>
                    <span class="score-badge ${cls}" style="white-space:nowrap;">${succ}/${entries.length} solved | avg ${avg}</span>
                </div>
                <div class="prompt-group-body">`;
            entries.forEach((e, ri) => { html += renderCard(e.rec, `G${gi}R${ri}`, e.gidx); });
            html += `</div></div>`;
        });
    } else {
        const pageData = filteredData.slice(start, end);
        pageData.forEach((record, idx) => {
            html += renderCard(record, `T${start + idx}`, start + idx);
        });
    }

    html += `<div class="pagination" style="margin-top:16px;">
        <button class="btn btn-page" onclick="goPage(-1)" ${currentPage===0?'disabled':''}>&#9664; Prev</button>
        <span class="page-info">Page ${currentPage+1} / ${totalPages}</span>
        <button class="btn btn-page" onclick="goPage(1)" ${currentPage>=totalPages-1?'disabled':''}>Next &#9654;</button>
    </div>`;

    mainEl.innerHTML = html;
    window.scrollTo(0, 0);
}

function goPage(delta) { currentPage += delta; renderPage(); }

function renderCard(record, id, globalIdx) {
    const reward = getReward(record);
    const turnCount = getTurnCount(record);
    const scoreClass = reward === 1.0 ? 'score-success' : reward === 0.0 ? 'score-fail' : 'score-partial';
    const scoreIcon = reward === 1.0 ? '&#9989;' : reward === 0.0 ? '&#10060;' : '&#128310;';
    const solved = record.solved === true;
    const solvedBadge = `<span class="solved-badge ${solved?'solved-yes':'solved-no'}">${solved?'solved':'unsolved'}</span>`;

    // Show the original cumulative env reward (distinct from success-based score).
    let envRewardBadge = '';
    if (typeof record.env_reward === 'number') {
        envRewardBadge = `<span class="env-reward">env R: ${record.env_reward.toFixed(2)}</span>`;
    }

    // Store chat_completions JSON (base64) for lazy render.
    const payload = btoa(unescape(encodeURIComponent(JSON.stringify({
        chat: getChat(record),
    }))));

    return `
    <div class="traj-card" id="card-${id}" data-raw="${payload}" data-rendered="0">
        <div class="traj-card-header" onclick="toggleCard('${id}')">
            <div class="traj-title">
                <span class="collapse-arrow">&#9654;</span>
                <span>${scoreIcon} #${globalIdx + 1}</span>
            </div>
            <div class="traj-meta">
                <span class="turn-count">${turnCount} turns</span>
                ${envRewardBadge}
                ${solvedBadge}
                <span class="score-badge ${scoreClass}">Score: ${reward}</span>
            </div>
        </div>
        <div class="traj-card-body" id="body-${id}"></div>
    </div>`;
}

function toggleCard(id) {
    const card = document.getElementById('card-' + id);
    const body = document.getElementById('body-' + id);
    const arrow = card.querySelector('.traj-card-header .collapse-arrow');
    const isExpanding = !body.classList.contains('expanded');

    if (isExpanding && card.dataset.rendered === '0') {
        const payload = JSON.parse(decodeURIComponent(escape(atob(card.dataset.raw))));
        const chat = payload.chat || [];
        const hideSystem = document.getElementById('hide-system').checked;

        let html = '';
        let asstIdx = 0;
        chat.forEach((msg) => {
            const role = msg.role || 'unknown';
            const hidden = (hideSystem && role === 'system') ? ' style="display:none"' : '';
            const roleIcon = role === 'system' ? '&#9881;&#65039;' : role === 'user' ? '&#128100;' : '&#129302;';
            let actionBadge = '';
            let turnTag = '';
            if (role === 'assistant') {
                turnTag = `<span class="turn-tag">Turn ${asstIdx + 1}</span>`;
                const act = extractAction(msg.content || '');
                if (act) actionBadge = `<span class="action-badge">&#9654; ${escapeHtml(act)}</span>`;
                asstIdx++;
            }
            html += `<div class="turn-item" data-role="${role}"${hidden}>
                <div class="turn-role-line">
                    <span class="turn-role role-${role}">${roleIcon} ${escapeHtml(role)}</span>
                    ${turnTag}
                    ${actionBadge}
                </div>
                ${formatContent(msg.content || '', role)}
            </div>`;
        });

        body.innerHTML = html;
        card.dataset.rendered = '1';
    }

    body.classList.toggle('expanded');
    arrow.classList.toggle('rotated');
}

function toggleGroup(el) {
    const body = el.nextElementSibling;
    const arrow = el.querySelector('.collapse-arrow');
    body.classList.toggle('expanded');
    arrow.classList.toggle('rotated');
}

function escapeHtml(t) {
    return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// Render a message's content inside a single scrollable content box. The text is
// HTML-escaped first, then the agent's action commands are highlighted:
//   - ```...``` triple-backtick blocks -> prominent .action-block
//   - `...` / ``...`` inline spans      -> lighter .action-inline
// WebShop's [SEP] field delimiters are dimmed for readability. Highlighting runs
// on escaped text, and the literal markers ([SEP], backticks) never appear inside
// the injected <span> tags, so the replacements can't corrupt each other.
function formatContent(content, role) {
    const text = String(content);
    if (!text.trim()) return '';
    let escaped = escapeHtml(text);
    // 1) Triple-backtick action blocks (the agent's final action).
    escaped = escaped.replace(/`{3}\s*([\s\S]*?)\s*`{3}/g,
        (m, a) => '<span class="action-block">' + a.trim() + '</span>');
    // 2) Remaining single/double-backtick inline spans (kept on one line).
    escaped = escaped.replace(/`{1,2}\s*([^`\n]+?)\s*`{1,2}/g,
        (m, a) => '<span class="action-inline">' + a.trim() + '</span>');
    // 3) Dim WebShop [SEP] delimiters (literal, not regex-special beyond escaping).
    escaped = escaped.replace(/\[SEP\]/g, '<span class="sep-tag">[SEP]</span>');
    return '<div class="content-box"><pre class="content-seg">' + escaped + '</pre></div>';
}

function expandAll() {
    document.querySelectorAll('.traj-card').forEach(card => {
        const id = card.id.replace('card-','');
        const body = document.getElementById('body-' + id);
        if (body && !body.classList.contains('expanded')) toggleCard(id);
    });
    document.querySelectorAll('.prompt-group-body').forEach(el => el.classList.add('expanded'));
    document.querySelectorAll('.prompt-group-header .collapse-arrow').forEach(el => el.classList.add('rotated'));
}

function collapseAll() {
    document.querySelectorAll('.traj-card-body').forEach(el => el.classList.remove('expanded'));
    document.querySelectorAll('.prompt-group-body').forEach(el => el.classList.remove('expanded'));
    document.querySelectorAll('.collapse-arrow').forEach(el => el.classList.remove('rotated'));
}

function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    updateThemeBtn(next);
}

function updateThemeBtn(theme) {
    const btn = document.getElementById('theme-btn');
    btn.innerHTML = theme === 'dark' ? '&#127769; Dark' : '&#9728;&#65039; Light';
}

document.addEventListener('DOMContentLoaded', function() {
    const theme = '__INITIAL_THEME__';
    document.documentElement.setAttribute('data-theme', theme);
    updateThemeBtn(theme);
    loadFile();
});
</script>
</body>
</html>"""


def create_handler(directory, valid_names, html_content):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)

            if parsed.path == '/' or parsed.path == '/index.html':
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(html_content.encode('utf-8'))

            elif parsed.path == '/api/load':
                params = parse_qs(parsed.query)
                filename = params.get('file', [''])[0]
                if filename not in valid_names:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'File not found')
                    return

                filepath = os.path.join(directory, filename)
                try:
                    data = load_pt(filepath)
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'text/plain; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(f"Error loading {filename}: {e}".encode('utf-8'))
                    return

                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'Not Found')

        def log_message(self, format, *args):
            pass

    return Handler


def build_file_options(files):
    """Build grouped <optgroup> options for train and val files."""
    train = [f for f in files if f['type'] == 'train']
    val = [f for f in files if f['type'] == 'val']
    parts = []
    if train:
        parts.append('<optgroup label="Training">')
        parts += [f'<option value="{f["name"]}">{f["name"]}</option>' for f in train]
        parts.append('</optgroup>')
    if val:
        parts.append('<optgroup label="Validation">')
        parts += [f'<option value="{f["name"]}">{f["name"]}</option>' for f in val]
        parts.append('</optgroup>')
    return ''.join(parts)


def main():
    parser = argparse.ArgumentParser(description='Text-Env Trajectory Visualizer (.pt) — alfworld / webshop')
    parser.add_argument('--path', required=True,
                        help='Path to a .pt trajectory file OR a directory containing .pt files')
    parser.add_argument('--port', type=int, default=8767, help='Port (default: 8767)')
    parser.add_argument('--theme', choices=['dark', 'light'], default='dark', help='Color theme (default: dark)')
    parser.add_argument('--no-browser', action='store_true', help='Do not auto-open browser')
    args = parser.parse_args()

    target = os.path.abspath(args.path)

    # Accept either a single .pt file or a directory of .pt files.
    if os.path.isfile(target):
        if not target.endswith('.pt'):
            print(f"Error: {target} is not a .pt file")
            return
        directory = os.path.dirname(target)
        files = [{'name': os.path.basename(target),
                  'type': classify_file(os.path.basename(target)) or 'train',
                  'num': 0}]
    elif os.path.isdir(target):
        directory = target
        files = get_pt_files(directory)
        if not files:
            print(f"Error: No .pt files found in {directory}")
            return
    else:
        print(f"Error: {target} is neither a file nor a directory")
        return

    valid_names = {f['name'] for f in files}
    train_files = [f for f in files if f['type'] == 'train']
    val_files = [f for f in files if f['type'] == 'val']

    # Default type label reflects the first file in the dropdown.
    type_label = "Training" if (train_files or not val_files) else "Validation"

    file_options = build_file_options(files)
    html_content = HTML_TEMPLATE.replace('__TYPE_LABEL__', type_label)
    html_content = html_content.replace('__TRAIN_COUNT__', str(len(train_files)))
    html_content = html_content.replace('__VAL_COUNT__', str(len(val_files)))
    html_content = html_content.replace('__FILE_OPTIONS__', file_options)
    html_content = html_content.replace('__INITIAL_THEME__', args.theme)

    print("\U0001F4AC Text-Env Trajectory Visualizer (.pt)")
    print(f"{'-' * 44}")
    print(f"\U0001F4C2 {directory}")
    print(f"\U0001F4C4 {len(files)} .pt files  ({len(train_files)} train, {len(val_files)} val)")
    if train_files:
        print(f"   train: {train_files[0]['name']} -> {train_files[-1]['name']}")
    if val_files:
        print(f"   val:   {val_files[0]['name']} -> {val_files[-1]['name']}")
    print(f"{'-' * 44}")

    handler = create_handler(directory, valid_names, html_content)
    server = HTTPServer(('0.0.0.0', args.port), handler)

    url = f"http://localhost:{args.port}"
    print(f"\U0001F310 Server: {url}")
    print(f"   Ctrl+C to stop\n")

    if not args.no_browser:
        import webbrowser, threading
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\U0001F44B Stopped.")
        server.shutdown()


if __name__ == '__main__':
    main()
