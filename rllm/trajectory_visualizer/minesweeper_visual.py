#!/usr/bin/env python3
"""
Minesweeper Trajectory Visualizer
=================================
A web-based viewer for Minesweeper RL rollout/validation trajectories
(rLLM chat_completions format).

Serves BOTH training files (e.g. 1.jsonl, 2.jsonl, ...) and validation
files (e.g. val_0.jsonl, val_20.jsonl, ...) from a single chat_completions
directory. Uses a local HTTP server with on-demand file loading, pagination,
lazy-rendered cards, group-by-prompt, theme toggle, and per-turn advantage
display for training rollouts.

Usage:
    python3 minesweeper_visual.py --path <chat_completions_dir> [--port 8765]

Example:
    python3 minesweeper_visual.py --path .../chat_completions
"""

import argparse
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse


def load_jsonl(filepath):
    """Load a jsonl file and return list of records (parsed JSON dicts)."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def classify_file(fname):
    """Return 'train', 'val', or None for a given .jsonl filename.

    - Training files have a pure-integer stem: '1.jsonl', '23.jsonl'.
    - Validation files start with 'val_' and end with an integer: 'val_0.jsonl'.
    """
    if not fname.endswith('.jsonl'):
        return None
    stem = fname[:-len('.jsonl')]
    if stem.isdigit():
        return 'train'
    if stem.startswith('val_') and stem[len('val_'):].isdigit():
        return 'val'
    return None


def get_jsonl_files(directory):
    """Return training and validation JSONL files in numeric order.

    Returns a list of dicts: {'name': filename, 'type': 'train'|'val', 'num': int}.
    Train files are sorted by integer stem ascending, followed by val files
    sorted by their integer suffix ascending.
    """
    train_files = []
    val_files = []
    for f in os.listdir(directory):
        kind = classify_file(f)
        if kind == 'train':
            train_files.append({'name': f, 'type': 'train', 'num': int(f[:-len('.jsonl')])})
        elif kind == 'val':
            val_files.append({'name': f, 'type': 'val', 'num': int(f[len('val_'):-len('.jsonl')])})
    train_files.sort(key=lambda x: x['num'])
    val_files.sort(key=lambda x: x['num'])
    return train_files + val_files


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Minesweeper Trajectory Visualizer</title>
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
    --cell-unrevealed: #37474f;
    --cell-revealed: #11181f;
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
    --cell-unrevealed: #c4cad6;
    --cell-revealed: #eef1f6;
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

.adv-strip {
    font-size: 0.7rem; color: var(--text-secondary); padding: 6px 18px;
    background: var(--bg-secondary); border-bottom: 1px solid var(--border-color);
    display: flex; gap: 6px; flex-wrap: wrap; align-items: center;
}
.adv-chip { padding: 1px 6px; border-radius: 8px; font-weight: 700; font-size: 0.68rem; font-family: monospace; }
.adv-pos { background: rgba(129, 199, 132, 0.2); color: var(--accent-green); }
.adv-neg { background: rgba(229, 115, 115, 0.2); color: var(--accent-red); }
.adv-zero { background: rgba(95, 99, 104, 0.18); color: var(--text-muted); }

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

.content-pre {
    white-space: pre-wrap; word-wrap: break-word;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.8rem; line-height: 1.7; color: var(--text-primary);
    padding: 10px 14px; background: var(--bg-primary);
    border-radius: var(--radius-sm); border: 1px solid var(--border-color);
    max-height: 520px; overflow-y: auto; margin: 0;
}

/* Unified content box: text segments and boards flow continuously inside a
   single bordered/scrollable container instead of separate boxes. */
.content-box {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.8rem; line-height: 1.7; color: var(--text-primary);
    padding: 10px 14px; background: var(--bg-primary);
    border-radius: var(--radius-sm); border: 1px solid var(--border-color);
    max-height: 560px; overflow-y: auto;
}
.content-seg {
    white-space: pre-wrap; word-wrap: break-word; margin: 0;
    font-family: inherit; font-size: inherit; line-height: inherit; color: inherit;
}

.action-block {
    display: inline-block; margin: 4px 0; padding: 4px 12px;
    background: rgba(255, 183, 77, 0.15); color: var(--accent-orange);
    border: 1px solid rgba(255, 183, 77, 0.45); border-radius: var(--radius-sm);
    font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.85rem;
}

/* Per-turn header badges */
.turn-tag {
    font-size: 0.68rem; font-weight: 700; color: var(--text-muted);
    background: rgba(95, 99, 104, 0.18); padding: 1px 7px; border-radius: 8px;
}
.action-badge {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 700;
    color: var(--accent-orange); background: rgba(255, 183, 77, 0.18);
    border: 1px solid rgba(255, 183, 77, 0.4); padding: 1px 8px; border-radius: 8px;
}

/* Minesweeper grid — block-level so it sits naturally on its own line(s)
   inside the unified content box, with text flowing above and below. */
.ms-grid {
    display: block; width: -moz-fit-content; width: fit-content;
    background: var(--bg-secondary);
    border-radius: var(--radius-sm); padding: 6px; margin: 8px 0;
    border: 1px solid var(--border-color); line-height: 1;
}
.ms-row { display: flex; }
.ms-cell {
    width: 26px; height: 26px; display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700; font-family: 'JetBrains Mono', monospace;
    border: 1px solid var(--border-color); margin: 0;
}
.ms-head { background: var(--bg-tertiary); color: var(--text-secondary); font-weight: 700; }
.ms-unrevealed { background: var(--cell-unrevealed); color: var(--text-muted); }
.ms-revealed { background: var(--cell-revealed); }
.ms-flag { background: rgba(255, 183, 77, 0.25); }
.ms-mine { background: rgba(229, 115, 115, 0.35); }
.n1 { color: #4fc3f7; } .n2 { color: #81c784; } .n3 { color: #e57373; }
.n4 { color: #ce93d8; } .n5 { color: #ffb74d; } .n6 { color: #4dd0e1; }
.n7 { color: #f48fb1; } .n8 { color: #bdbdbd; }

.prompt-group { margin-bottom: 14px; border: 1px solid var(--border-color); border-radius: var(--radius); overflow: hidden; }
.prompt-group-header {
    background: linear-gradient(135deg, var(--bg-tertiary), var(--bg-secondary));
    padding: 12px 18px; display: flex; justify-content: space-between; align-items: center;
    cursor: pointer; border-bottom: 1px solid var(--border-color); user-select: none;
}
.prompt-group-header:hover { background: linear-gradient(135deg, rgba(79,195,247,0.08), var(--bg-secondary)); }
.prompt-group-body { display: none; padding: 10px; }
.prompt-group-body.expanded { display: block; }

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
        <h1>&#128163; Minesweeper Trajectory Visualizer</h1>
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
        const data = await resp.json();
        fileCache[file] = data;
        currentData = data;
        onFilterChange();
    } catch(e) {
        mainEl.innerHTML = '<div class="loading">Error loading: ' + e.message + '</div>';
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

// Extract the action an assistant turn took. The model emits exactly one
// action inside a ``` block (e.g. ```reveal 3 2``` or ```flag 0 4```); we
// prefer the LAST such occurrence (the final decision), falling back to the
// last bare "reveal R C" / "flag R C" anywhere in the text.
function extractAction(content) {
    const text = String(content);
    const fenced = /```\s*((?:reveal|flag)\s+\d+\s+\d+)\s*```/gi;
    let m, last = null;
    while ((m = fenced.exec(text)) !== null) last = m[1];
    if (last) return last.replace(/\s+/g, ' ').trim();
    const bare = /(?:reveal|flag)\s+\d+\s+\d+/gi;
    let b, lastBare = null;
    while ((b = bare.exec(text)) !== null) lastBare = b[0];
    return lastBare ? lastBare.replace(/\s+/g, ' ').trim() : null;
}

function getPromptKey(record) {
    // Group key = first user message content (initial board observation).
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
    if (groupMode === 'prompt') {
        const groups = {};
        const groupOrder = [];
        filteredData.forEach((record, idx) => {
            const key = getPromptKey(record);
            if (!groups[key]) { groups[key] = []; groupOrder.push(key); }
            groups[key].push({rec: record, gidx: idx});
        });
        var allGroups = groupOrder.map(k => groups[k]);
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
        <div class="stat-card"><div class="stat-value" style="color:var(--accent-orange)">${avgReward}</div><div class="stat-label">Avg Reward</div></div>
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

            html += `<div class="prompt-group">
                <div class="prompt-group-header" onclick="toggleGroup(this)">
                    <div style="display:flex;align-items:center;gap:8px;">
                        <span class="collapse-arrow">&#9654;</span>
                        <span style="font-weight:600;">Prompt #${gi+1}</span>
                        <span style="color:var(--text-muted);font-size:0.78rem">(${entries.length} rollout${entries.length>1?'s':''})</span>
                    </div>
                    <span class="score-badge ${cls}">${succ}/${entries.length} solved | avg ${avg}</span>
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

function advClass(v) { return v > 1e-9 ? 'adv-pos' : (v < -1e-9 ? 'adv-neg' : 'adv-zero'); }

function renderCard(record, id, globalIdx) {
    const reward = getReward(record);
    const turnCount = getTurnCount(record);
    const scoreClass = reward === 1.0 ? 'score-success' : reward === 0.0 ? 'score-fail' : 'score-partial';
    const scoreIcon = reward === 1.0 ? '&#9989;' : reward === 0.0 ? '&#10060;' : '&#128310;';
    const solved = record.solved === true;
    const solvedBadge = `<span class="solved-badge ${solved?'solved-yes':'solved-no'}">${solved?'solved':'unsolved'}</span>`;

    // Compact advantage strip (training only).
    const turnAdvs = Array.isArray(record.turn_advantages_raw) ? record.turn_advantages_raw : null;
    let advStrip = '';
    if (turnAdvs && turnAdvs.length) {
        let chips = '';
        turnAdvs.forEach((v, i) => {
            chips += `<span class="adv-chip ${advClass(v)}">T${i+1}:${v.toFixed(3)}</span>`;
        });
        let tokSummary = '';
        const tok = record.token_advantages_final;
        if (Array.isArray(tok) && tok.length) {
            let mn = Infinity, mx = -Infinity, sum = 0;
            for (const x of tok) { if (x < mn) mn = x; if (x > mx) mx = x; sum += x; }
            tokSummary = `<span style="color:var(--text-muted);margin-left:8px;">token_adv n=${tok.length} min=${mn.toFixed(3)} mean=${(sum/tok.length).toFixed(3)} max=${mx.toFixed(3)}</span>`;
        }
        advStrip = `<div class="adv-strip"><span style="font-weight:700;color:var(--text-muted);">Turn adv:</span>${chips}${tokSummary}</div>`;
    }

    // Store chat_completions JSON (base64) for lazy render.
    const payload = btoa(unescape(encodeURIComponent(JSON.stringify({
        chat: getChat(record),
        turn_advantages: turnAdvs,
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
                ${solvedBadge}
                <span class="score-badge ${scoreClass}">Score: ${reward}</span>
            </div>
        </div>
        ${advStrip}
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
        const turnAdvs = payload.turn_advantages;
        const hideSystem = document.getElementById('hide-system').checked;

        let html = '';
        let asstIdx = 0;
        chat.forEach((msg) => {
            const role = msg.role || 'unknown';
            const hidden = (hideSystem && role === 'system') ? ' style="display:none"' : '';
            const roleIcon = role === 'system' ? '&#9881;&#65039;' : role === 'user' ? '&#128100;' : '&#129302;';
            let advBadge = '';
            let actionBadge = '';
            let turnTag = '';
            if (role === 'assistant') {
                turnTag = `<span class="turn-tag">Turn ${asstIdx + 1}</span>`;
                const act = extractAction(msg.content || '');
                if (act) actionBadge = `<span class="action-badge">&#9654; ${escapeHtml(act)}</span>`;
                if (turnAdvs && asstIdx < turnAdvs.length) {
                    const v = turnAdvs[asstIdx];
                    advBadge = `<span class="adv-chip ${advClass(v)}">adv ${v.toFixed(4)}</span>`;
                }
                asstIdx++;
            }
            html += `<div class="turn-item" data-role="${role}"${hidden}>
                <div class="turn-role-line">
                    <span class="turn-role role-${role}">${roleIcon} ${escapeHtml(role)}</span>
                    ${turnTag}
                    ${actionBadge}
                    ${advBadge}
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

// Detect contiguous minesweeper board blocks and render them inline inside a
// single continuous content box, so the grid sits naturally within the text
// flow (text above, board, text below) rather than in a separate box.
// Lone ``` fence lines that merely wrap a board are dropped so the result
// reads continuously.
function formatContent(content, role) {
    const lines = String(content).split('\n');
    const headerRe = /^\s+0(\s+\d+)+\s*$/;           // column header: "   0  1  2 ..."
    const rowRe = /^\s*\d+(\s+[.F0-8*])+\s*$/;        // board row: " 0  .  1  0 ..."
    const fenceRe = /^\s*```\s*$/;                    // lone code-fence line

    let segs = '';
    let textBuf = [];
    let i = 0;

    function flushText() {
        if (textBuf.length) {
            segs += renderTextBlock(textBuf.join('\n'), role);
            textBuf = [];
        }
    }

    while (i < lines.length) {
        const line = lines[i];
        const hasHeaderBoard = headerRe.test(line) && i + 1 < lines.length && rowRe.test(lines[i + 1]);
        // Headerless board: a run of >=2 consecutive board rows (model sometimes
        // reprints a board without the column-index header). Require at least one
        // genuine board glyph (. F or *) so we don't mistake numeric prose lists
        // (e.g. two lines of "1  2  3") for a board.
        const headerlessBoard = !hasHeaderBoard && rowRe.test(line) && i + 1 < lines.length
            && rowRe.test(lines[i + 1]) && /[.F*]/.test(line + lines[i + 1]);
        if (hasHeaderBoard || headerlessBoard) {
            // A board starts here. Drop a trailing lone ``` fence that opened it.
            if (textBuf.length && fenceRe.test(textBuf[textBuf.length - 1])) {
                textBuf.pop();
            }
            flushText();
            const boardLines = [];
            let j = i;
            if (hasHeaderBoard) { boardLines.push(line); j = i + 1; }
            while (j < lines.length && rowRe.test(lines[j])) {
                boardLines.push(lines[j]);
                j++;
            }
            segs += renderBoard(boardLines, hasHeaderBoard);
            // Skip a lone ``` fence that closed the board.
            if (j < lines.length && fenceRe.test(lines[j])) j++;
            i = j;
        } else {
            textBuf.push(line);
            i++;
        }
    }
    flushText();
    if (!segs) return '';
    return '<div class="content-box">' + segs + '</div>';
}

// Render reasoning text as a continuous inline segment (no own border/box).
// For any turn, highlight standalone action commands (```reveal R C``` /
// ```flag R C``` or bare). Leading/trailing blank lines are trimmed so
// segments butt cleanly against adjacent boards.
function renderTextBlock(text, role) {
    const trimmed = text.replace(/^\n+/, '').replace(/\n+$/, '');
    if (!trimmed.trim()) return '';
    let escaped = escapeHtml(trimmed);
    escaped = escaped.replace(/`{0,3}\s*((?:reveal|flag)\s+\d+\s+\d+)\s*`{0,3}/g,
        '<span class="action-block">$1</span>');
    return '<pre class="content-seg">' + escaped + '</pre>';
}

function renderBoard(boardLines, hasHeader) {
    // With a header, boardLines[0] is the column header and the rest are rows.
    // Without one, every line is a data row; derive ncols from the widest row.
    let header = null;
    let rowStart = 0;
    if (hasHeader) {
        header = boardLines[0].trim().split(/\s+/);  // ['0','1','2',...]
        rowStart = 1;
    }
    let ncols = header ? header.length : 0;
    if (!header) {
        for (let r = 0; r < boardLines.length; r++) {
            const n = boardLines[r].trim().split(/\s+/).length - 1;  // minus row index
            if (n > ncols) ncols = n;
        }
    }
    let h = '<div class="ms-grid">';
    if (header) {
        h += '<div class="ms-row"><span class="ms-cell ms-head"></span>';
        for (const c of header) h += `<span class="ms-cell ms-head">${escapeHtml(c)}</span>`;
        h += '</div>';
    }
    for (let r = rowStart; r < boardLines.length; r++) {
        const toks = boardLines[r].trim().split(/\s+/);
        if (toks.length < 1) continue;
        const rowIdx = toks[0];
        const cells = toks.slice(1);
        h += '<div class="ms-row">';
        h += `<span class="ms-cell ms-head">${escapeHtml(rowIdx)}</span>`;
        for (let c = 0; c < ncols; c++) {
            const sym = c < cells.length ? cells[c] : '.';
            h += renderCell(sym);
        }
        h += '</div>';
    }
    h += '</div>';
    return h;
}

function renderCell(sym) {
    if (sym === '.') return '<span class="ms-cell ms-unrevealed">&middot;</span>';
    if (sym === 'F') return '<span class="ms-cell ms-flag">&#128681;</span>';
    if (sym === '*') return '<span class="ms-cell ms-mine">&#128163;</span>';
    if (sym === '0') return '<span class="ms-cell ms-revealed"></span>';
    if (/^[1-8]$/.test(sym)) return `<span class="ms-cell ms-revealed n${sym}">${sym}</span>`;
    return `<span class="ms-cell ms-revealed">${escapeHtml(sym)}</span>`;
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
                data = load_jsonl(filepath)

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
    parser = argparse.ArgumentParser(description='Minesweeper Trajectory Visualizer')
    parser.add_argument('--path', required=True, help='Path to the chat_completions directory')
    parser.add_argument('--port', type=int, default=8765, help='Port (default: 8765)')
    parser.add_argument('--theme', choices=['dark', 'light'], default='dark', help='Color theme (default: dark)')
    parser.add_argument('--no-browser', action='store_true', help='Do not auto-open browser')
    args = parser.parse_args()

    directory = os.path.abspath(args.path)
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        return

    files = get_jsonl_files(directory)
    if not files:
        print(f"Error: No .jsonl files found in {directory}")
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

    print("\U0001F4A3 Minesweeper Trajectory Visualizer")
    print(f"{'-' * 44}")
    print(f"\U0001F4C2 {directory}")
    print(f"\U0001F4C4 {len(files)} files  ({len(train_files)} train, {len(val_files)} val)")
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
