"""Trajectory visualization tools for rLLM agents.

Standalone visualizers for inspecting rollout trajectories:
- ``gradio_visualizer.py``: generic Gradio-based trajectory viewer (``.pt`` files).
- ``{sokoban,minesweeper,rush_hour}_visual.py``: per-game JSONL viewers.
- ``{sokoban,minesweeper,rush_hour}_visual_pt.py``: per-game ``.pt`` viewers.
- ``scripts/``: launch wrappers for each visualizer.
"""
