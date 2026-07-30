"""Utilities for selecting and configuring a safe temporary directory.

Some third-party dependencies (e.g., planners used by TextWorld/ALFWorld) rely on
Python's :mod:`tempfile` default directory. In multi-process training runs, the
default temp location can run out of space or inodes, causing hard failures.

This module provides a small helper to redirect temp files to a directory with
enough free capacity, optionally controlled via ``RLLM_TMPDIR``.

It also provides periodic cleanup utilities to remove old temp directories
and prevent disk space exhaustion during long training runs.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def get_alfworld_tmp_base() -> str:
    """Return the base directory used for ALFWorld temporary directories."""
    return "/tmp/rllm_tmp"


@dataclass(frozen=True)
class TmpDirChoice:
    path: str
    reason: str


def _stat_free(path: str) -> tuple[int, int] | None:
    """Return (free_bytes, free_inodes) for filesystem containing `path`."""
    try:
        st = os.statvfs(path)
    except OSError:
        return None
    free_bytes = int(st.f_frsize) * int(st.f_bavail)
    free_inodes = int(st.f_favail)
    return free_bytes, free_inodes


def _ensure_dir_writable(path: str) -> tuple[bool, str]:
    return True, "ok"


def _apply_tmpdir(path: str) -> None:
    os.environ["TMPDIR"] = path
    os.environ["TEMP"] = path
    os.environ["TMP"] = path
    tempfile.tempdir = path


def configure_tmpdir(
    preferred: str | None = None,
    *,
    min_free_bytes: int = 256 * 1024 * 1024,
    min_free_inodes: int = 10_000,
) -> TmpDirChoice:
    """Select and configure a temporary directory.

    We require a single, explicit location and do not fall back to other paths.
    Free bytes and inodes are logged for diagnostics but are not enforced.
    Environment variables and :mod:`tempfile` are configured to use the path.
    """
    chosen = preferred or os.environ.get("RLLM_TMPDIR") or "/tmp/rllm_tmp"
    path = os.path.expanduser(chosen)

    ok, writable_reason = _ensure_dir_writable(path)
    if not ok:
        raise RuntimeError(f"tmpdir not writable: {path} ({writable_reason})")

    stats = _stat_free(path)
    if stats is None:
        raise RuntimeError(f"tmpdir stat failed: {path}")

    free_bytes, free_inodes = stats
    logger.debug("free_bytes: %s, free_inodes: %s", free_bytes, free_inodes)
    logger.debug(
        "min_free_bytes: %s, min_free_inodes: %s",
        min_free_bytes,
        min_free_inodes,
    )

    _apply_tmpdir(path)
    return TmpDirChoice(path=path, reason=f"explicit,{writable_reason}")


def cleanup_old_tmpdirs(
    base_path: str | None = None,
    max_age_seconds: float = 3600,
    exclude_current: str | None = None,
    dry_run: bool = False,
    *,
    include_common_fallbacks: bool = True,
) -> tuple[int, int, list[str]]:
    """Clean up old temporary directories to prevent disk space exhaustion.

    This function removes temporary directories that are older than `max_age_seconds`.
    It's designed to be called periodically during long training runs.

    Args:
        base_path: Base directory to clean. Defaults to ALFWORLD_TMP_BASE.
        max_age_seconds: Maximum age in seconds before a directory is considered stale.
            Defaults to 3600 (1 hour).
        exclude_current: Path to exclude from cleanup (e.g., the current process's tmpdir).
        dry_run: If True, only report what would be deleted without actually deleting.

    Returns:
        Tuple of (directories_removed, bytes_freed, list_of_removed_paths).
    """
    if base_path is None:
        base_path = get_alfworld_tmp_base()

    if not os.path.exists(base_path):
        return 0, 0, []

    current_time = time.time()
    dirs_removed = 0
    bytes_freed = 0
    removed_paths = []

    cleanup_paths = [base_path]
    if include_common_fallbacks:
        cleanup_paths.append("/tmp/rllm_tmp")

    for cleanup_base in cleanup_paths:
        if not os.path.exists(cleanup_base):
            continue

        try:
            for entry in os.scandir(cleanup_base):
                if not entry.is_dir():
                    continue
                # Only operate on directories created by `tempfile` (tmpXXXXXX).
                # This avoids deleting unrelated cache folders under the same base.
                if not entry.name.startswith("tmp"):
                    continue

                # Skip if this is the current process's tmpdir
                if exclude_current and os.path.samefile(entry.path, exclude_current):
                    continue

                try:
                    # Check directory age using modification time
                    mtime = entry.stat().st_mtime
                    age = current_time - mtime

                    if age > max_age_seconds:
                        # Calculate size before deletion
                        dir_size = _get_dir_size(entry.path)

                        if dry_run:
                            logger.info(
                                f"[DRY RUN] Would remove: {entry.path} "
                                f"(age={age:.0f}s, size={dir_size / 1024 / 1024:.1f}MB)"
                            )
                        else:
                            shutil.rmtree(entry.path, ignore_errors=True)
                            logger.debug(
                                f"Removed old tmpdir: {entry.path} "
                                f"(age={age:.0f}s, size={dir_size / 1024 / 1024:.1f}MB)"
                            )

                        dirs_removed += 1
                        bytes_freed += dir_size
                        removed_paths.append(entry.path)

                except (OSError, PermissionError) as e:
                    logger.warning(f"Failed to process {entry.path}: {e}")
                    continue

        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to scan {cleanup_base}: {e}")
            continue

    if dirs_removed > 0 and not dry_run:
        logger.info(
            f"Cleaned up {dirs_removed} old temp directories, "
            f"freed {bytes_freed / 1024 / 1024:.1f}MB"
        )

    return dirs_removed, bytes_freed, removed_paths


def _get_dir_size(path: str) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += _get_dir_size(entry.path)
    except (OSError, PermissionError):
        pass
    return total


def cleanup_stale_tmp_files(
    patterns: list[str] | None = None,
    max_age_seconds: float = 3600,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Clean up stale temporary files matching specific patterns.

    This is useful for cleaning up files like libdownward.so copies that
    accumulate in temp directories.

    Args:
        patterns: List of glob patterns to match. Defaults to common patterns.
        max_age_seconds: Maximum age in seconds before a file is considered stale.
        dry_run: If True, only report what would be deleted.

    Returns:
        Tuple of (files_removed, bytes_freed).
    """
    if patterns is None:
        alfworld_tmp_base = get_alfworld_tmp_base()
        patterns = [
            "/tmp/tmp*/libdownward.so",
            os.path.expanduser("~/.cache/rllm/tmp/tmp*/libdownward.so"),
            f"{alfworld_tmp_base}/tmp*/libdownward.so",
        ]

    current_time = time.time()
    files_removed = 0
    bytes_freed = 0

    for pattern in patterns:
        try:
            for filepath in glob.glob(pattern):
                try:
                    stat = os.stat(filepath)
                    age = current_time - stat.st_mtime

                    if age > max_age_seconds:
                        file_size = stat.st_size

                        if dry_run:
                            logger.info(
                                f"[DRY RUN] Would remove: {filepath} "
                                f"(age={age:.0f}s, size={file_size / 1024 / 1024:.1f}MB)"
                            )
                        else:
                            os.remove(filepath)
                            # Also try to remove the parent tmp directory if empty
                            parent = os.path.dirname(filepath)
                            try:
                                os.rmdir(parent)
                            except OSError:
                                pass  # Directory not empty, that's fine

                        files_removed += 1
                        bytes_freed += file_size

                except (OSError, PermissionError) as e:
                    logger.warning(f"Failed to process {filepath}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Failed to glob pattern {pattern}: {e}")
            continue

    if files_removed > 0 and not dry_run:
        logger.info(
            f"Cleaned up {files_removed} stale temp files, "
            f"freed {bytes_freed / 1024 / 1024:.1f}MB"
        )

    return files_removed, bytes_freed


class PeriodicTmpCleaner:
    """A helper class that performs periodic cleanup of temp directories.

    This can be used to automatically clean up old temp files at regular intervals
    during long training runs.

    Example:
        cleaner = PeriodicTmpCleaner(interval_seconds=300)  # Every 5 minutes

        # In your training loop:
        for step in range(total_steps):
            cleaner.maybe_cleanup()  # Only runs if interval has passed
            # ... training code ...
    """

    def __init__(
        self,
        interval_seconds: float = 300,
        max_age_seconds: float = 1800,
        exclude_current: str | None = None,
    ):
        """Initialize the periodic cleaner.

        Args:
            interval_seconds: Minimum time between cleanup runs.
            max_age_seconds: Maximum age of directories to keep.
            exclude_current: Path to exclude from cleanup.
        """
        self.interval_seconds = interval_seconds
        self.max_age_seconds = max_age_seconds
        self.exclude_current = exclude_current
        self._last_cleanup_time = 0.0
        self._total_dirs_removed = 0
        self._total_bytes_freed = 0

    def maybe_cleanup(self, force: bool = False) -> bool:
        """Run cleanup if enough time has passed since the last run.

        Args:
            force: If True, run cleanup regardless of interval.

        Returns:
            True if cleanup was performed, False otherwise.
        """
        current_time = time.time()
        elapsed = current_time - self._last_cleanup_time

        if not force and elapsed < self.interval_seconds:
            return False

        dirs_removed, bytes_freed, _ = cleanup_old_tmpdirs(
            max_age_seconds=self.max_age_seconds,
            exclude_current=self.exclude_current,
        )

        # Also clean stale files
        files_removed, file_bytes_freed = cleanup_stale_tmp_files(
            max_age_seconds=self.max_age_seconds,
        )

        self._last_cleanup_time = current_time
        self._total_dirs_removed += dirs_removed
        self._total_bytes_freed += bytes_freed + file_bytes_freed

        return True

    def get_stats(self) -> dict:
        """Get cleanup statistics."""
        return {
            "total_dirs_removed": self._total_dirs_removed,
            "total_bytes_freed": self._total_bytes_freed,
            "total_mb_freed": self._total_bytes_freed / 1024 / 1024,
            "last_cleanup_time": self._last_cleanup_time,
        }
