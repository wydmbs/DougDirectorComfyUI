"""
registry_safety.py -- makes the Assets registry safe to write from more than one
place at a time.

The registry is a single .xlsx file that openpyxl rewrites wholesale on every
save. That is entirely fine when a person clicks Lock a few dozen times. It is
not fine once the assistant director is locking panels on its own, because two
overlapping saves can leave a half-written workbook behind, and a half-written
registry is the whole project.

Three protections, all cheap:

  * a cross-process file lock, so only one writer touches the workbook at a time
  * an atomic replace, so a crash mid-write leaves the previous good file intact
  * a rolling backup, so a bad write is recoverable rather than terminal

This keeps the .xlsx as the single shared backbone (see SUITE.md) rather than
introducing a second store alongside it.
"""

import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime

LOCK_SUFFIX = ".lock"
BACKUP_DIRNAME = "registry_backups"
BACKUP_KEEP = 20
DEFAULT_TIMEOUT_S = 30.0


class RegistryLockTimeout(Exception):
    """Raised when another writer held the registry lock for too long."""


def _lock_path(path: str) -> str:
    return os.path.abspath(path) + LOCK_SUFFIX


@contextmanager
def registry_lock(path: str, timeout_s: float = DEFAULT_TIMEOUT_S, poll_s: float = 0.05):
    """Exclusive cross-process lock for one registry file.

    Uses O_CREAT|O_EXCL, which is atomic on both Windows and POSIX, rather than
    fcntl/msvcrt so the same code path works on either. A stale lock left behind
    by a killed process is reclaimed after `timeout_s` so the app can never be
    permanently wedged by a crash.
    """
    lock_file = _lock_path(path)
    os.makedirs(os.path.dirname(lock_file) or ".", exist_ok=True)
    started = time.time()
    handle = None
    while True:
        try:
            handle = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            waited = time.time() - started
            if waited >= timeout_s:
                # Reclaim a lock that is older than the timeout: the only way it
                # can still exist is that its owner died without cleaning up.
                try:
                    age = time.time() - os.path.getmtime(lock_file)
                except OSError:
                    age = 0
                if age >= timeout_s:
                    try:
                        os.unlink(lock_file)
                        continue
                    except OSError:
                        pass
                raise RegistryLockTimeout(
                    f"The registry stayed locked by another writer for {timeout_s:.0f}s: {path}"
                )
            time.sleep(poll_s)
    try:
        os.write(handle, str(os.getpid()).encode("utf-8"))
        os.close(handle)
        handle = None
        yield
    finally:
        if handle is not None:
            try:
                os.close(handle)
            except OSError:
                pass
        try:
            os.unlink(lock_file)
        except OSError:
            pass


def backup_registry(path: str, tag: str = "") -> str:
    """Copy the current registry aside. Returns the backup path, or "" if there
    was nothing to back up yet."""
    if not os.path.exists(path):
        return ""
    folder = os.path.join(os.path.dirname(os.path.abspath(path)) or ".", BACKUP_DIRNAME)
    os.makedirs(folder, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    suffix = f"_{tag}" if tag else ""
    name = f"{os.path.splitext(os.path.basename(path))[0]}_{stamp}{suffix}.xlsx"
    dest = os.path.join(folder, name)
    shutil.copy2(path, dest)
    _prune_backups(folder)
    return dest


def _prune_backups(folder: str, keep: int = BACKUP_KEEP) -> None:
    try:
        entries = sorted(
            (os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".xlsx")),
            key=os.path.getmtime,
            reverse=True,
        )
    except OSError:
        return
    for stale in entries[keep:]:
        try:
            os.unlink(stale)
        except OSError:
            pass


def save_workbook_atomic(workbook, path: str) -> None:
    """Save to a temp file in the same directory, then replace.

    os.replace is atomic within a filesystem, so a reader either sees the whole
    previous workbook or the whole new one -- never a partially written file.
    """
    folder = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(folder, exist_ok=True)
    handle, temp_path = tempfile.mkstemp(suffix=".xlsx", prefix=".registry_", dir=folder)
    os.close(handle)
    try:
        workbook.save(temp_path)
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise
