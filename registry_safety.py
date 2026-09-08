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
import subprocess
import tempfile
import time
import uuid
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
    fcntl/msvcrt so the same code path works on either.

    Reclaiming a stale lock is the delicate part. Age alone does not prove the
    owner died -- a slow save over OneDrive can legitimately exceed the timeout --
    so reclaim only happens when the recorded process is genuinely gone. Even
    then, two waiters could both decide to reclaim, so ownership is verified
    after acquiring: whoever finds someone else's token in the file backs off
    rather than proceeding alongside them.
    """
    lock_file = _lock_path(path)
    os.makedirs(os.path.dirname(lock_file) or ".", exist_ok=True)
    started = time.time()
    token = f"{os.getpid()}:{uuid.uuid4().hex}"

    while True:
        try:
            handle = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(handle, token.encode("utf-8"))
            os.close(handle)
        except FileExistsError:
            if time.time() - started >= timeout_s:
                if _reclaimable(lock_file, timeout_s):
                    try:
                        os.unlink(lock_file)
                    except OSError:
                        pass
                    started = time.time()  # give the retry a fresh budget
                    continue
                raise RegistryLockTimeout(
                    f"The registry stayed locked by another writer for {timeout_s:.0f}s: {path}"
                )
            time.sleep(poll_s)
            continue

        # Confirm we still own what we just created. If a racing reclaimer
        # deleted our lock and wrote its own, back off instead of both of us
        # believing we hold it -- that is the corruption this exists to stop.
        if _owner_of(lock_file) != token:
            time.sleep(poll_s)
            continue
        break

    try:
        yield
    finally:
        if _owner_of(lock_file) == token:
            try:
                os.unlink(lock_file)
            except OSError:
                pass


def _owner_of(lock_file: str) -> str:
    try:
        with open(lock_file, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _process_alive(pid: int) -> bool:
    """Best-effort liveness check. Unknown is treated as alive, because wrongly
    declaring a live writer dead is what corrupts the file."""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import subprocess
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=10)
            return str(pid) in (result.stdout or "")
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True          # exists, owned by someone else
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    except Exception:        # noqa: BLE001 - unknown means assume alive
        return True


def _reclaimable(lock_file: str, timeout_s: float) -> bool:
    """A lock may be taken over only when its owner is provably gone."""
    owner = _owner_of(lock_file)
    if not owner:
        # No token at all: an old-format or truncated lock. Fall back to age.
        try:
            return (time.time() - os.path.getmtime(lock_file)) >= timeout_s
        except OSError:
            return True
    pid_text = owner.split(":", 1)[0]
    try:
        pid = int(pid_text)
    except ValueError:
        return False
    if pid == os.getpid():
        return False         # our own lock; never steal from ourselves
    return not _process_alive(pid)


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


def _backup_sort_key(path: str):
    """Sort by the timestamp in the filename, not mtime.

    Filesystem timestamps are coarse enough on Windows that two backups written
    a few milliseconds apart can tie, leaving their order undefined -- so
    "restore the most recent" could quietly restore the wrong one. The name
    carries microseconds and sorts lexicographically.
    """
    return (os.path.basename(path), os.path.getmtime(path))


def _prune_backups(folder: str, keep: int = BACKUP_KEEP) -> None:
    try:
        entries = sorted(
            (os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".xlsx")),
            key=_backup_sort_key,
            reverse=True,
        )
    except OSError:
        return
    for stale in entries[keep:]:
        try:
            os.unlink(stale)
        except OSError:
            pass


def save_workbook_atomic(workbook, path: str, backup: bool = True) -> None:
    """Save to a temp file in the same directory, then replace.

    os.replace is atomic within a filesystem, so a reader either sees the whole
    previous workbook or the whole new one -- never a partially written file.

    The previous version is copied aside first. Backups were originally written
    but never called, which meant the recovery story was fiction: atomic replace
    protects against a *crash* mid-write, but nothing protected against a write
    that succeeded and was wrong.
    """
    folder = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(folder, exist_ok=True)
    if backup:
        try:
            backup_registry(path)
        except OSError:
            pass  # a failed backup must not block the save itself
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


def restore_latest_backup(path: str) -> str:
    """Put the most recent backup back. Returns the backup used, or ""."""
    folder = os.path.join(os.path.dirname(os.path.abspath(path)) or ".", BACKUP_DIRNAME)
    if not os.path.isdir(folder):
        return ""
    candidates = sorted(
        (os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".xlsx")),
        key=_backup_sort_key, reverse=True)
    if not candidates:
        return ""
    shutil.copy2(candidates[0], path)
    return candidates[0]
