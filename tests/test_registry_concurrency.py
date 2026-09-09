"""Concurrency check for the registry write-safety layer.

Spawns real threads that all hammer lock_entry() on the same workbook, then
verifies every single write survived and the file is still readable. Without
the lock this loses rows and can corrupt the workbook outright.
"""

import os
import shutil
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import registry_safety
import storyboard_store as store

WRITERS = 8
PER_WRITER = 6


def _patch_open(should_deny):
    """Swap os.open for one that can pretend the lock file is delete-pending."""
    real_open = os.open

    def fake_open(path, flags, *args, **kwargs):
        if str(path).endswith(registry_safety.LOCK_SUFFIX) and should_deny():
            raise PermissionError(13, "Access is denied", str(path))
        return real_open(path, flags, *args, **kwargs)

    registry_safety.os.open = fake_open
    return real_open


def scenario_delete_pending():
    """Windows raises PermissionError, not FileExistsError, while a just-unlinked
    lock file is still delete-pending. That is contention and must be waited out,
    not handed back to the caller as a failed write.
    """
    workdir = tempfile.mkdtemp(prefix="registry_denied_")
    path = os.path.join(workdir, "storyboard.xlsx")
    remaining = [4]

    def should_deny():
        if remaining[0] > 0:
            remaining[0] -= 1
            return True
        return False

    real_open = _patch_open(should_deny)
    try:
        with registry_safety.registry_lock(path, timeout_s=5.0, poll_s=0.01):
            acquired = True
    except Exception as exc:  # noqa: BLE001 - any escape at all is the bug
        acquired = False
        print(f"    transient denial escaped: {type(exc).__name__}: {exc}")
    finally:
        registry_safety.os.open = real_open

    retried = remaining[0] == 0
    print(f"transient denial -> acquired={acquired} all_retries_used={retried}")

    # And a denial that never clears must still surface as a lock timeout rather
    # than a raw PermissionError, so callers only ever handle the one exception.
    real_open = _patch_open(lambda: True)
    try:
        with registry_safety.registry_lock(path, timeout_s=0.2, poll_s=0.01):
            raised = "nothing"
    except registry_safety.RegistryLockTimeout:
        raised = "RegistryLockTimeout"
    except Exception as exc:  # noqa: BLE001
        raised = type(exc).__name__
    finally:
        registry_safety.os.open = real_open

    print(f"permanent denial -> raised={raised}")
    shutil.rmtree(workdir, ignore_errors=True)
    return acquired and retried and raised == "RegistryLockTimeout"


def main():
    workdir = tempfile.mkdtemp(prefix="registry_race_")
    path = os.path.join(workdir, "storyboard.xlsx")
    errors = []

    def writer(worker):
        for n in range(PER_WRITER):
            try:
                store.lock_entry(
                    path=path,
                    entry_id=f"SHOT:w{worker}_{n}",
                    entry_type="SHOT",
                    beat=f"beat{n}",
                    description="concurrency probe",
                    prompt_positive="p",
                    prompt_negative_add="",
                    model="mock",
                    seed=n,
                    reused_from="",
                    image_path="",
                    note=f"worker {worker} write {n}",
                )
            except Exception as exc:  # noqa: BLE001 - the whole point is to catch anything
                errors.append(f"worker {worker} write {n}: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(WRITERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected = WRITERS * PER_WRITER
    rows = store.list_entries(path)
    ids = {r.get("entry_id") for r in rows}
    missing = {f"SHOT:w{w}_{n}" for w in range(WRITERS) for n in range(PER_WRITER)} - ids

    print(f"expected rows : {expected}")
    print(f"rows present  : {len(rows)}")
    print(f"unique ids    : {len(ids)}")
    print(f"missing       : {len(missing)}")
    print(f"write errors  : {len(errors)}")
    for e in errors[:5]:
        print("   ", e)

    leftover_locks = [f for f in os.listdir(workdir) if f.endswith(".lock")]
    print(f"stale locks   : {len(leftover_locks)}")

    ok = (not errors) and (not missing) and len(ids) == expected and not leftover_locks
    shutil.rmtree(workdir, ignore_errors=True)

    print("--- delete-pending lock file ---")
    ok = scenario_delete_pending() and ok

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
