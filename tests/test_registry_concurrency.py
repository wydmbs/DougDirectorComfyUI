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

import storyboard_store as store

WRITERS = 8
PER_WRITER = 6


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
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
