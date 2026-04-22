"""
Convert pixel-space drift NPZ files to velocity fields (m/s) and save to a parallel directory tree.

Example usage:
  python -m src.data_creation.data_creation_v3.convert_pixel_drift_to_velocity \
    --in-root /data/VECTOR_FIELDS_24h_pairs/HV_HH \
    --out-root /data/VECTOR_FIELDS_24h_pairs_velocity/HV_HH
"""
import os
import re
import argparse
import numpy as np
from datetime import datetime

TS_RE = re.compile(
    r"(?P<t0>\d{8}T\d{4})__(?P<t1>\d{8}T\d{4})(?:_backward)?_(?P<dir>past|future)\.npz$"
)


PIXEL_SIZE_M = 100.0

def dt_seconds_from_filename(path):
    fname = os.path.basename(path)
    m = TS_RE.search(fname)
    if not m:
        raise ValueError(f"Bad filename format: {fname}")

    t0 = datetime.strptime(m.group("t0"), "%Y%m%dT%H%M")
    t1 = datetime.strptime(m.group("t1"), "%Y%m%dT%H%M")
    dt = (t1 - t0).total_seconds()
    if dt <= 0:
        raise ValueError(f"Non-positive dt in {fname}: {dt}")
    return dt

def convert_file(in_path, out_path, overwrite=False):
    if (not overwrite) and os.path.exists(out_path):
        return "skipped"

    dt_sec = dt_seconds_from_filename(in_path)

    with np.load(in_path, allow_pickle=True) as data:
        u_pix = data["u"].astype(np.float32, copy=False)
        v_pix = data["v"].astype(np.float32, copy=False)
        meta  = data["meta"].item()  # dict

    scale = PIXEL_SIZE_M / dt_sec  # m/s per pixel
    u_ms = u_pix * scale
    v_ms = v_pix * scale

    meta = dict(meta)
    meta.update({
        "pixel_size_m": float(PIXEL_SIZE_M),
        "dt_seconds": float(dt_sec),
        "u_units": "m s-1",
        "v_units": "m s-1",
        "convention": "u,v are image-grid velocities (cols, rows) in m/s",
        "original_units": "pixels",
    })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, u=u_ms, v=v_ms, meta=meta)
    return "ok"

def _list_files(in_root):
    """Return a deterministic sorted list of *_past.npz and *_future.npz under in_root."""
    files = []
    for dirpath, _, filenames in os.walk(in_root):
        for fn in filenames:
            if fn.endswith(("_past.npz", "_future.npz")):
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files

def _contiguous_slice(n, task_id, n_tasks):
    """
    Contiguous partition of range(n) across n_tasks tasks.
    task_id is 1-indexed (SGE_TASK_ID).
    Returns (start, end) with end exclusive.
    """
    if not (1 <= task_id <= n_tasks):
        raise ValueError(f"task_id must be in [1..n_tasks], got {task_id}/{n_tasks}")

    q, r = divmod(n, n_tasks)  # base size q, first r chunks get +1

    if task_id <= r:
        start = (task_id - 1) * (q + 1)
        end = start + (q + 1)
    else:
        start = r * (q + 1) + (task_id - r - 1) * q
        end = start + q

    return start, min(end, n)

def convert_tree(in_root, out_root, overwrite=False, sanity_print=5, task_id=None, n_tasks=None):
    n_ok = n_fail = n_skip = 0
    printed = 0

    # build file list once, optionally slice to a contiguous chunk
    all_files = _list_files(in_root)

    if (task_id is not None) and (n_tasks is not None):
        start, end = _contiguous_slice(len(all_files), task_id, n_tasks)
        files = all_files[start:end]
        print(f"[task {task_id}/{n_tasks}] total_files={len(all_files)} "
              f"chunk=[{start}:{end}) -> {len(files)} files")
    else:
        files = all_files
        print(f"[single task] total_files={len(files)}")

    for in_path in files:
        rel = os.path.relpath(in_path, in_root)
        out_path = os.path.join(out_root, rel)

        try:
            status = convert_file(in_path, out_path, overwrite=overwrite)
            if status == "ok":
                n_ok += 1
                if printed < sanity_print:
                    dt = dt_seconds_from_filename(in_path)
                    with np.load(out_path, allow_pickle=True) as d2:
                        u = d2["u"]; v = d2["v"]
                    speed = np.sqrt(u*u + v*v)
                    print(f"[OK] {rel}  dt={dt/3600:.2f}h  "
                          f"speed(m/s) min={np.nanmin(speed):.4f} max={np.nanmax(speed):.4f}")
                    printed += 1
            elif status == "skipped":
                n_skip += 1
        except Exception as e:
            n_fail += 1
            print(f"[FAIL] {in_path}\n  {e}\n")

    print(f"Done. Converted={n_ok}, skipped={n_skip}, failed={n_fail}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", required=True, help="Root of pixel-drift NPZ tree")
    ap.add_argument("--out-root", required=True, help="Root of output velocity NPZ tree")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--sanity-print", type=int, default=5)

    ap.add_argument("--task-id", type=int, default=None, help="1-indexed (e.g. $SGE_TASK_ID)")
    ap.add_argument("--n-tasks", type=int, default=None, help="total tasks (e.g. $SGE_TASK_LAST)")

    args = ap.parse_args()

    convert_tree(
        args.in_root,
        args.out_root,
        overwrite=args.overwrite,
        sanity_print=args.sanity_print,
        task_id=args.task_id,
        n_tasks=args.n_tasks,
    )
