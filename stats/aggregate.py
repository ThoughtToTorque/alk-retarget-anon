"""Load per-rollout JSON files (pipeline/retarget_runner.py output) into a
tidy table.

The canonical layout is  <root>/<task>/<seed>/rollout_<variant>.json
(e.g. data/box_open/0/rollout_full.json), but any directory tree is
accepted: every file whose basename matches ``rollout_*.json`` is loaded.

Each rollout becomes one flat row (dict) with the keys in :data:`ROW_KEYS`.
Missing / not-yet-produced fields (e.g. ``discrete_correct`` before the
VLM error-injection phase) are ``None`` and are skipped gracefully by the
downstream report code.

Output options:
  - list of dicts (always)                     -> :func:`load_rollouts`
  - CSV (always available, stdlib csv)         -> :func:`write_csv`
  - pandas DataFrame (only if pandas exists)   -> :func:`to_dataframe`

Python 3.8 compatible; no third-party hard dependencies.
"""

import csv
import json
import os
from typing import Any, Dict, Iterator, List, Optional

# Column order of the tidy table.
ROW_KEYS = [
    "task",
    "seed",
    "method",
    "variant",
    "success",
    "grasp_success",
    "rot_err",            # final rotation error vs GT [deg]
    "trans_err",          # final translation error vs GT [m]
    "rot_err_init",       # Procrustes-only (pre-refinement) rotation error [deg]
    "trans_err_init",     # Procrustes-only translation error [m]
    "chamfer_before",     # [m]
    "chamfer_after",      # [m]
    "discrete_correct",   # 1/0/None: all discrete choices (phi) correct?
    "discrete_flags",     # per-phi dict as JSON string, or None
    "failure_stage",      # None | 'grasp' | 'post_action' | ...
    "conditioning",       # ALK conditioning number (sigma2+sigma3 etc.), or None
    "injected_error_rate",  # error-injection experiments, or None
    "camera",
    "n_exec_steps",
    "time_s",
    "source_file",
]


def _as01(value: Any) -> Optional[int]:
    """Booleans -> 1/0, None stays None."""
    if value is None:
        return None
    return int(bool(value))


def _first(rec: Dict[str, Any], *keys: str) -> Any:
    """First present, non-None value among ``keys`` in ``rec``."""
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return None


def _extract_discrete(rec: Dict[str, Any]):
    """Return (overall_correct, per_phi_flags_json_or_None).

    Looks for (in priority order):
      1. top-level ``discrete_correct``: bool, or dict {phi_name: bool}
      2. a ``vlm`` section with ``discrete_correct`` / ``correct`` /
         per-phi ``phi*_correct`` keys
    Oracle-only rollouts carry no such field -> (None, None); "the oracle is
    correct by construction" is a *protocol* statement, not data, so we do
    not fabricate a flag here.
    """
    val = rec.get("discrete_correct")
    if val is None:
        vlm = rec.get("vlm")
        if isinstance(vlm, dict):
            val = _first(vlm, "discrete_correct", "correct")
            if val is None:
                phi = {k: v for k, v in vlm.items()
                       if k.startswith("phi") and k.endswith("_correct")}
                if phi:
                    val = phi
    if val is None:
        return None, None
    if isinstance(val, dict):
        # ignore None entries (phi4/phi5 may be n/a for some tasks)
        flags = {k: bool(v) for k, v in val.items() if v is not None}
        overall = int(all(flags.values())) if flags else None
        return overall, json.dumps(flags, sort_keys=True)
    return _as01(val), None


def _extract_conditioning(rec: Dict[str, Any]) -> Optional[float]:
    """ALK conditioning metric if the runner recorded one (Prop-3 analysis)."""
    val = _first(rec, "conditioning", "alk_conditioning", "sigma23")
    if val is None:
        diag = rec.get("oracle", {}).get("diagnostics", {})
        if isinstance(diag, dict):
            val = _first(diag, "conditioning", "alk_conditioning", "sigma23")
    return None if val is None else float(val)


def row_from_record(rec: Dict[str, Any], source_file: Optional[str] = None) -> Dict[str, Any]:
    """Flatten one rollout JSON record into a tidy row (dict, ROW_KEYS)."""
    variant = _first(rec, "variant", "method")
    overall, flags = _extract_discrete(rec)
    row = {
        "task": rec.get("task"),
        "seed": rec.get("seed"),
        "method": _first(rec, "method", "variant"),
        "variant": variant,
        "success": _as01(rec.get("success")),
        "grasp_success": _as01(_first(rec, "grasp_success", "grasped")),
        "rot_err": _first(rec, "rot_err_deg", "rot_err"),
        "trans_err": _first(rec, "trans_err_m", "trans_err"),
        "rot_err_init": _first(rec, "rot_err_init_deg", "rot_err_init"),
        "trans_err_init": _first(rec, "trans_err_init_m", "trans_err_init"),
        "chamfer_before": _first(rec, "chamfer_before_m", "chamfer_before"),
        "chamfer_after": _first(rec, "chamfer_after_m", "chamfer_after"),
        "discrete_correct": overall,
        "discrete_flags": flags,
        "failure_stage": rec.get("failure_stage"),
        "conditioning": _extract_conditioning(rec),
        "injected_error_rate": _first(
            rec, "injected_error_rate", "error_injection_rate", "inject_rate"),
        "camera": rec.get("camera"),
        "n_exec_steps": rec.get("n_exec_steps"),
        "time_s": rec.get("time_s"),
        "source_file": source_file,
    }
    return row


def iter_rollout_files(root: str) -> Iterator[str]:
    """Yield every ``rollout_*.json`` under ``root``, sorted for determinism.

    Subtrees whose directory name ends with ``_outofscope`` are skipped:
    they hold preserved rollouts from task variants later ruled out of the
    method's scope (e.g. ``pour_cube_outofscope``) and must not be pooled
    into the live tables.
    """
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.endswith("_outofscope")]
        dirnames.sort()
        for name in sorted(filenames):
            if name.startswith("rollout_") and name.endswith(".json"):
                hits.append(os.path.join(dirpath, name))
    for path in sorted(hits):
        yield path


def load_rollouts(root: str, strict: bool = False) -> List[Dict[str, Any]]:
    """Load all rollout JSONs under ``root`` into a tidy list of row dicts.

    ``strict=False`` (default) skips unreadable files with a warning on
    stderr; ``strict=True`` raises instead.
    """
    rows = []
    for path in iter_rollout_files(root):
        try:
            with open(path, "r") as f:
                rec = json.load(f)
            rows.append(row_from_record(rec, source_file=path))
        except (ValueError, OSError):
            if strict:
                raise
            import sys
            print("aggregate: skipping unreadable %s" % path, file=sys.stderr)
    return rows


def write_csv(rows: List[Dict[str, Any]], path: str) -> str:
    """Write the tidy table to CSV (columns = ROW_KEYS). Returns ``path``."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_KEYS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def to_dataframe(rows: List[Dict[str, Any]]):
    """Return a pandas DataFrame if pandas is installed, else None.

    (pandas is NOT a dependency of this package, so all
    downstream code works on the plain list-of-dicts representation.)
    """
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        return None
    return pd.DataFrame(rows, columns=ROW_KEYS)


# ---------------------------------------------------------------------------
# small query helpers used by report.py (keep them here so report.py stays
# purely about statistics/formatting)
# ---------------------------------------------------------------------------

def unique(rows: List[Dict[str, Any]], key: str) -> List[Any]:
    """Sorted unique non-None values of ``key``."""
    return sorted({r[key] for r in rows if r.get(key) is not None})


def select(rows: List[Dict[str, Any]], **conds: Any) -> List[Dict[str, Any]]:
    """Rows matching all key=value conditions."""
    out = rows
    for k, v in conds.items():
        out = [r for r in out if r.get(k) == v]
    return out


def column(rows: List[Dict[str, Any]], key: str, drop_none: bool = True) -> List[Any]:
    vals = [r.get(key) for r in rows]
    if drop_none:
        vals = [v for v in vals if v is not None]
    return vals


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="Aggregate per-rollout JSONs into a tidy CSV table.")
    ap.add_argument("root", help="directory tree containing rollout_*.json files")
    ap.add_argument("-o", "--out", default=None,
                    help="output CSV path (default: <root>/rollouts.csv)")
    args = ap.parse_args(argv)
    rows = load_rollouts(args.root)
    out = args.out or os.path.join(args.root, "rollouts.csv")
    write_csv(rows, out)
    print("wrote %d rows -> %s" % (len(rows), out))


if __name__ == "__main__":
    main()
