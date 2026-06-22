#!/usr/bin/env python3
"""Dataset-level quality summary across data_metrics.py JSON output files.

Usage:
    python data_metrics_summary.py --root /path/to/root [--pattern '*.json'] [--d31] [--json out.json]

Recursively scans --root for JSON files matching --pattern (repeatable, default '*.json')
and reads the schema written by data_metrics.py (top-level "video" and/or "omegaprime" keys).
Each file is auto-classified by content; files with neither key are skipped. Prints aggregate
statistics and a per-file breakdown. Self-contained — no orchestrator layout assumptions.
"""

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Warn thresholds for video metrics: (threshold, "low"|"high")
# "low"  → warn if value < threshold
# "high" → warn if value > threshold
_VIDEO_WARN: Dict[str, Tuple[float, str]] = {
    "tds":   (95.0, "low"),
    "tcomp": (95.0, "low"),
    "dup":   (99.0, "low"),
    "sci":   (70.0, "low"),
    "blk":   (70.0, "low"),
    "fov":   (70.0, "low"),
}

_VIDEO_LABELS = {
    "tds":   "TDS",
    "tcomp": "Temporal Completeness",
    "dup":   "Unique Record Rate",
    "sci":   "SCI",
    "blk":   "Partial Blockage",
    "fov":   "FOV Stability",
}

_OMEGA_LABELS = {
    "attr":     "Attribute Completeness",
    "rec":      "Record Completeness",
    "fmt":      "Format Consistency",
    "dup":      "Duplicate Rate",
    "tcomp":    "Temporal Completeness",
    "cls_pct":  "Class Diversity",
    "traj_pct": "Trajectory Plausibility",
}

# passes_threshold keys in omegaprime metrics → short flag label
_OMEGA_PASS_KEYS = [
    ("rec_ok",   "Rec"),
    ("fmt_ok",   "Fmt"),
    ("dup_ok",   "Dup"),
    ("tcomp_ok", "TComp"),
]

# D3.1 official metric names
_D31_VIDEO_LABELS = {
    "tcomp": "Temporal Completeness",
    "dup":   "Unique Record Rate",
    "sci":   "Sensor Consistency",
    "tds":   "Temporal Distortion Score [custom]",
    "blk":   "Partial Blockage [custom]",
    "fov":   "FOV Stability [custom]",
}

_D31_OMEGA_SCALAR_LABELS = {
    "attr":     "Attribute Completeness",
    "rec":      "Record Completeness",
    "fmt":      "Data Format Consistency",
    "dup":      "Duplicate Record Rate",
    "tcomp":    "Temporal Completeness",
    "cls_pct":  "Class Diversity",
    "traj_pct": "Trajectory Plausibility [custom]",
}

# (key, full D3.1 name, column abbreviation)
_D31_OMEGA_PASS_KEYS = [
    ("attr_ok",  "Attribute Completeness",           "Attr"),
    ("rec_ok",   "Record Completeness",              "Rec"),
    ("cls",      "Class Completeness",               "Cls"),
    ("fmt_ok",   "Data Format Consistency",          "Fmt"),
    ("dup_ok",   "Duplicate Record Rate",            "Dup"),
    ("tcomp_ok", "Temporal Completeness",            "TComp"),
    ("obj_cov",  "Object Type Coverage",             "ObjCov"),
    ("traj",     "Trajectory Plausibility [custom]", "Traj"),
    ("fmeta",    "File Metadata [custom]",           "FileMeta"),
]

# Maps scalar metric key → (pass_key, threshold label)
_OMEGA_SCALAR_PASS: Dict[str, Tuple[str, str]] = {
    "attr":     ("attr_ok",  "=100%"),
    "rec":      ("rec_ok",   "≥95%"),
    "fmt":      ("fmt_ok",   "≥95%"),
    "dup":      ("dup_ok",   "≤1%"),
    "tcomp":    ("tcomp_ok", "≥98%"),
    "cls_pct":  ("cls",      "≥2 per field"),
    "traj_pct": ("traj",     "≤2% implaus."),
}

# Pure pass/fail rows (no scalar): (pass_key, label, threshold)
_OMEGA_PASSFAIL_ROWS: List[Tuple[str, str, str]] = [
    ("fmeta", "File Metadata [custom]", "—"),
]


# ── File discovery ─────────────────────────────────────────────────────────────

def find_metric_files(root: Path, patterns: List[str]) -> List[Path]:
    """Return sorted unique JSON files under root matching any of the glob patterns."""
    found = set()
    for pat in patterns:
        found.update(p for p in root.rglob(pat) if p.is_file())
    return sorted(found)


# ── Metric extraction ──────────────────────────────────────────────────────────

def _scale(v: Optional[float]) -> Optional[float]:
    return round(v * 100, 2) if v is not None else None


def _section_getter(section: Dict):
    """Return a g(key) closure reading float values from a metric sub-dict."""
    def g(key):
        v = section.get(key)
        return float(v) if v is not None else None
    return g


def _extract_video(vd: Dict) -> Dict:
    """Extract video metric scalars from a data_metrics 'video' sub-dict."""
    temporal = _section_getter(vd.get("compute_temporal_metrics", {}))
    dupes    = _section_getter(vd.get("compute_duplicate_record_rate_video", {}))
    sensor   = _section_getter(vd.get("compute_sensor_consistency", {}))
    degrad   = _section_getter(vd.get("compute_sensor_degradation", {}))

    raw_dup = dupes("Duplicate Record Rate (%)")
    return {
        "tds":   _scale(temporal("Temporal Distortion Score (TDS)")),
        "tcomp": temporal("Temporal Completeness (%)"),
        # Inverted: reported as Unique Record Rate % (100 - raw duplicate rate)
        "dup":   round(100.0 - raw_dup, 2) if raw_dup is not None else None,
        "sci":   _scale(sensor("Composite SCI (0..1)")),
        "blk":   _scale(degrad("Partial Blockage Score (0..1)")),
        "fov":   _scale(degrad("FOV Change Score (0..1)")),
    }


def _cls_diversity_pct(case2_checks: Optional[Dict]) -> Optional[float]:
    if not case2_checks:
        return None
    return round(sum(case2_checks.values()) / len(case2_checks) * 100, 2)


def _traj_plausibility_pct(implausible_fraction: Optional[float]) -> Optional[float]:
    if implausible_fraction is None:
        return None
    return round((1.0 - implausible_fraction) * 100, 2)


def _extract_omega(od: Dict) -> Dict:
    """Extract OmegaPrime metric scalars and pass flags from an 'omegaprime' sub-dict."""
    def g(section, key):
        v = od.get(section, {}).get(key)
        return float(v) if v is not None else None

    return {
        "attr":     g("attribute_completeness", "completeness_pct"),
        "rec":      g("record_completeness",    "completeness_pct"),
        "fmt":      g("format_consistency",     "consistency_pct"),
        "dup":      g("duplicate_rate",         "duplicate_rate_pct"),
        "tcomp":    g("temporal_completeness",  "completeness_pct"),
        "attr_ok":  od.get("attribute_completeness",  {}).get("passes_threshold"),
        "rec_ok":   od.get("record_completeness",     {}).get("passes_threshold"),
        "fmt_ok":   od.get("format_consistency",      {}).get("passes_threshold"),
        "dup_ok":   od.get("duplicate_rate",          {}).get("passes_threshold"),
        "tcomp_ok": od.get("temporal_completeness",   {}).get("passes_threshold"),
        "cls":      od.get("class_completeness",      {}).get("case2_passes"),
        "obj_cov":  od.get("object_type_coverage",    {}).get("passes"),
        "traj":     od.get("trajectory_plausibility", {}).get("passes_threshold"),
        "fmeta":    od.get("file_metadata",           {}).get("passes"),
        "cls_pct":  _cls_diversity_pct(od.get("class_completeness", {}).get("case2_checks")),
        "traj_pct": _traj_plausibility_pct(od.get("trajectory_plausibility", {}).get("implausible_fraction")),
    }


def load_rows(files: List[Path], root: Path) -> Tuple[List[Dict], List[Dict]]:
    """Read files; return (video_rows, omega_rows) each as {"path", "vals"} dicts.

    Auto-classifies by top-level "video"/"omegaprime" keys; a file may yield both.
    """
    video_rows, omega_rows = [], []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        rel = _relpath(f, root)
        if isinstance(d.get("video"), dict):
            video_rows.append({"path": rel, "vals": _extract_video(d["video"])})
        if isinstance(d.get("omegaprime"), dict):
            omega_rows.append({"path": rel, "vals": _extract_omega(d["omegaprime"])})
    return video_rows, omega_rows


def _relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


# ── Aggregation helpers ────────────────────────────────────────────────────────

def _agg(vals: List[float]) -> Dict:
    if not vals:
        return {}
    return {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals), "n": len(vals)}


def _collect_agg(rows: List[Dict], keys: List[str]) -> Dict[str, List[float]]:
    acc: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        for k in keys:
            if row.get(k) is not None:
                acc[k].append(row[k])
    return acc


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _fv(val: Optional[float], fmt: str = ".1f") -> str:
    if val is None:
        return "—"
    return f"{val:{fmt}}"


def _pf(val: Optional[bool]) -> str:
    if val is None:
        return "—"
    return "✓" if val else "✗"


def _video_flags(vals: Dict) -> str:
    flags = []
    _abbrev = {"tds": "TDS", "tcomp": "TComp", "dup": "Dup", "sci": "SCI", "blk": "Blk", "fov": "FOV"}
    for key, (thresh, direction) in _VIDEO_WARN.items():
        v = vals.get(key)
        if v is None:
            continue
        if direction == "low" and v < thresh:
            flags.append(f"{_abbrev[key]}↓")
        elif direction == "high" and v > thresh:
            flags.append(f"{_abbrev[key]}↑")
    return ", ".join(flags) or "—"


def _omega_flags(vals: Dict) -> str:
    flags = [label for key, label in _OMEGA_PASS_KEYS if vals.get(key) is False]
    return ", ".join(f"{f}↓" for f in flags) or "—"


def _print_agg(title: str, acc: Dict[str, List[float]], labels: Dict[str, str], label_width: int = 28):
    print(f"\n{title}")
    print(f"  {'Metric':<{label_width}}  {'mean %':>7}  {'min %':>7}  {'max %':>7}  {'n':>4}")
    for key, label in labels.items():
        vals = acc.get(key, [])
        if not vals:
            continue
        a = _agg(vals)
        print(f"  {label:<{label_width}}  {_fv(a['mean']):>7}  {_fv(a['min']):>7}  {_fv(a['max']):>7}  {a['n']:>4}")


def _print_omega_agg(title: str, acc: Dict[str, List[float]], rows: List[Dict],
                     labels: Dict[str, str], label_width: int = 36) -> None:
    """Aggregate table for OmegaPrime metrics with inline pass/fail and threshold."""
    print(f"\n{title}")
    print(f"  {'Metric':<{label_width}}  {'mean %':>7}  {'min %':>7}  {'max %':>7}  {'n':>4}   {'Passed':<7}  Threshold")
    for key, label in labels.items():
        a = _agg(acc.get(key, []))
        mean = f"{_fv(a['mean']):>7}" if a else f"{'—':>7}"
        mn   = f"{_fv(a['min']):>7}"  if a else f"{'—':>7}"
        mx   = f"{_fv(a['max']):>7}"  if a else f"{'—':>7}"
        n    = f"{a['n']:>4}"        if a else f"{'':>4}"
        pass_str = thresh = ""
        if key in _OMEGA_SCALAR_PASS:
            pk, thresh = _OMEGA_SCALAR_PASS[key]
            pv = [r[pk] for r in rows if r.get(pk) is not None]
            if pv:
                np_ = sum(1 for v in pv if v)
                ind = "✓" if np_ == len(pv) else "✗" if np_ == 0 else "~"
                pass_str = f"{ind} {np_}/{len(pv)}"
        print(f"  {label:<{label_width}}  {mean}  {mn}  {mx}  {n}   {pass_str:<7}  {thresh}")
    for pk, label, thresh in _OMEGA_PASSFAIL_ROWS:
        pv = [r[pk] for r in rows if r.get(pk) is not None]
        if not pv:
            continue
        np_ = sum(1 for v in pv if v)
        ind = "✓" if np_ == len(pv) else "✗" if np_ == 0 else "~"
        print(f"  {label:<{label_width}}  {'—':>7}  {'—':>7}  {'—':>7}  {'':>4}   {ind} {np_}/{len(pv):<4}  {thresh}")


# ── Main output ────────────────────────────────────────────────────────────────

def _print_header(title: str, root: Path, n_files: int, n_video: int, n_omega: int) -> None:
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"  Root:    {root}")
    print(f"  Scanned: {date.today()}   Files: {n_files}   "
          f"Video: {n_video}   OmegaPrime: {n_omega}")
    print(f"{'='*62}")


def print_summary(root: Path, video_rows: List[Dict], omega_rows: List[Dict], n_files: int):
    v_vals = [r["vals"] for r in video_rows]
    o_vals = [r["vals"] for r in omega_rows]
    _print_header("Dataset Quality Summary", root, n_files, len(v_vals), len(o_vals))

    _print_agg(f"VIDEO QUALITY  ({len(v_vals)} files)",
               _collect_agg(v_vals, list(_VIDEO_LABELS)), _VIDEO_LABELS)
    if o_vals:
        _print_omega_agg(f"OMEGAPRIME QUALITY  ({len(o_vals)} files)",
                         _collect_agg(o_vals, list(_OMEGA_LABELS)), o_vals,
                         _OMEGA_LABELS, label_width=28)

    # Per-file video table
    print(f"\nPER-FILE: VIDEO {'─'*52}")
    print(f"  {'File':<40}  {'TDS%':>6}  {'TComp%':>7}  {'Uniq%':>6}  {'SCI%':>6}  {'Blk%':>6}  {'FOV%':>6}  Flags")
    for r in video_rows:
        v = r["vals"]
        print(f"  {_clip(r['path']):<40}"
              f"  {_fv(v['tds']):>6}  {_fv(v['tcomp']):>7}  {_fv(v['dup']):>6}"
              f"  {_fv(v['sci']):>6}  {_fv(v['blk']):>6}  {_fv(v['fov']):>6}  {_video_flags(v)}")

    # Per-file omegaprime table
    print(f"\nPER-FILE: OMEGAPRIME {'─'*47}")
    print(f"  {'File':<40}  {'Attr%':>6}  {'Rec%':>6}  {'Fmt%':>6}  {'Dup%':>5}  {'TComp%':>7}  Flags")
    for r in omega_rows:
        v = r["vals"]
        print(f"  {_clip(r['path']):<40}"
              f"  {_fv(v['attr']):>6}  {_fv(v['rec']):>6}  {_fv(v['fmt']):>6}"
              f"  {_fv(v['dup'], '.2f'):>5}  {_fv(v['tcomp']):>7}  {_omega_flags(v)}")


def print_d31_summary(root: Path, video_rows: List[Dict], omega_rows: List[Dict], n_files: int):
    """Print D3.1 metric summary with official names."""
    v_vals = [r["vals"] for r in video_rows]
    o_vals = [r["vals"] for r in omega_rows]
    _print_header("Dataset Quality Summary (D3.1 Metrics)", root, n_files, len(v_vals), len(o_vals))

    _print_agg(f"VIDEO QUALITY  ({len(v_vals)} files)",
               _collect_agg(v_vals, list(_D31_VIDEO_LABELS)), _D31_VIDEO_LABELS, label_width=36)
    if o_vals:
        _print_omega_agg(f"OMEGAPRIME QUALITY  ({len(o_vals)} files)",
                         _collect_agg(o_vals, list(_D31_OMEGA_SCALAR_LABELS)), o_vals,
                         _D31_OMEGA_SCALAR_LABELS)

    # Per-file video table
    print(f"\nPER-FILE: VIDEO (D3.1) {'─'*45}")
    print(f"  {'File':<40}  {'TComp%':>7}  {'Uniq%':>6}  {'SCI%':>6}  {'TDS%':>6}  {'Blk%':>6}  {'FOV%':>6}  Flags")
    for r in video_rows:
        v = r["vals"]
        print(f"  {_clip(r['path']):<40}"
              f"  {_fv(v['tcomp']):>7}  {_fv(v['dup'], '.2f'):>6}  {_fv(v['sci']):>6}"
              f"  {_fv(v['tds']):>6}  {_fv(v['blk']):>6}  {_fv(v['fov']):>6}  {_video_flags(v)}")

    # Per-file omegaprime table
    print(f"\nPER-FILE: OMEGAPRIME (D3.1) {'─'*40}")
    print(f"  {'File':<40}  {'Attr%':>6}  {'Rec%':>6}  {'Fmt%':>6}  {'Dup%':>5}  {'TComp%':>7}  "
          f"{'Cls':>3}  {'ObjCov':>6}  {'Traj':>4}  {'FilM':>4}  Flags")
    for r in omega_rows:
        v = r["vals"]
        flags = ", ".join(
            f"{abbr}↓" for k, _, abbr in _D31_OMEGA_PASS_KEYS if v.get(k) is False
        ) or "—"
        print(f"  {_clip(r['path']):<40}"
              f"  {_fv(v['attr']):>6}  {_fv(v['rec']):>6}  {_fv(v['fmt']):>6}"
              f"  {_fv(v['dup'], '.2f'):>5}  {_fv(v['tcomp']):>7}"
              f"  {_pf(v.get('cls')):>3}  {_pf(v.get('obj_cov')):>6}"
              f"  {_pf(v.get('traj')):>4}  {_pf(v.get('fmeta')):>4}  {flags}")


def _clip(s: str, width: int = 40) -> str:
    return s if len(s) <= width else "…" + s[-(width - 1):]


def build_json(root: Path, video_rows: List[Dict], omega_rows: List[Dict], n_files: int) -> Dict:
    """Build a JSON-serialisable summary dict (mirrors print_summary structure)."""
    def agg_block(rows, keys):
        acc = _collect_agg([r["vals"] for r in rows], keys)
        return {k: _agg(acc[k]) for k in keys if acc.get(k)}

    return {
        "root": str(root),
        "scanned": str(date.today()),
        "n_files": n_files,
        "video": {
            "n_files": len(video_rows),
            "aggregated": agg_block(video_rows, list(_VIDEO_LABELS)),
        },
        "omegaprime": {
            "n_files": len(omega_rows),
            "aggregated": agg_block(omega_rows, list(_OMEGA_LABELS)),
        },
        "files": [
            {"path": r["path"], **{k: r["vals"].get(k) for k in _VIDEO_LABELS}, "kind": "video"}
            for r in video_rows
        ] + [
            {"path": r["path"], **{k: r["vals"].get(k) for k in _OMEGA_LABELS}, "kind": "omegaprime"}
            for r in omega_rows
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path, help="Root directory to scan")
    parser.add_argument("--pattern", action="append", metavar="GLOB",
                        help="Filename glob to match (repeatable; default '*.json')")
    parser.add_argument("--json", dest="json_out", type=Path, metavar="FILE",
                        help="Also write a JSON summary to FILE")
    parser.add_argument("--d31", action="store_true",
                        help="Show all D3.1 metrics with official names (class completeness, "
                             "trajectory plausibility, file metadata, object type coverage)")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"--root does not exist: {root}")

    patterns = args.pattern or ["*.json"]
    files = find_metric_files(root, patterns)
    video_rows, omega_rows = load_rows(files, root)

    if args.d31:
        print_d31_summary(root, video_rows, omega_rows, len(files))
    else:
        print_summary(root, video_rows, omega_rows, len(files))

    if args.json_out:
        data = build_json(root, video_rows, omega_rows, len(files))
        args.json_out.write_text(json.dumps(data, indent=2))
        print(f"\nJSON written to {args.json_out}")


if __name__ == "__main__":
    main()
