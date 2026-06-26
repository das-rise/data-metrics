#!/usr/bin/env python3
"""Traffic density, speed, flow and Level-of-Service metrics.

Computes traffic-flow metrics from vehicle trajectories (OmegaPrime MCAP or an
OpenLABEL JSON with world cuboids) grounded on an OpenDRIVE (.xodr) map. The map
supplies lane geometry, lengths and the drivable surface, so density is reported
in real-world units (veh/km, veh/m2) rather than a pixel proxy.

Four density definitions are produced, plus space-mean speed, flow (q = k*v) and
HCM Level-of-Service:
  - whole-scene density (veh per lane-km),
  - per-lane density + LOS,
  - area occupancy (footprint area / drivable area, and veh/m2),
  - roundabout ring occupancy and per-approach density (when a roundabout is found).
"""
import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

_VERSION = "0.1.0"
_LANE_SAMPLE_STEP_M = 1.0      # reference-line sampling resolution
_RING_RADIAL_COV_MAX = 0.15    # non-junction roads below this radial CoV form the ring
_MOVING_SPEED_MIN_MS = 0.1     # vehicles slower than this are excluded from space-mean speed

# HCM6 basic freeway-segment LOS thresholds (density in veh/km/lane). These are the
# de-facto reference grades; for urban arterials / roundabouts they are only indicative.
# Override with --los-spec (JSON list of [grade, upper_density]).
_DEFAULT_LOS_TABLE = [["A", 7], ["B", 11], ["C", 16], ["D", 22], ["E", 28], ["F", 1e9]]


# --------------------------------------------------------------------------- #
# OpenDRIVE parsing
# --------------------------------------------------------------------------- #
@dataclass
class Lane:
    """One driving lane: its drivable polygon plus length and roundabout role."""
    road_id: str
    lane_id: int
    junction: str
    polygon: Polygon
    length_m: float
    centerline: np.ndarray          # Nx2 points in map-local metres
    role: str = "normal"            # normal | ring | approach


@dataclass
class RoadNetwork:
    lanes: List[Lane]
    polygons: List[Polygon]         # parallel to lanes, for STRtree assignment
    tree: STRtree
    drivable: Polygon
    e0: float                       # map-local = global_utm - (e0, n0)
    n0: float
    crs: str
    centre: Tuple[float, float]
    has_roundabout: bool = False


def _parse_xml_no_ns(path: str) -> ET.Element:
    """Parse XML, stripping namespace prefixes from every tag."""
    it = ET.iterparse(path)
    for _, el in it:
        el.tag = el.tag.split("}")[-1]
    return it.root


def _geo_offset(geo_text: str) -> Tuple[float, float, str]:
    """Return (e0, n0, base_crs) for an OpenDRIVE geoReference.

    The map frame is the projected CRS shifted so that (lon_0, lat_0) is the origin;
    map-local = global_projected - proj(lon_0, lat_0).
    """
    lat0 = re.search(r"\+lat_0=([0-9.eE+\-]+)", geo_text)
    lon0 = re.search(r"\+lon_0=([0-9.eE+\-]+)", geo_text)
    base = re.sub(r"\s*\+(lat|lon)_0=[0-9.eE+\-]+", "", geo_text).strip()
    if not (lat0 and lon0):
        return 0.0, 0.0, base
    e0, n0 = Transformer.from_crs("EPSG:4326", base, always_xy=True).transform(
        float(lon0.group(1)), float(lat0.group(1)))
    return e0, n0, base


def _sample_geometry(g: ET.Element, step: float) -> List[Tuple[float, float, float]]:
    """Sample one planView <geometry> as (x, y, heading) points in map-local metres."""
    x0, y0 = float(g.attrib["x"]), float(g.attrib["y"])
    hdg, length = float(g.attrib["hdg"]), float(g.attrib["length"])
    n = max(2, int(math.ceil(length / step)) + 1)
    pp = g.find("paramPoly3")
    if pp is None:                              # straight line
        return [(x0 + s * math.cos(hdg), y0 + s * math.sin(hdg), hdg)
                for s in np.linspace(0.0, length, n)]
    a = pp.attrib
    cu = [float(a[k]) for k in ("aU", "bU", "cU", "dU")]
    cv = [float(a[k]) for k in ("aV", "bV", "cV", "dV")]
    pts = []
    for p in np.linspace(0.0, 1.0, n):
        u = cu[0] + cu[1] * p + cu[2] * p * p + cu[3] * p ** 3
        v = cv[0] + cv[1] * p + cv[2] * p * p + cv[3] * p ** 3
        du = cu[1] + 2 * cu[2] * p + 3 * cu[3] * p * p
        dv = cv[1] + 2 * cv[2] * p + 3 * cv[3] * p * p
        x = x0 + u * math.cos(hdg) - v * math.sin(hdg)
        y = y0 + u * math.sin(hdg) + v * math.cos(hdg)
        pts.append((x, y, hdg + math.atan2(dv, du)))
    return pts


def _road_reference(road: ET.Element, step: float) -> List[Tuple[float, float, float]]:
    """Concatenated reference-line samples across a road's planView geometries."""
    pts: List[Tuple[float, float, float]] = []
    for g in road.find("planView").findall("geometry"):
        pts.extend(_sample_geometry(g, step))
    return pts


def _polyline_length(pts: np.ndarray) -> float:
    return float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0


def _build_driving_lanes(road: ET.Element, ref: List[Tuple[float, float, float]]) -> List[Lane]:
    """Build a Lane (polygon + centreline) for each driving lane of a road.

    Lane widths are taken as constant (the 'a' coefficient); higher-order terms are
    absent in the supported maps. Right lanes (negative id) offset along -normal.
    """
    ref_arr = np.array(ref)
    normals = np.column_stack([-np.sin(ref_arr[:, 2]), np.cos(ref_arr[:, 2])])
    base = ref_arr[:, :2]
    sec = road.find("lanes/laneSection")
    lanes: List[Lane] = []
    for side, sign in (("left", 1.0), ("right", -1.0)):
        group = sec.find(side)
        if group is None:
            continue
        inner = 0.0
        for ln in sorted(group.findall("lane"), key=lambda l: abs(int(l.attrib["id"]))):
            width = float(ln.find("width").attrib["a"])
            outer = inner + width
            if ln.attrib.get("type") == "driving":
                centre = base + sign * (inner + width / 2) * normals
                inb = base + sign * inner * normals
                oub = base + sign * outer * normals
                poly = Polygon(np.vstack([oub, inb[::-1]])).buffer(0)
                if not poly.is_empty:
                    lanes.append(Lane(road.attrib["id"], int(ln.attrib["id"]),
                                      road.attrib["junction"], poly,
                                      _polyline_length(centre), centre))
            inner = outer
    return lanes


def _classify_roundabout(lanes: List[Lane], centre: Tuple[float, float]) -> bool:
    """Tag lanes as ring/approach by radial CoV about the scene centre.

    Non-junction roads whose centreline stays at near-constant radius are the
    circulatory roadway (ring); the remaining non-junction roads are approaches.
    Returns True if a ring was found.
    """
    cx, cy = centre
    by_road: Dict[str, List[Lane]] = {}
    for ln in lanes:
        by_road.setdefault(ln.road_id, []).append(ln)
    has_ring = False
    for road_lanes in by_road.values():
        if road_lanes[0].junction != "-1":
            continue                            # junction connectors are neither
        pts = np.vstack([ln.centerline for ln in road_lanes])
        radii = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        cov = radii.std() / radii.mean() if radii.mean() else 1.0
        role = "ring" if cov < _RING_RADIAL_COV_MAX else "approach"
        has_ring = has_ring or role == "ring"
        for ln in road_lanes:
            ln.role = role
    return has_ring


def parse_opendrive(path: str) -> RoadNetwork:
    """Parse an OpenDRIVE map into driving-lane geometry in a map-local metric frame."""
    root = _parse_xml_no_ns(path)
    geo = root.find("header/geoReference")
    e0, n0, crs = _geo_offset(geo.text if geo is not None else "")
    lanes: List[Lane] = []
    for road in root.findall("road"):
        ref = _road_reference(road, _LANE_SAMPLE_STEP_M)
        if ref:
            lanes.extend(_build_driving_lanes(road, ref))
    if not lanes:
        raise ValueError("OpenDRIVE map has no driving lanes")
    polygons = [ln.polygon for ln in lanes]
    drivable = unary_union(polygons)
    all_pts = np.vstack([ln.centerline for ln in lanes])
    centre = (float(all_pts[:, 0].mean()), float(all_pts[:, 1].mean()))
    has_roundabout = _classify_roundabout(lanes, centre)
    return RoadNetwork(lanes, polygons, STRtree(polygons), drivable,
                       e0, n0, crs, centre, has_roundabout)


# --------------------------------------------------------------------------- #
# Trajectory loading -> per-frame vehicle table
# --------------------------------------------------------------------------- #
_TRAJ_COLUMNS = ["frame", "id", "x", "y", "yaw", "length", "width", "vx", "vy"]


def _frames_from_time(total_nanos: np.ndarray) -> Tuple[np.ndarray, float]:
    """Map timestamps to dense frame indices and return (frames, hz)."""
    uniq = np.unique(total_nanos)
    frame = np.searchsorted(uniq, total_nanos)
    dt = float(np.median(np.diff(uniq))) / 1e9 if len(uniq) > 1 else 0.0
    return frame, (1.0 / dt if dt > 0 else float("nan"))


def load_omegaprime(path: str):
    """Load an OmegaPrime MCAP into the unified per-frame vehicle table (+ hz).

    Inlined (rather than importing data_metrics) to avoid pulling in cv2/av.
    """
    import pandas as pd
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    rows = []
    with open(path, "rb") as f:
        for _, _, _, gt in make_reader(f, decoder_factories=[DecoderFactory()]) \
                .iter_decoded_messages(topics=["ground_truth"]):
            t = gt.timestamp.seconds * 1_000_000_000 + gt.timestamp.nanos
            for mo in gt.moving_object:
                b = mo.base
                rows.append((t, int(mo.id.value), b.position.x, b.position.y,
                             b.orientation.yaw, b.dimension.length, b.dimension.width,
                             b.velocity.x, b.velocity.y))
    df = pd.DataFrame(rows, columns=["t", "id", "x", "y", "yaw", "length", "width", "vx", "vy"])
    df["frame"], hz = _frames_from_time(df["t"].to_numpy())
    return df[_TRAJ_COLUMNS], hz


def _read_json_any(path: str):
    """Load a JSON file, transparently handling gzip / zip / bz2 / xz compression.

    The container is detected from the leading magic bytes (not the extension), so a
    compressed OpenLABEL export works with any filename. A zip is read from its first
    `.json` member (else its first file).
    """
    with open(path, "rb") as f:
        magic = f.read(6)
    if magic[:2] == b"\x1f\x8b":
        import gzip
        opener = lambda: gzip.open(path, "rb")
    elif magic[:3] == b"BZh":
        import bz2
        opener = lambda: bz2.open(path, "rb")
    elif magic[:6] == b"\xfd7zXZ\x00":
        import lzma
        opener = lambda: lzma.open(path, "rb")
    elif magic[:4] == b"PK\x03\x04":
        import zipfile
        zf = zipfile.ZipFile(path)
        names = [n for n in zf.namelist() if not n.endswith("/")]
        member = next((n for n in names if n.lower().endswith(".json")), names[0])
        opener = lambda: zf.open(member)
    else:
        opener = lambda: open(path, "rb")
    with opener() as fh:
        return json.load(fh)


def load_openlabel(path: str, expected_hz: Optional[float]):
    """Load an OpenLABEL JSON (world cuboids) into the table; velocity by finite diff.

    The file may be plain or gzip/zip/bz2/xz compressed (auto-detected).
    """
    import pandas as pd

    ol = _read_json_any(path)["openlabel"]
    rows = []
    for fkey, frame in ol["frames"].items():
        fi = int(fkey)
        for oid, obj in frame.get("objects", {}).items():
            cub = obj.get("object_data", {}).get("cuboid")
            if not cub:
                continue
            v = cub[0]["val"]                   # [x, y, z, roll, pitch, yaw, L, W, H]
            rows.append((fi, int(oid), v[0], v[1], v[5], v[6], v[7]))
    df = pd.DataFrame(rows, columns=["frame", "id", "x", "y", "yaw", "length", "width"])
    hz = expected_hz or 30.0
    df = df.sort_values(["id", "frame"])
    df["vx"] = df.groupby("id")["x"].diff() * hz
    df["vy"] = df.groupby("id")["y"].diff() * hz
    return df[_TRAJ_COLUMNS], hz


def load_trajectories(omegaprime: Optional[str], openlabel: Optional[str],
                      expected_hz: Optional[float]):
    """Dispatch to the right loader; coordinate shift to the map frame is the caller's job."""
    if omegaprime:
        return load_omegaprime(omegaprime)
    return load_openlabel(openlabel, expected_hz)


# --------------------------------------------------------------------------- #
# Lane assignment
# --------------------------------------------------------------------------- #
def assign_lanes(net: RoadNetwork, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Return the lane index (into net.lanes) containing each (x, y), or -1."""
    pts = [Point(x, y) for x, y in zip(xs, ys)]
    assigned = np.full(len(pts), -1, dtype=int)
    # Bbox-prefilter via the STRtree, then confirm with an explicit point-in-polygon
    # test (predicate semantics vary across shapely versions; this is robust).
    in_idx, tree_idx = net.tree.query(pts)
    for ip, it in zip(in_idx, tree_idx):
        if assigned[ip] == -1 and net.polygons[it].contains(pts[ip]):
            assigned[ip] = it
    return assigned


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def _summary(series: np.ndarray) -> Dict:
    """Mean / peak / p95 / min summary of a per-frame series."""
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return {"mean": None, "peak": None, "p95": None, "min": None}
    return {"mean": round(float(s.mean()), 4), "peak": round(float(s.max()), 4),
            "p95": round(float(np.percentile(s, 95)), 4), "min": round(float(s.min()), 4)}


def _los_grade(density: float, table: List) -> str:
    for grade, upper in table:
        if density <= upper:
            return grade
    return table[-1][0]


def _harmonic_mean_speed(speeds: np.ndarray) -> float:
    """Space-mean speed (m/s): harmonic mean over moving vehicles."""
    moving = speeds[speeds > _MOVING_SPEED_MIN_MS]
    return float(len(moving) / np.sum(1.0 / moving)) if moving.size else 0.0


def _frame_groups(frames: np.ndarray, n_frames: int) -> List[np.ndarray]:
    """Row indices for each frame id (0..n_frames-1)."""
    order = np.argsort(frames, kind="stable")
    sorted_f = frames[order]
    bounds = np.searchsorted(sorted_f, np.arange(n_frames + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(n_frames)]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_whole_scene_density(per_frame_counts: np.ndarray, total_lane_km: float,
                                los_table: List, want_series: bool) -> Dict:
    """Network-wide density (veh per lane-km) and its LOS."""
    density = per_frame_counts / total_lane_km if total_lane_km else per_frame_counts * np.nan
    out = {"total_lane_km": round(total_lane_km, 4),
           "density_veh_per_lane_km": _summary(density),
           "mean_level_of_service": _los_grade(float(np.nanmean(density)), los_table),
           "vehicles_on_road": _summary(per_frame_counts)}
    if want_series:
        out["time_series"] = [round(float(d), 4) for d in density]
    return out


def _lane_entry(road: str, lane_id: Optional[int], role: str, length_m: float,
                count_series: np.ndarray, los_table: List) -> Dict:
    """Density (veh/km) + LOS summary for one lane or merged group."""
    dens = count_series / (length_m / 1000.0)
    return {"road": road, "lane": lane_id, "role": role,
            "length_m": round(length_m, 1),
            "mean_density": round(float(dens.mean()), 3),
            "peak_density": round(float(dens.max()), 3),
            "peak_level_of_service": _los_grade(float(dens.max()), los_table)}


def _lane_merge_key(ln: Lane) -> Optional[Tuple]:
    """Group short multi-segment structures so density isn't quantisation-dominated.

    Ring arcs merge into one 'ring' road; each junction's connector lanes merge per
    junction. Ordinary non-junction lanes (returns None) stay individual.
    """
    if ln.role == "ring":
        return ("ring", "ring", "ring")
    if ln.junction != "-1":
        return ("j" + ln.junction, f"junction {ln.junction}", "junction")
    return None


def compute_per_lane_density(net: RoadNetwork, lane_idx: np.ndarray, frame_groups: List,
                             los_table: List) -> Dict:
    """Per-lane mean/peak density (veh/km) and worst-case LOS.

    Ring arcs and each junction's connector lanes are merged (see _lane_merge_key):
    instantaneous density on an isolated few-metre segment is otherwise dominated by
    quantisation. Ordinary lanes are reported individually.
    """
    n_frames = len(frame_groups)
    counts = np.zeros((len(net.lanes), n_frames))
    for fi, rows in enumerate(frame_groups):
        li = lane_idx[rows]
        for k in li[li >= 0]:
            counts[k, fi] += 1
    lanes_out, merged = [], {}
    for k, ln in enumerate(net.lanes):
        key = _lane_merge_key(ln)
        if key is None:
            if ln.length_m > 0:
                lanes_out.append(_lane_entry(ln.road_id, ln.lane_id, ln.role,
                                             ln.length_m, counts[k], los_table))
            continue
        grp = merged.setdefault(key[0], [np.zeros(n_frames), 0.0, key[1], key[2]])
        grp[0] += counts[k]
        grp[1] += ln.length_m
    for cnt, length, label, role in merged.values():
        if length > 0:
            lanes_out.append(_lane_entry(label, None, role, length, cnt, los_table))
    lanes_out.sort(key=lambda d: d["peak_density"], reverse=True)
    return {"n_lanes": len(lanes_out), "lanes": lanes_out}


def compute_area_occupancy(footprint_per_frame: np.ndarray, counts: np.ndarray,
                           drivable_area_m2: float, want_series: bool) -> Dict:
    """Footprint-area occupancy fraction and veh/m2 over the drivable surface."""
    occ = footprint_per_frame / drivable_area_m2 if drivable_area_m2 else footprint_per_frame * np.nan
    veh_m2 = counts / drivable_area_m2 if drivable_area_m2 else counts * np.nan
    out = {"drivable_area_m2": round(drivable_area_m2, 1),
           "area_occupancy_fraction": _summary(occ),
           "veh_per_m2": _summary(veh_m2)}
    if want_series:
        out["time_series_occupancy"] = [round(float(o), 5) for o in occ]
    return out


def compute_speed_flow(speed_per_frame: np.ndarray, density_per_frame: np.ndarray) -> Dict:
    """Space-mean speed (km/h) and flow q = k*v (veh/h per lane)."""
    speed_kmh = speed_per_frame * 3.6
    flow = density_per_frame * speed_kmh        # veh/lane-km * km/h = veh/h/lane
    return {"space_mean_speed_kmh": _summary(speed_kmh),
            "flow_veh_per_h_per_lane": _summary(flow)}


def compute_roundabout(net: RoadNetwork, lane_idx: np.ndarray, frame_groups: List,
                       footprint: np.ndarray) -> Dict:
    """Ring occupancy and per-approach density for a detected roundabout."""
    if not net.has_roundabout:
        return {"detected": False}
    ring_k = [k for k, ln in enumerate(net.lanes) if ln.role == "ring"]
    ring_km = sum(net.lanes[k].length_m for k in ring_k) / 1000.0
    ring_area = unary_union([net.lanes[k].polygon for k in ring_k]).area
    ring_set = set(ring_k)
    approaches: Dict[str, List[int]] = {}
    for k, ln in enumerate(net.lanes):
        if ln.role == "approach":
            approaches.setdefault(ln.road_id, []).append(k)

    ring_counts, ring_occ, app_series = [], [], {r: [] for r in approaches}
    for rows in frame_groups:
        li = lane_idx[rows]
        ring_mask = np.isin(li, list(ring_set))
        ring_counts.append(int(ring_mask.sum()))
        ring_occ.append(float(footprint[rows][ring_mask].sum()) / ring_area if ring_area else 0.0)
        for r, ks in approaches.items():
            app_series[r].append(int(np.isin(li, ks).sum()))
    app_out = []
    for r, ks in approaches.items():
        km = sum(net.lanes[k].length_m for k in ks) / 1000.0
        dens = np.array(app_series[r]) / km if km else np.array(app_series[r]) * np.nan
        app_out.append({"road": r, "length_m": round(km * 1000, 1),
                        "density_veh_per_km": _summary(dens)})
    return {"detected": True, "ring_lane_km": round(ring_km, 3),
            "ring_vehicle_count": _summary(np.array(ring_counts)),
            "ring_density_veh_per_km": _summary(np.array(ring_counts) / ring_km if ring_km else []),
            "ring_area_occupancy_fraction": _summary(np.array(ring_occ)),
            "approaches": app_out}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def compute_traffic_density(omegaprime: Optional[str], openlabel: Optional[str],
                            opendrive: str, expected_hz: Optional[float],
                            los_table: List, want_series: bool) -> Dict:
    """Run the full pipeline and return the metrics dict."""
    net = parse_opendrive(opendrive)
    df, hz = load_trajectories(omegaprime, openlabel, expected_hz)
    xs = df["x"].to_numpy() - net.e0
    ys = df["y"].to_numpy() - net.n0
    lane_idx = assign_lanes(net, xs, ys)

    frames = df["frame"].to_numpy()
    n_frames = int(frames.max()) + 1 if len(frames) else 0
    footprint = (df["length"].to_numpy() * df["width"].to_numpy())
    speed = np.hypot(df["vx"].to_numpy(), df["vy"].to_numpy())
    groups = _frame_groups(frames, n_frames)

    on_counts = np.array([int((lane_idx[r] >= 0).sum()) for r in groups], dtype=float)
    fp_frame = np.array([footprint[r][lane_idx[r] >= 0].sum() for r in groups])
    sp_frame = np.array([_harmonic_mean_speed(speed[r][lane_idx[r] >= 0]) for r in groups])
    total_lane_km = sum(ln.length_m for ln in net.lanes) / 1000.0
    density = on_counts / total_lane_km if total_lane_km else on_counts * np.nan

    return {
        "provenance": {
            "tool": "traffic_density", "version": _VERSION,
            "source": omegaprime or openlabel,
            "source_format": "OmegaPrime MCAP" if omegaprime else "OpenLABEL JSON",
            "opendrive": opendrive, "map_crs": net.crs,
            "frames": n_frames, "sampling_hz": round(hz, 4) if hz == hz else None,
            "lanes": len(net.lanes), "roundabout_detected": net.has_roundabout,
        },
        "whole_scene": compute_whole_scene_density(on_counts, total_lane_km, los_table, want_series),
        "per_lane": compute_per_lane_density(net, lane_idx, groups, los_table),
        "area_occupancy": compute_area_occupancy(fp_frame, on_counts, net.drivable.area, want_series),
        "speed_flow": compute_speed_flow(sp_frame, density),
        "roundabout": compute_roundabout(net, lane_idx, groups, footprint),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Traffic density / speed / flow / LOS metrics from trajectories + an OpenDRIVE map.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--omegaprime", metavar="FILE", help="OmegaPrime MCAP trajectory source")
    src.add_argument("--openlabel", metavar="FILE", help="OpenLABEL JSON trajectory source (world cuboids)")
    parser.add_argument("--opendrive", metavar="FILE", required=True, help="OpenDRIVE .xodr map")
    parser.add_argument("--expected-hz", type=float, default=None,
                        help="Frame rate; used for OpenLABEL velocity finite-difference (default 30).")
    parser.add_argument("--los-spec", metavar="FILE",
                        help="JSON list of [grade, upper_density_veh_per_km_lane] overriding the HCM defaults.")
    parser.add_argument("--time-series", action="store_true",
                        help="Include per-frame density/occupancy arrays in the output.")
    parser.add_argument("--json", dest="json_out", metavar="FILE", help="Write result JSON to FILE.")
    args = parser.parse_args()

    los_table = _DEFAULT_LOS_TABLE
    if args.los_spec:
        with open(args.los_spec) as f:
            los_table = json.load(f)

    try:
        result = compute_traffic_density(args.omegaprime, args.openlabel, args.opendrive,
                                         args.expected_hz, los_table, args.time_series)
    except Exception as exc:                    # surface failure without a traceback dump
        result = {"error": f"{type(exc).__name__}: {exc}"}

    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text)


if __name__ == "__main__":
    main()
