#!/usr/bin/env python3
"""Traffic density, flow and roundabout Level-of-Service metrics.

Computes traffic-flow metrics from vehicle trajectories (OmegaPrime MCAP or an
OpenLABEL JSON with world cuboids) grounded on an OpenDRIVE (.xodr) map. The map
supplies lane geometry, lengths and the drivable surface, so everything is in
physical units rather than a pixel proxy. Trajectories are first cleaned (short
flicker tracks, teleport samples, and parked/static vehicles removed). Output:
  - data_quality: noise/reliability indicators (static, short, teleport, off-lane),
  - geometry_agnostic: area occupancy + veh/m2 (need only the drivable polygon),
  - network: Edie space-time density/flow/speed over all driving lanes,
  - by_type: per car/van/truck/bus class density/flow,
  - roads: per-road segment density/flow/speed,
  - junctions: turning movements (in->out counts) per junction,
  - roundabout: ring Edie + delay/LOS, per-approach entry/exit + conflicting flow.

Density/flow/speed use Edie's generalized definitions (robust to short segments);
LOS is delay-based (the correct scale for a roundabout), measured from trajectories.
A roundabout is only reported when the ring forms a near-complete loop.
"""
import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import opendrive_map as odm
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

_VERSION = "0.3.0"
_LANE_SAMPLE_STEP_M = 1.0      # reference-line sampling resolution
_RING_CIRC_MIN = 0.75          # min circumferential fraction for a lane to count as ring
_RING_SECTOR_COVERAGE = 0.75   # fraction of 10-deg sectors a real ring must cover (loop check)
_FREE_FLOW_PERCENTILE = 85     # speed percentile per region used as the free-flow reference

# Trajectory-cleaning defaults (overridable on the CLI). The MCAP carries no confidence,
# so cleaning is based on track duration, lifetime motion, and per-sample speed.
_MIN_TRACK_S = 0.5            # tracks shorter than this are dropped (flicker false positives)
_STATIC_DISP_M = 2.0         # tracks whose centre moves less than this over their life are parked
_MAX_SPEED_MS = 55.0         # samples faster than this are teleport / ID-switch artifacts

# HCM roundabout / unsignalized-intersection LOS by control delay (s/veh). This is the
# correct delay-based scale for interrupted flow (a roundabout), unlike freeway density LOS.
# Override with --los-spec (JSON list of [grade, upper_delay_seconds]).
_DELAY_LOS_TABLE = [["A", 10], ["B", 15], ["C", 25], ["D", 35], ["E", 50], ["F", 1e9]]

# OSI vehicle_classification.type -> coarse group. Unlisted codes fall back to "other".
_OSI_VEHICLE_GROUP = {2: "car", 3: "car", 4: "car", 5: "car", 6: "van", 7: "truck",
                      8: "truck", 9: "truck", 10: "motorcycle", 11: "bicycle",
                      12: "bus", 13: "bus", 14: "bus"}
# OpenLABEL object-type string -> the same coarse groups.
_OPENLABEL_VEHICLE_GROUP = {"car": "car", "vehicle": "car", "van": "van", "truck": "truck",
                            "trailer": "truck", "bus": "bus", "motorcycle": "motorcycle",
                            "bicycle": "bicycle"}


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
    junction_names: Dict[str, str] = field(default_factory=dict)
    parking_polys: List[Polygon] = field(default_factory=list)
    parking_tree: Optional[STRtree] = None


def _strip_ns(root: ET.Element) -> ET.Element:
    """Strip namespace prefixes from every tag in a parsed tree, in place."""
    for el in root.iter():
        el.tag = el.tag.split("}")[-1]
    return root


def _base_crs(geo_reference: Optional[str]) -> str:
    """Projected CRS string with the (redundant for UTM) +lat_0/+lon_0 stripped."""
    if not geo_reference:
        return ""
    return re.sub(r"\s*\+(lat|lon)_0=[0-9.eE+\-]+", "", geo_reference).strip()


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


def _circumferential_fraction(centerline: np.ndarray, centre: Tuple[float, float]) -> float:
    """Length-weighted fraction of a lane that runs tangentially about the centre.

    1.0 = a constant-radius arc (circulatory roadway); 0.0 = a pure radial spoke. Computed
    as 1 − |unit_tangent · unit_radial| averaged over segments.
    """
    seg = np.diff(centerline, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    if seglen.sum() <= 0:
        return 0.0
    safe = np.where(seglen == 0, 1.0, seglen)
    direction = seg / safe[:, None]
    mid = (centerline[:-1] + centerline[1:]) / 2 - np.asarray(centre)
    radius = np.hypot(mid[:, 0], mid[:, 1])
    radial = mid / np.where(radius == 0, 1.0, radius)[:, None]
    radial_comp = np.abs((direction * radial).sum(axis=1))
    return float(1.0 - np.average(radial_comp, weights=seglen))


def _assign_roles(by_road: Dict[str, List[Lane]], centre: Tuple[float, float]) -> List[Lane]:
    """Tag each road's lanes ring / approach / normal by circumferential fraction; return ring."""
    ring: List[Lane] = []
    for road_lanes in by_road.values():
        circ = float(np.mean([_circumferential_fraction(ln.centerline, centre)
                              for ln in road_lanes]))
        if circ >= _RING_CIRC_MIN:
            role = "ring"
            ring += road_lanes
        elif road_lanes[0].junction == "-1":
            role = "approach"
        else:
            role = "normal"
        for ln in road_lanes:
            ln.role = role
    return ring


def _ring_sector_coverage(ring: List[Lane], centre: Tuple[float, float]) -> float:
    """Fraction of 36 ten-degree sectors about the centre that the ring lanes occupy.

    A real circulatory roadway is a near-complete loop (≈1.0); a short curved connector
    spuriously flagged as circumferential covers only a sliver.
    """
    pts = np.vstack([ln.centerline for ln in ring])
    ang = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    sectors = np.floor((ang + np.pi) / (np.pi / 18)).astype(int)
    return len(set(sectors.tolist())) / 36.0


def _classify_roundabout(lanes: List[Lane],
                         centre: Tuple[float, float]) -> Tuple[bool, Tuple[float, float]]:
    """Detect a roundabout and tag lane roles; return (has_roundabout, centre).

    Ring lanes run circumferentially about the centre (plain ring roads plus the junction
    connectors that carry the ring through an entry/exit). A candidate ring is accepted only
    if it forms a near-complete loop (sector coverage ≥ _RING_SECTOR_COVERAGE) — this rejects
    short curved connectors on straight-road scenes. The centre is refined to the candidate
    ring's centroid before the loop check. On rejection, ring lanes are demoted.
    """
    by_road: Dict[str, List[Lane]] = {}
    for ln in lanes:
        by_road.setdefault(ln.road_id, []).append(ln)
    ring = _assign_roles(by_road, centre)
    if ring:
        centre = (float(np.vstack([ln.centerline for ln in ring])[:, 0].mean()),
                  float(np.vstack([ln.centerline for ln in ring])[:, 1].mean()))
        ring = _assign_roles(by_road, centre)
    if ring and _ring_sector_coverage(ring, centre) >= _RING_SECTOR_COVERAGE:
        return True, centre
    for ln in lanes:                                   # no real loop → demote any ring lanes
        if ln.role == "ring":
            ln.role = "approach" if ln.junction == "-1" else "normal"
    return False, centre


def parse_opendrive(path: str) -> RoadNetwork:
    """Parse an OpenDRIVE map file into driving-lane geometry (map-local metric frame)."""
    od = odm.RoadNetwork.from_file(path, lane_types=["driving"], interval=_LANE_SAMPLE_STEP_M)
    return _network_from_odmap(od, _strip_ns(ET.parse(path).getroot()))


def parse_opendrive_text(xml: str) -> RoadNetwork:
    """Parse an OpenDRIVE map from an in-memory XML string (e.g. embedded in an MCAP)."""
    od = odm.RoadNetwork.from_text(xml, lane_types=["driving"], interval=_LANE_SAMPLE_STEP_M)
    return _network_from_odmap(od, _strip_ns(ET.fromstring(xml)))


def _road_point_at_s(road: ET.Element, s: float) -> Tuple[float, float, float]:
    """(x, y, heading) at reference-line arc-length `s` on a road (sampled, interpolated)."""
    pts = _road_reference(road, 0.5)
    xy = np.array([(p[0], p[1]) for p in pts])
    hd = np.array([p[2] for p in pts])
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    i = min(int(np.searchsorted(cum, s)), len(xy) - 1)
    return xy[i, 0], xy[i, 1], hd[i]


def _parking_areas(root: ET.Element) -> List[Polygon]:
    """World-frame polygons for every `<object type="parking">` outline in the map.

    An object sits at arc-length `s`, lateral offset `t` and heading `hdg` on its road; its
    `cornerLocal` (u, v) corners are placed by the road heading there. These are the parking
    lots your map renders — vehicles inside them are parking, not road traffic.
    """
    polys: List[Polygon] = []
    for road in root.findall("road"):
        for obj in road.findall(".//object"):
            corners = obj.findall(".//cornerLocal")
            if obj.attrib.get("type") != "parking" or len(corners) < 3:
                continue
            x0, y0, th = _road_point_at_s(road, float(obj.attrib.get("s", 0)))
            t, hdg = float(obj.attrib.get("t", 0)), float(obj.attrib.get("hdg", 0))
            ox, oy = x0 - math.sin(th) * t, y0 + math.cos(th) * t
            ca, sa = math.cos(th + hdg), math.sin(th + hdg)
            pts = [(ox + u * ca - v * sa, oy + u * sa + v * ca)
                   for u, v in ((float(c.attrib["u"]), float(c.attrib["v"])) for c in corners)]
            poly = Polygon(pts).buffer(0)
            if not poly.is_empty:
                polys.append(poly)
    return polys


def _network_from_odmap(od: "odm.RoadNetwork", root: ET.Element) -> RoadNetwork:
    """Adapt an opendrive-map network into the metrics RoadNetwork (lanes + domain logic).

    Lane geometry, the authoritative header <offset> and the CRS come from opendrive-map;
    roundabout roles, parking areas and junction names are metrics-domain logic kept here.
    """
    road_junction = {r.id: r.junction for r in od.roads}
    lanes = [
        Lane(ml.road_id, ml.lane_id, road_junction.get(ml.road_id, "-1"),
             ml.polygon, ml.length_m, ml.centerline)
        for ml in od.lanes
    ]
    if not lanes:
        raise ValueError("OpenDRIVE map has no driving lanes")
    polygons = [ln.polygon for ln in lanes]
    drivable = unary_union(polygons)
    all_pts = np.vstack([ln.centerline for ln in lanes])
    centre = (float(all_pts[:, 0].mean()), float(all_pts[:, 1].mean()))
    has_roundabout, centre = _classify_roundabout(lanes, centre)
    junction_names = {j.attrib["id"]: j.attrib.get("name", "")
                      for j in root.findall("junction")}
    parking = _parking_areas(root)
    # Authoritative map offset comes from the header <offset>, not the redundant
    # geoReference +lat_0/+lon_0 (which may be absent on standard UTM maps).
    e0, n0, _z = od.offset
    crs = _base_crs(od.geo_reference)
    return RoadNetwork(lanes, polygons, STRtree(polygons), drivable, e0, n0, crs, centre,
                       has_roundabout, junction_names, parking,
                       STRtree(parking) if parking else None)


# --------------------------------------------------------------------------- #
# Trajectory loading -> per-frame vehicle table
# --------------------------------------------------------------------------- #
_TRAJ_COLUMNS = ["frame", "id", "x", "y", "yaw", "length", "width", "vx", "vy", "vclass"]


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
                             b.velocity.x, b.velocity.y,
                             _OSI_VEHICLE_GROUP.get(int(mo.vehicle_classification.type), "other")))
    df = pd.DataFrame(rows, columns=["t", "id", "x", "y", "yaw", "length", "width",
                                     "vx", "vy", "vclass"])
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


def load_opendrive_from_mcap(path: str) -> Optional[str]:
    """Return the OpenDRIVE XML embedded in an OmegaPrime MCAP, or None if absent.

    OmegaPrime stores the map as osi3.MapAsamOpenDrive on the 'ground_truth_map' topic.
    """
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    with open(path, "rb") as f:
        for _, _, _, pb in make_reader(f, decoder_factories=[DecoderFactory()]) \
                .iter_decoded_messages(topics=["ground_truth_map"]):
            xml = getattr(pb, "open_drive_xml_content", "")
            if xml:
                return xml
    return None


def load_openlabel(path: str, expected_hz: Optional[float]):
    """Load an OpenLABEL JSON (world cuboids) into the table; velocity by finite diff.

    The file may be plain or gzip/zip/bz2/xz compressed (auto-detected).
    """
    import pandas as pd

    ol = _read_json_any(path)["openlabel"]
    vclass = {int(oid): _OPENLABEL_VEHICLE_GROUP.get(o.get("type", "").lower(), "other")
              for oid, o in ol.get("objects", {}).items()}
    rows = []
    for fkey, frame in ol["frames"].items():
        fi = int(fkey)
        for oid, obj in frame.get("objects", {}).items():
            cub = obj.get("object_data", {}).get("cuboid")
            if not cub:
                continue
            v = cub[0]["val"]                   # [x, y, z, roll, pitch, yaw, L, W, H]
            rows.append((fi, int(oid), v[0], v[1], v[5], v[6], v[7],
                         vclass.get(int(oid), "other")))
    df = pd.DataFrame(rows, columns=["frame", "id", "x", "y", "yaw", "length", "width", "vclass"])
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


def clean_trajectories(df, hz: float, min_track_s: float, static_disp_m: float,
                       max_speed_ms: float) -> Tuple[np.ndarray, Dict]:
    """Classify tracks and return (keep_mask, stats) for trajectory cleaning.

    `keep_mask` selects the samples used for traffic metrics: tracks long enough to be real
    (drops flicker false positives), that actually move over their life (parked/static cars
    are excluded — but queued vehicles move eventually and are kept), with teleport / ID-switch
    samples removed. Cleaning is uniform across sources; thresholds are CLI-configurable.
    """
    ids = df["id"].to_numpy()
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    speed = np.nan_to_num(np.hypot(df["vx"].to_numpy(), df["vy"].to_numpy()))
    uniq, inv, counts = np.unique(ids, return_inverse=True, return_counts=True)
    dur = counts / hz

    def _span(a):
        lo = np.full(len(uniq), np.inf)
        hi = np.full(len(uniq), -np.inf)
        np.minimum.at(lo, inv, a)
        np.maximum.at(hi, inv, a)
        return hi - lo
    disp = np.hypot(_span(x), _span(y))
    short = dur < min_track_s
    static = (disp < static_disp_m) & ~short
    teleport = speed > max_speed_ms
    keep = ~(short | static)[inv] & ~teleport
    stats = {
        "n_objects": int(len(uniq)),
        "n_static": int(static.sum()),
        "n_short_tracks": int(short.sum()),
        "n_teleport_samples": int(teleport.sum()),
        "samples_kept_fraction": round(float(keep.mean()), 4) if keep.size else 0.0,
        "track_duration_s": {"median": round(float(np.median(dur)), 2),
                             "p10": round(float(np.percentile(dur, 10)), 2)},
    }
    return keep, stats


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


def in_parking(net: RoadNetwork, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Boolean per-sample: inside a mapped parking area (not road traffic)."""
    if not net.parking_polys:
        return np.zeros(len(xs), dtype=bool)
    pts = [Point(x, y) for x, y in zip(xs, ys)]
    out = np.zeros(len(pts), dtype=bool)
    in_idx, tree_idx = net.parking_tree.query(pts)
    for ip, it in zip(in_idx, tree_idx):
        if not out[ip] and net.parking_polys[it].contains(pts[ip]):
            out[ip] = True
    return out


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


def _los_grade(value: float, table: List) -> str:
    """Grade a value (control delay, s/veh) against an ascending threshold table."""
    for grade, upper in table:
        if value <= upper:
            return grade
    return table[-1][0]


def _edie(n_samples: int, sum_speed_ms: float, n_frames: int, length_km: float) -> Dict:
    """Edie's generalized density / flow / speed over a space-time region.

    A region is a set of lanes (total `length_km`) observed for `n_frames` at fixed dt.
    Each in-region vehicle-frame contributes dt of time and v*dt of distance, so
        k = total_time/(L*T) = mean_count/L     [veh/km]
        v = total_distance/total_time = mean sample speed   [km/h]
        q = total_distance/(L*T) = k*v          [veh/h]
    These are robust to short segments (unlike an instantaneous count/length).
    """
    if length_km <= 0 or n_frames <= 0 or n_samples == 0:
        return {"density_veh_per_km": 0.0, "flow_veh_per_h": 0.0, "speed_kmh": 0.0}
    k = n_samples / (n_frames * length_km)
    v_kmh = (sum_speed_ms / n_samples) * 3.6
    return {"density_veh_per_km": round(k, 3),
            "flow_veh_per_h": round(k * v_kmh, 1),
            "speed_kmh": round(v_kmh, 2)}


def _control_delay(mask: np.ndarray, ids: np.ndarray, speed: np.ndarray, hz: float) -> float:
    """Mean control delay (s/veh) for vehicles traversing a masked region.

    Free-flow speed is the region's _FREE_FLOW_PERCENTILE speed; per vehicle the delay is
    dt*Σ(1 − v/v_ff) (time lost relative to free flow), clamped at 0 and averaged.
    """
    sp = speed[mask]
    if sp.size == 0:
        return 0.0
    v_ff = float(np.percentile(sp, _FREE_FLOW_PERCENTILE))
    if v_ff <= 0:
        return 0.0
    lost = (1.0 - sp / v_ff) / hz                       # seconds lost per sample
    vid = ids[mask]
    order = np.argsort(vid, kind="stable")
    starts = np.unique(vid[order], return_index=True)[1]
    per_vehicle = np.add.reduceat(lost[order], starts)
    return round(float(np.clip(per_vehicle, 0.0, None).mean()), 2)


def _wrap(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_geometry_agnostic(footprint_pf: np.ndarray, counts_pf: np.ndarray,
                              drivable_area_m2: float, want_series: bool) -> Dict:
    """Topology-free indicators: only need the drivable polygon, comparable across scenes."""
    occ = footprint_pf / drivable_area_m2
    out = {"drivable_area_m2": round(drivable_area_m2, 1),
           "area_occupancy_fraction": _summary(occ),
           "veh_per_m2": _summary(counts_pf / drivable_area_m2)}
    if want_series:
        out["time_series_occupancy"] = [round(float(o), 5) for o in occ]
    return out


def compute_network(on_mask: np.ndarray, ids: np.ndarray, speed: np.ndarray,
                    counts_pf: np.ndarray, n_frames: int, total_lane_km: float,
                    want_series: bool) -> Dict:
    """Whole drivable network: robust Edie k/q/v plus instantaneous-density summary."""
    density_pf = counts_pf / total_lane_km if total_lane_km else counts_pf * np.nan
    out = {"total_lane_km": round(total_lane_km, 4),
           "edie": _edie(int(on_mask.sum()), float(speed[on_mask].sum()), n_frames, total_lane_km),
           "instantaneous_density_veh_per_lane_km": _summary(density_pf),
           "vehicles_on_road": _summary(counts_pf)}
    if want_series:
        out["time_series_density"] = [round(float(d), 4) for d in density_pf]
    return out


def compute_data_quality(stats: Dict, keep: np.ndarray, lane_idx_full: np.ndarray,
                         vclass: np.ndarray, ids: np.ndarray, parking: np.ndarray) -> Dict:
    """Trajectory-quality indicators — how clean / reliable the (pseudo-)labelled set is.

    `off_lane_fraction` is over kept-moving samples and already excludes parking, so it flags
    genuine driving-lane coverage gaps rather than parked / manoeuvring cars.
    """
    off_lane = float((lane_idx_full[keep] < 0).mean()) if keep.any() else 0.0
    out = dict(stats)
    out["off_lane_fraction"] = round(off_lane, 4)
    out["unknown_type_objects"] = int(np.unique(ids[vclass == "other"]).size)
    out["parking_samples"] = int(parking.sum())
    out["parking_vehicles"] = int(np.unique(ids[parking]).size)
    out["removed"] = {"short_tracks": stats["n_short_tracks"],
                      "static_vehicles": stats["n_static"],
                      "teleport_samples": stats["n_teleport_samples"],
                      "parking_samples": int(parking.sum())}
    return out


def compute_by_type(vclass: np.ndarray, on_mask: np.ndarray, ids: np.ndarray,
                    speed: np.ndarray, n_frames: int, total_lane_km: float) -> Dict:
    """Per coarse vehicle class: on-road object count and Edie density/flow/speed."""
    out = {}
    for grp in sorted(set(vclass[on_mask].tolist())):
        m = on_mask & (vclass == grp)
        out[grp] = {"count": int(np.unique(ids[m]).size),
                    "edie": _edie(int(m.sum()), float(speed[m].sum()), n_frames, total_lane_km)}
    return out


def compute_roads(net: RoadNetwork, road: np.ndarray, speed: np.ndarray,
                  n_frames: int) -> List[Dict]:
    """Per driving-road Edie density/flow/speed — generic per-segment density for any scene."""
    lane_km = {}
    for ln in net.lanes:
        lane_km[ln.road_id] = lane_km.get(ln.road_id, 0.0) + ln.length_m / 1000.0
    out = []
    for rid in np.unique(road[road != ""]):
        m = road == rid
        km = lane_km.get(rid, 0.0)
        if km <= 0 or not m.any():
            continue
        out.append({"road": str(rid), "lane_km": round(km, 3),
                    "edie": _edie(int(m.sum()), float(speed[m].sum()), n_frames, km)})
    out.sort(key=lambda d: d["edie"]["flow_veh_per_h"], reverse=True)
    return out


def compute_turning_movements(net: RoadNetwork, ids: np.ndarray, frames: np.ndarray,
                              road: np.ndarray, duration_h: float) -> List[Dict]:
    """Per-junction (in_road → out_road) movement counts from vehicle road sequences.

    For each vehicle, the ordered sequence of roads it visits is collapsed (dropping off-road
    gaps); whenever it crosses a junction connector road, the roads immediately before and
    after are recorded as one turning movement at that junction.
    """
    conn = {ln.road_id: ln.junction for ln in net.lanes if ln.junction != "-1"}
    if not conn:
        return []
    from collections import Counter
    moves: Dict[str, Counter] = {}
    order = np.lexsort((frames, ids))
    sid, sroad = ids[order], road[order]
    bounds = np.flatnonzero(np.diff(sid)) + 1
    for seg in np.split(np.arange(len(sid)), bounds):
        seq = [r for j, r in enumerate(sroad[seg]) if r != "" and (j == 0 or r != sroad[seg][j - 1])]
        for k in range(1, len(seq) - 1):
            if seq[k] in conn:
                moves.setdefault(conn[seq[k]], Counter())[(seq[k - 1], seq[k + 1])] += 1
    out = []
    for jid, cnt in moves.items():
        movements = [{"from": a, "to": b, "count": c, "veh_per_h": round(c / duration_h, 1)}
                     for (a, b), c in sorted(cnt.items(), key=lambda kv: -kv[1])]
        out.append({"junction_id": jid, "name": net.junction_names.get(jid, ""),
                    "movements": movements})
    out.sort(key=lambda d: d["junction_id"])
    return out


def _circulating_direction(sel: np.ndarray, ids: np.ndarray, frames: np.ndarray,
                           phi: np.ndarray) -> int:
    """+1 if ring traffic circulates CCW (net increasing angle), else -1.

    Uses the summed per-step angular sweep of ring vehicles — robust and consistent with
    the crossing test, unlike an instantaneous velocity sign.
    """
    sid, sfr, sphi = ids[sel], frames[sel], phi[sel]
    order = np.lexsort((sfr, sid))
    sid, sfr, sphi = sid[order], sfr[order], sphi[order]
    step_ok = (sid[1:] == sid[:-1]) & (sfr[1:] - sfr[:-1] == 1)
    dphi = _wrap(sphi[1:] - sphi[:-1])[step_ok]
    return 1 if dphi.size and float(dphi.sum()) >= 0 else -1


def _arm_direction(sel: np.ndarray, xs: np.ndarray, ys: np.ndarray, vx: np.ndarray,
                   vy: np.ndarray, centre: Tuple[float, float]) -> str:
    """'entry' if vehicles on the arm move toward the ring centre, else 'exit'."""
    if not sel.any():
        return "unknown"
    phi = np.arctan2(ys[sel] - centre[1], xs[sel] - centre[0])
    radial_v = np.cos(phi) * vx[sel] + np.sin(phi) * vy[sel]
    return "entry" if float(np.nansum(radial_v)) < 0 else "exit"


def _circulating_flow(sel: np.ndarray, ids: np.ndarray, frames: np.ndarray, phi: np.ndarray,
                      thetas: np.ndarray, direction: int, duration_h: float) -> np.ndarray:
    """Veh/h crossing each ring angle `theta` in the circulating direction (cross-section).

    Counts, per ring vehicle, every consecutive-frame step whose angular sweep passes a
    theta in the circulating sense — a true cross-section count localized at each entry.
    """
    sid, sfr, sphi = ids[sel], frames[sel], phi[sel]
    order = np.lexsort((sfr, sid))
    sid, sfr, sphi = sid[order], sfr[order], sphi[order]
    step_ok = (sid[1:] == sid[:-1]) & (sfr[1:] - sfr[:-1] == 1)
    dphi = _wrap(sphi[1:] - sphi[:-1])
    valid = step_ok & (np.sign(dphi) == direction)
    if not valid.any():
        return np.zeros(len(thetas))
    rel = _wrap(thetas[None, :] - sphi[:-1][valid][:, None])      # arc to each theta
    dp = dphi[valid][:, None]
    hit = (rel > 0) & (rel <= dp) if direction > 0 else (rel < 0) & (rel >= dp)
    return hit.sum(axis=0) / duration_h if duration_h else np.zeros(len(thetas))


def _approach_angle(lanes: List[Lane], centre: Tuple[float, float]) -> float:
    """Angle (rad) of an approach's ring-end (its lowest-radius centreline point)."""
    pts = np.vstack([ln.centerline for ln in lanes])
    r = np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1])
    p = pts[int(np.argmin(r))]
    return float(np.arctan2(p[1] - centre[1], p[0] - centre[0]))


def compute_roundabout(net: RoadNetwork, ids: np.ndarray, frames: np.ndarray,
                       xs: np.ndarray, ys: np.ndarray, vx: np.ndarray, vy: np.ndarray,
                       speed: np.ndarray, footprint: np.ndarray, role: np.ndarray,
                       road: np.ndarray, n_frames: int, hz: float, los_table: List) -> Dict:
    """Roundabout analysis: ring Edie + delay/LOS, per-approach flow, per-entry circulating flow.

    `role` is a per-sample label ('ring'/'approach'/'normal'/'off'); `road` the road id.
    """
    if not net.has_roundabout:
        return {"detected": False}
    duration_h = n_frames / hz / 3600.0
    cx, cy = net.centre
    phi = np.arctan2(ys - cy, xs - cx)
    ring_sel = role == "ring"
    direction = _circulating_direction(ring_sel, ids, frames, phi)

    ring_km = sum(ln.length_m for ln in net.lanes if ln.role == "ring") / 1000.0
    ring_area = unary_union([ln.polygon for ln in net.lanes if ln.role == "ring"]).area
    occ_pf = np.bincount(frames[ring_sel], weights=footprint[ring_sel], minlength=n_frames) / ring_area
    ring_delay = _control_delay(ring_sel, ids, speed, hz)
    ring = {"lane_km": round(ring_km, 3),
            "edie": _edie(int(ring_sel.sum()), float(speed[ring_sel].sum()), n_frames, ring_km),
            "area_occupancy_fraction": _summary(occ_pf),
            "control_delay_s": ring_delay,
            "level_of_service": _los_grade(ring_delay, los_table)}

    approach_roads = sorted({ln.road_id for ln in net.lanes if ln.role == "approach"})
    thetas = np.array([_approach_angle([ln for ln in net.lanes if ln.road_id == r], net.centre)
                       for r in approach_roads])
    circ = _circulating_flow(ring_sel, ids, frames, phi, thetas, direction, duration_h)
    approaches = []
    for r, theta, cf in zip(approach_roads, thetas, circ):
        a_sel = (role == "approach") & (road == r)
        km = sum(ln.length_m for ln in net.lanes if ln.road_id == r) / 1000.0
        delay = _control_delay(a_sel, ids, speed, hz)
        approaches.append({
            "road": r, "lane_km": round(km, 3),
            "direction": _arm_direction(a_sel, xs, ys, vx, vy, net.centre),
            "edie": _edie(int(a_sel.sum()), float(speed[a_sel].sum()), n_frames, km),
            "circulating_flow_veh_per_h": round(float(cf), 1),
            "control_delay_s": delay, "level_of_service": _los_grade(delay, los_table)})
    return {"detected": True,
            "circulating_direction": "CCW" if direction > 0 else "CW",
            "ring": ring, "approaches": approaches}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _resolve_map(omegaprime: Optional[str], opendrive: Optional[str]) -> Tuple[RoadNetwork, str]:
    """Load the road network from an explicit .xodr, else from the MCAP's embedded map."""
    if opendrive:
        return parse_opendrive(opendrive), opendrive
    if omegaprime:
        xml = load_opendrive_from_mcap(omegaprime)
        if xml:
            return parse_opendrive_text(xml), f"{omegaprime} (embedded)"
        raise ValueError("no --opendrive given and the MCAP has no embedded ground_truth_map")
    raise ValueError("--opendrive is required for an OpenLABEL source")


def _sample_role_road(net: RoadNetwork, lane_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-sample lane role ('off' when unassigned) and road id ('' when unassigned)."""
    lane_role = np.array([ln.role for ln in net.lanes] + ["off"])
    lane_road = np.array([ln.road_id for ln in net.lanes] + [""])
    idx = np.where(lane_idx >= 0, lane_idx, len(net.lanes))
    return lane_role[idx], lane_road[idx]


def compute_traffic_density(omegaprime: Optional[str], openlabel: Optional[str],
                            opendrive: Optional[str], expected_hz: Optional[float],
                            los_table: List, want_series: bool, clean_cfg: Dict) -> Dict:
    """Run the full pipeline and return the metrics dict."""
    net, map_source = _resolve_map(omegaprime, opendrive)
    df, hz = load_trajectories(omegaprime, openlabel, expected_hz)
    keep0, clean_stats = clean_trajectories(df, hz, **clean_cfg)

    ids = df["id"].to_numpy()
    xs = df["x"].to_numpy() - net.e0
    ys = df["y"].to_numpy() - net.n0
    vx, vy = df["vx"].to_numpy(), df["vy"].to_numpy()
    vclass = df["vclass"].to_numpy().astype(str)
    frames = df["frame"].to_numpy()
    n_frames = int(frames.max()) + 1 if len(frames) else 0
    footprint = df["length"].to_numpy() * df["width"].to_numpy()
    speed = np.nan_to_num(np.hypot(vx, vy))
    lane_idx = assign_lanes(net, xs, ys)
    parking = in_parking(net, xs, ys)                 # parked / manoeuvring — not road traffic
    keep = keep0 & ~parking
    data_quality = compute_data_quality(clean_stats, keep, lane_idx, vclass, ids, parking)
    data_quality["thresholds"] = clean_cfg

    k = keep                                          # traffic metrics use kept-moving samples
    role, road = _sample_role_road(net, lane_idx[k])
    on_mask = lane_idx[k] >= 0
    counts_pf = np.bincount(frames[k][on_mask], minlength=n_frames).astype(float)
    fp_pf = np.bincount(frames[k][on_mask], weights=footprint[k][on_mask], minlength=n_frames)
    total_lane_km = sum(ln.length_m for ln in net.lanes) / 1000.0
    duration_h = n_frames / hz / 3600.0 if hz == hz and hz else 0.0

    return {
        "provenance": {
            "tool": "traffic_density", "version": _VERSION,
            "source": omegaprime or openlabel,
            "source_format": "OmegaPrime MCAP" if omegaprime else "OpenLABEL JSON",
            "opendrive": map_source, "map_crs": net.crs,
            "frames": n_frames, "sampling_hz": round(hz, 4) if hz == hz else None,
            "lanes": len(net.lanes), "roundabout_detected": net.has_roundabout,
        },
        "data_quality": data_quality,
        "geometry_agnostic": compute_geometry_agnostic(fp_pf, counts_pf, net.drivable.area, want_series),
        "network": compute_network(on_mask, ids[k], speed[k], counts_pf, n_frames,
                                   total_lane_km, want_series),
        "by_type": compute_by_type(vclass[k], on_mask, ids[k], speed[k], n_frames, total_lane_km),
        "roads": compute_roads(net, road, speed[k], n_frames),
        "junctions": compute_turning_movements(net, ids[k], frames[k], road, duration_h),
        "roundabout": compute_roundabout(net, ids[k], frames[k], xs[k], ys[k], vx[k], vy[k],
                                         speed[k], footprint[k], role, road, n_frames, hz, los_table),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Traffic density (Edie), flow and roundabout LOS from trajectories + an OpenDRIVE map.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--omegaprime", metavar="FILE", help="OmegaPrime MCAP trajectory source")
    src.add_argument("--openlabel", metavar="FILE", help="OpenLABEL JSON trajectory source (world cuboids)")
    parser.add_argument("--opendrive", metavar="FILE",
                        help="OpenDRIVE .xodr map. Optional with --omegaprime: if omitted, "
                             "the map embedded in the MCAP (ground_truth_map) is used.")
    parser.add_argument("--expected-hz", type=float, default=None,
                        help="Frame rate; used for OpenLABEL velocity finite-difference (default 30).")
    parser.add_argument("--los-spec", metavar="FILE",
                        help="JSON list of [grade, upper_control_delay_seconds] overriding the "
                             "HCM delay-based LOS defaults.")
    parser.add_argument("--min-track-s", type=float, default=_MIN_TRACK_S,
                        help=f"Drop tracks shorter than this (s); default {_MIN_TRACK_S}.")
    parser.add_argument("--static-disp-m", type=float, default=_STATIC_DISP_M,
                        help="Exclude vehicles whose centre moves less than this over their "
                             f"life, i.e. parked (m); default {_STATIC_DISP_M}.")
    parser.add_argument("--max-speed-ms", type=float, default=_MAX_SPEED_MS,
                        help=f"Drop samples faster than this as teleports (m/s); default {_MAX_SPEED_MS}.")
    parser.add_argument("--time-series", action="store_true",
                        help="Include per-frame density/occupancy arrays in the output.")
    parser.add_argument("--json", dest="json_out", metavar="FILE", help="Write result JSON to FILE.")
    args = parser.parse_args()

    los_table = _DELAY_LOS_TABLE
    if args.los_spec:
        with open(args.los_spec) as f:
            los_table = json.load(f)
    clean_cfg = {"min_track_s": args.min_track_s, "static_disp_m": args.static_disp_m,
                 "max_speed_ms": args.max_speed_ms}

    try:
        result = compute_traffic_density(args.omegaprime, args.openlabel, args.opendrive,
                                         args.expected_hz, los_table, args.time_series, clean_cfg)
    except Exception as exc:                    # surface failure without a traceback dump
        result = {"error": f"{type(exc).__name__}: {exc}"}

    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text)


if __name__ == "__main__":
    main()
