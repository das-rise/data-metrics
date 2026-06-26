# Traffic Density Metrics — Design & Reference

`traffic_density.py` computes real-world traffic-flow metrics for a scene from vehicle
trajectories grounded on an OpenDRIVE road map. Unlike a pixel-area proxy, every number is
in physical units (veh/km, veh/m², km/h, veh/h), so values are comparable across scenes and
interpretable against standard traffic-engineering reference points.

This document explains **how the tool works** (the pipeline and the geometry) and **how each
metric is calculated and interpreted**. For the CLI option table see the
`traffic_density.py` section of [README.md](README.md).

---

## 1. Inputs

| Input | Role | Notes |
|---|---|---|
| **OpenDRIVE map** (`.xodr`) | Road/lane geometry, lane lengths, drivable surface, junctions. | Required. Always supplies the spatial reference. |
| **OmegaPrime MCAP** | Per-frame vehicle trajectories with **velocity**, dimensions, type. | One of the two trajectory sources. |
| **OpenLABEL JSON** | Per-frame vehicle **world cuboids** (metres). | Alternative source; velocity is derived by finite difference. May be plain or **gzip/zip/bz2/xz** compressed (auto-detected from magic bytes — these files are large). |

Exactly one trajectory source is given (`--omegaprime` *or* `--openlabel`); the map is always
given (`--opendrive`). Both trajectory sources describe vehicles in the **same global UTM
frame**, which is what lets either be cross-checked against the other.

---

## 2. Pipeline overview

```
            ┌────────────────┐     ┌──────────────────────┐
 .xodr ───► │ parse_opendrive│     │  load_trajectories   │ ◄─── MCAP / OpenLABEL
            │  lanes+geometry│     │  per-frame veh table │
            └───────┬────────┘     └──────────┬───────────┘
                    │  lane polygons,          │  x,y,yaw,L,W,vx,vy
                    │  drivable area, ring      │
                    ▼                           ▼
              ┌──────────────────────────────────────┐
              │  align coords  (global UTM → map frame)│
              │  assign_lanes  (point-in-lane polygon) │
              └──────────────────┬─────────────────────┘
                                 ▼
        per-frame series → whole_scene │ per_lane │ area_occupancy │ speed_flow │ roundabout
                                 ▼
                       mean / peak / p95 / min summaries  (+ optional time series)
```

---

## 3. Parsing the OpenDRIVE map (`parse_opendrive`)

### 3.1 Coordinate alignment (`_geo_offset`)
Trajectories are in **global UTM** (e.g. `x≈320710, y≈6375215`). The map's `<geoReference>`
is the same UTM projection but shifted to a local origin via `+lat_0/+lon_0`, so map
coordinates are small numbers centred on zero. The tool computes the origin offset

```
(E0, N0) = proj_UTM(lon_0, lat_0)
map_local = global_UTM − (E0, N0)
```

and subtracts `(E0, N0)` from every vehicle position before lane assignment. This assumes the
trajectory CRS and the map CRS are the **same UTM zone** (true for the supported data); the
tool does not reproject between different zones.

### 3.2 Reference-line geometry (`_sample_geometry`)
Each road's `planView` is a sequence of `<geometry>` primitives. Two primitive types occur in
the supported maps and both are evaluated analytically, sampled every `_LANE_SAMPLE_STEP_M`
(1.0 m):

- **`line`** — straight: `(x,y) = (x0 + s·cosθ, y0 + s·sinθ)`, heading `θ = hdg`.
- **`paramPoly3`** — cubic parametric curve in local `(u,v)` over a normalized parameter
  `p ∈ [0,1]`:
  ```
  u(p) = aU + bU·p + cU·p² + dU·p³        v(p) = aV + bV·p + cV·p² + dV·p³
  ```
  Each `(u,v)` is rotated by `hdg` and translated to `(x0,y0)`. The local tangent
  `atan2(v', u')` is added to `hdg` to give the heading used for the lane normal.

(No `spiral`/clothoid primitives are present, so no Fresnel-integral handling is needed.)

### 3.3 Lane polygons (`_build_driving_lanes`)
Lane widths are **constant** (the `a` coefficient of `<width>`; higher-order terms are absent).
For each reference-line sample the left-hand normal is `n = (−sinθ, cosθ)`. Walking outward
from the reference line, lane *k* occupies the band between cumulative offsets `inner` and
`outer = inner + width`; right lanes (negative id) use `−n`. The lane **centreline** is the
band midline; the lane **polygon** is `outer_edge ⧺ reversed(inner_edge)`, cleaned with
`buffer(0)` to repair the self-intersections that constant-width offsetting produces on tight
curves. Only lanes with `type="driving"` are kept.

The lane's **length** is the polyline length of its centreline. The scene **drivable area** is
the shapely union of all driving-lane polygons, and all polygons are indexed in an `STRtree`.

### 3.4 Roundabout detection (`_classify_roundabout`)
The scene **centre** is the centroid of all lane-centreline points. For every **non-junction**
road, the radial distances of its centreline to the centre are computed and reduced to a
coefficient of variation `CoV = std/mean`:

- `CoV < _RING_RADIAL_COV_MAX` (0.15) → **ring** (a circulatory arc stays at near-constant
  radius). In the reference scene the ring arcs have `CoV ≤ 0.05`.
- otherwise → **approach** (a radial arm; its radius spans a wide range, `CoV ≥ 0.25`).

Junction connector roads are left as `normal`. A roundabout is "detected" iff at least one ring
lane is found, so the `roundabout` section degrades cleanly to `{"detected": false}` on
straight-road scenes.

---

## 4. Loading trajectories

A unified per-frame table is built with columns
`frame, id, x, y, yaw, length, width, vx, vy`.

- **OmegaPrime MCAP** (`load_omegaprime`) — iterates `ground_truth` messages, expands each
  `moving_object` into a row, and reads velocity directly. Frame indices and the sampling rate
  come from `_frames_from_time` (dense-ranked unique timestamps; `hz = 1/median(Δt)`). The
  loader is inlined rather than imported from `data_metrics.py` to avoid pulling in OpenCV/PyAV.
- **OpenLABEL JSON** (`load_openlabel`) — reads each frame object's `world` cuboid
  `val = [x, y, z, roll, pitch, yaw, L, W, H]`. Velocity is **finite-differenced** per track:
  `vx = Δx · hz`, `vy = Δy · hz` (using `--expected-hz`, default 30). The close agreement
  between finite-differenced and MCAP-recorded speeds (≈0.4 km/h) validates this. The file is
  opened through `_read_json_any`, which detects gzip/zip/bz2/xz from the leading magic bytes
  and decompresses on the fly (a zip is read from its first `.json` member), so a compressed
  export works under any filename.

---

## 5. Lane assignment (`assign_lanes`)

For each vehicle position the `STRtree` returns candidate lanes by bounding box, then an
explicit `polygon.contains(point)` confirms membership (this is robust to shapely version
differences in `query` predicate semantics). The first containing lane wins; vehicles in no
driving lane get index `−1` and are **excluded** from all on-road metrics (they are off-road
detections, parked vehicles outside lanes, or tracker noise). In the reference scene ~91 % of
samples fall on a driving lane.

---

## 6. Metrics — calculation and interpretation

All metrics are computed **per frame**, then each per-frame series is reduced to a
`{mean, peak, p95, min}` summary (`_summary`). With `--time-series` the raw per-frame arrays
are also emitted. "On-road" everywhere means *assigned to a driving lane*.

### 6.1 `whole_scene` — network density
```
total_lane_km = Σ lane.length_m / 1000
density(frame) = on_road_vehicle_count(frame) / total_lane_km      [veh per lane-km]
```
This is the headline number: the standard traffic density `k = N/L` aggregated over the whole
mapped network, normalized **per lane-kilometre** (the HCM convention). `mean_level_of_service`
grades the mean density (§6.6). `vehicles_on_road` reports the raw count for context.

**Interpretation (per-lane-km reference points):**

| Density (veh/km/lane) | Regime |
|---|---|
| < 11 | free flow (LOS A–B) |
| 11–28 | stable to near-capacity (LOS C–E) |
| ~25–40 | capacity / breakdown onset |
| > 28 | congested / forced flow (LOS F) |
| 120–180 | jam, bumper-to-bumper |

*Reference scene:* mean ≈ **7.3 veh/lane-km, LOS B** — light free-flowing roundabout traffic.

### 6.2 `per_lane` — per-lane density + LOS
For each driving lane, `density(frame) = count_in_lane(frame) / (lane.length_m/1000)`. Each
entry reports `mean_density`, `peak_density`, and `peak_level_of_service` (LOS at peak density),
sorted by peak. To stop instantaneous density on isolated few-metre segments from being
dominated by quantisation (one vehicle on 3 m reads as ~300 veh/km), two kinds of short,
multi-segment structures are **merged** (`_lane_merge_key`), each on the same lane-km basis as
the totals they belong to:

- the circulatory arcs → a single **`ring`** road;
- each junction's connector lanes → one **`junction <id>`** entry per junction.

Ordinary non-junction lanes (the approach arms and through lanes) stay individual. A `lane`
field of `null` marks a merged entry.

**Interpretation:** use this to find spatial hotspots and the worst-served structure. Trust
`mean_density` for the typical loading and read `peak_density` as transient clustering;
`length_m` tells you how much road the entry covers. On the longer approach arms `peak` is real
congestion, whereas on the shorter merged junctions a high `peak` can still reflect a brief
cluster of 2–3 vehicles.

### 6.3 `area_occupancy` — geometry-agnostic density
```
area_occupancy(frame) = Σ(Lᵢ·Wᵢ for on-road vehicles) / drivable_area_m²     [fraction 0..1]
veh_per_m2(frame)     = on_road_vehicle_count(frame) / drivable_area_m²       [veh/m²]
```
This needs no centreline or lane-count and so generalizes cleanly to **any topology**
(roundabouts, junctions, merges) — it is the most robust cross-scene comparator. The occupancy
fraction is the share of drivable tarmac physically covered by vehicle footprints.

**Interpretation:** a fraction near **0** is empty road; values around **0.10–0.15** indicate
dense traffic; the theoretical jam ceiling is well below 1.0 because of inter-vehicle gaps.
*Reference scene:* mean ≈ **0.029** (3 % of the tarmac covered) — sparse.

### 6.4 `speed_flow` — speed and flow
```
space_mean_speed(frame) = harmonic_mean( |v| of on-road vehicles with |v| > 0.1 m/s )   [→ km/h]
flow(frame) = whole_scene_density · space_mean_speed                                     [veh/h/lane]
```
**Space-mean speed** (harmonic, not arithmetic) is the theoretically correct average for the
fundamental relation `q = k·v`; slow vehicles weigh more heavily, as they should for a spatial
average. Stationary vehicles are excluded (`_MOVING_SPEED_MIN_MS`) to keep the harmonic mean
finite. **Flow** is the realized throughput implied by that density and speed.

**Interpretation:** speed and flow together place the scene on the fundamental diagram — high
speed + low density = free flow; falling speed with rising density = approaching capacity.
*Reference scene:* mean ≈ **31 km/h**, flow ≈ **210 veh/h/lane** (a high `peak` speed can occur
from velocity outliers — prefer `mean`/`p95`).

### 6.5 `roundabout` — circulatory + approach detail
Emitted only when a ring was detected (§3.4); otherwise `{"detected": false}`.

- **Ring:** `ring_lane_km` = total ring lane length; `ring_vehicle_count` = vehicles on any ring
  lane per frame; `ring_density_veh_per_km = count / ring_lane_km`; `ring_area_occupancy_fraction`
  = ring footprint / ring polygon area. This is the standard way to express roundabout loading
  (vehicles per unit of circulatory roadway).
- **Approaches:** for each approach arm, `density_veh_per_km = count_on_arm / arm_lane_km`,
  summarized over frames. Use this to see which entry is busiest.

*Reference scene:* ring length ≈ **160 m (lane-km basis)**, mean ≈ **6 veh/km**, peak ≈ 31.

### 6.6 Level of Service (`_los_grade`)
Density is graded A–F against a threshold table. The default is HCM6 **basic freeway-segment**
density (veh/km/lane): `A ≤7, B ≤11, C ≤16, D ≤22, E ≤28, F >28`.

> **Caveat:** these thresholds are calibrated for uninterrupted freeway flow. For urban
> arterials and roundabouts (interrupted flow, where LOS is normally delay-based) they are only
> *indicative*. Supply campaign-appropriate breakpoints with `--los-spec` (a JSON list of
> `[grade, upper_density]`).

---

## 7. Output structure

```jsonc
{
  "provenance":   { tool, version, source, source_format, opendrive, map_crs,
                    frames, sampling_hz, lanes, roundabout_detected },
  "whole_scene":  { total_lane_km, density_veh_per_lane_km{…}, mean_level_of_service,
                    vehicles_on_road{…}, [time_series] },
  "per_lane":     { n_lanes, lanes: [ { road, lane, role, length_m,
                    mean_density, peak_density, peak_level_of_service } … ] },
  "area_occupancy":{ drivable_area_m2, area_occupancy_fraction{…}, veh_per_m2{…},
                    [time_series_occupancy] },
  "speed_flow":   { space_mean_speed_kmh{…}, flow_veh_per_h_per_lane{…} },
  "roundabout":   { detected, ring_lane_km, ring_vehicle_count{…},
                    ring_density_veh_per_km{…}, ring_area_occupancy_fraction{…},
                    approaches: [ { road, length_m, density_veh_per_km{…} } … ] }
}
```
Each `{…}` is a `{mean, peak, p95, min}` summary. A run that fails to compute returns
`{"error": "<type>: <message>"}` instead of aborting.

---

## 8. Cross-source validation

Running both trajectory sources for the same scene against the same map yields closely matching
results — independent confirmation that geometry, alignment, lane assignment, and the
finite-difference velocity are correct:

| Metric | OmegaPrime MCAP | OpenLABEL JSON |
|---|---|---|
| whole-scene density (veh/lane-km) | 7.29 | 7.19 |
| space-mean speed (km/h) | 30.9 | 30.5 |
| on-road vehicles (mean) | 5.58 | 5.51 |
| ring vehicles (mean) | 0.96 | 0.94 |

---

## 9. Key parameters & assumptions

| Constant | Value | Meaning |
|---|---|---|
| `_LANE_SAMPLE_STEP_M` | 1.0 m | Reference-line sampling resolution. |
| `_RING_RADIAL_COV_MAX` | 0.15 | Radial CoV below which a non-junction road is ring. |
| `_MOVING_SPEED_MIN_MS` | 0.1 m/s | Stationary cutoff for space-mean speed. |
| `_DEFAULT_LOS_TABLE` | HCM6 freeway | Density→LOS breakpoints (override with `--los-spec`). |

**Assumptions / limitations**
- Trajectory and map CRS share one UTM zone (no reprojection between zones).
- Lane widths are constant; only `line` and `paramPoly3` geometry are evaluated.
- Only `type="driving"` lanes count as road; other lane types are ignored.
- Density on very short lanes is quantisation-limited — interpret with `length_m`.
- LOS thresholds are freeway-derived; treat as indicative for interrupted flow.
