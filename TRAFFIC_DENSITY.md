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
| **OpenDRIVE map** (`.xodr`) | Road/lane geometry, lane lengths, drivable surface, junctions. | Supplies the spatial reference. Can be passed explicitly, or — for an MCAP source — taken from the map **embedded in the MCAP** (`ground_truth_map`), so `--opendrive` is then optional. |
| **OmegaPrime MCAP** | Per-frame vehicle trajectories with **velocity**, dimensions, type. | One of the two trajectory sources. |
| **OpenLABEL JSON** | Per-frame vehicle **world cuboids** (metres). | Alternative source; velocity is derived by finite difference. May be plain or **gzip/zip/bz2/xz** compressed (auto-detected from magic bytes — these files are large). |

Exactly one trajectory source is given (`--omegaprime` *or* `--openlabel`). The map comes from
`--opendrive`, or — for an MCAP source — from the OpenDRIVE embedded in the MCAP, so
`--opendrive` is optional there (see §3); an OpenLABEL source always needs it. Both trajectory
sources describe vehicles in the **same global UTM frame**, which is what lets either be
cross-checked against the other.

---

## 2. Pipeline overview

```
            ┌────────────────┐     ┌──────────────────────┐
 .xodr ───► │ parse_opendrive│     │  load_trajectories   │ ◄─── MCAP / OpenLABEL
            │  lanes+geometry│     │  per-frame veh table │
            └───────┬────────┘     └──────────┬───────────┘
                    │  lane polygons,          │  x,y,yaw,L,W,vx,vy,vclass
                    │  drivable area, ring      │
                    ▼                           ▼
              ┌──────────────────────────────────────┐
              │  clean_trajectories (short/parked/    │
              │     teleport → keep mask)             │
              │  align coords (global UTM → map frame)│
              │  assign_lanes (point-in-lane polygon) │
              └──────────────────┬─────────────────────┘
                                 ▼
   Edie (space-time) k/q/v per region  +  per-frame occupancy/density  +  flow & delay
                                 ▼
  data_quality │ geometry_agnostic │ network │ by_type │ roads │ junctions │ roundabout
```

---

## 3. Parsing the OpenDRIVE map (`parse_opendrive`)

The map is resolved by `_resolve_map`: an explicit `--opendrive` file is parsed with
`parse_opendrive`; otherwise, for an MCAP source, the OpenDRIVE XML embedded on the
`ground_truth_map` topic (`osi3.MapAsamOpenDrive.open_drive_xml_content`) is extracted by
`load_opendrive_from_mcap` and parsed in-memory with `parse_opendrive_text`. Both paths feed
the same `_network_from_root` builder described below. An OpenLABEL source carries no map, so
it always requires `--opendrive`.

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
The scene **centre** is the centroid of all lane-centreline points. Each road is then scored by
its **circumferential fraction** (`_circumferential_fraction`): the length-weighted
`1 − |unit_tangent · unit_radial|` over its centreline segments, where 1.0 is a constant-radius
arc (runs *around* the centre) and 0.0 is a pure radial spoke (runs *toward* the centre).

- `circ ≥ _RING_CIRC_MIN` (0.75) → **ring** — the circulatory roadway. This deliberately spans
  both plain ring roads **and** the junction connectors that carry the ring *through* an
  entry/exit, because in OpenDRIVE a large part of the circulatory loop is modelled as
  junction-internal connecting roads. In the reference scene these score 0.84–0.96.
- otherwise, a **non-junction** road → **approach** (a radial arm; circ ≈ 0.01–0.06).
- otherwise (a junction connector that is only partly circumferential, circ ≈ 0.51–0.69) →
  `normal` — the entry/exit **merge–diverge slip**.

There is a clean gap (≈0.69 vs ≈0.84) between the slips and the true circulatory lanes, so the
0.75 threshold is robust.

**Loop check (avoids false positives).** Circumferential candidates alone are not enough — a
short curved connector on a straight-road scene can score ≥ 0.75. So after selecting candidate
ring lanes, the centre is **refined** to their centroid, roles are re-scored, and the ring is
accepted only if it forms a near-complete loop: ring-lane points are binned into 36 ten-degree
sectors about the centre and **≥ `_RING_SECTOR_COVERAGE` (75 %)** of sectors must be occupied
(`_ring_sector_coverage`). A real roundabout covers ~all sectors; a 9°-arc connector covers one
→ rejected, ring lanes demoted, `roundabout.detected = false`.

### 3.5 Parking areas (`_parking_areas`)
Driving lanes are not the whole story: a map can also define **parking lots** as
`<object type="parking">` with an `<outline>` of `cornerLocal` (u, v) corners. Each object is
placed at its road arc-length `s`, lateral offset `t` and heading `hdg` (`_road_point_at_s` +
the road heading there) to build a world-frame polygon. These are stored alongside the driving
lanes so vehicles inside them can be recognized as **parking, not road traffic** — without this,
parked cars and cars manoeuvring in/out of a lot show up as "off-lane" anomalies even though the
map clearly marks the area as parking.

---

## 4. Loading trajectories

A unified per-frame table is built with columns
`frame, id, x, y, yaw, length, width, vx, vy, vclass`.

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

`vclass` is the coarse vehicle group (car/van/truck/bus/motorcycle/bicycle/other), mapped from
the OSI `vehicle_classification.type` (MCAP) or the OpenLABEL object `type` string.

---

## 4b. Trajectory cleaning (`clean_trajectories`)

Pseudo-labelled (YOLO) clips are noisy and every scene has parked cars; the MCAP carries **no
confidence field**, so cleaning is based on track duration, lifetime motion, and per-sample
speed (uniform across sources, thresholds CLI-configurable). It returns a sample-level `keep`
mask used by all traffic metrics, plus the stats surfaced in `data_quality`:

- **short track** — `duration < --min-track-s` (0.5 s) → dropped (flicker false positives).
- **static / parked** — lifetime centre displacement `< --static-disp-m` (2.0 m), and not short
  → **excluded from traffic metrics** (a parked car is not traffic and otherwise dominates
  counts — in one example scene 7 parked cars were 76 % of all samples). A **queued** vehicle
  moves eventually, so its displacement exceeds the threshold and it is kept.
- **teleport sample** — `speed > --max-speed-ms` (55 m/s) → that sample is dropped (ID-switch /
  tracking jumps; pseudo-labelled clips show velocities into the hundreds of m/s).
- **in a parking area** (§3.5) — excluded from traffic metrics as parking, **per sample** (not
  per track): a car parked for most of a clip then driving off contributes only its on-road
  samples. In one example scene this moved a misleading 35 % "off-lane" down to 0.3 % — the rest
  was a single parked car manoeuvring in a lot.

---

## 5. Lane assignment (`assign_lanes`)

Assignment is **per sample** (per frame), not per vehicle. For each vehicle position the
`STRtree` returns candidate lanes by bounding box, then an explicit `polygon.contains(point)`
confirms membership (this is robust to shapely version differences in `query` predicate
semantics). The first containing lane wins; samples in no driving lane get index `−1` and are
**excluded** from the on-road metrics *for those frames only* — the same vehicle's on-lane
frames still count, so a vehicle is never wholly removed by clipping a lane edge. The off-lane
fraction (of kept-moving samples) is reported in `data_quality`.

Because driving-lane polygons are built at **constant width** (§3.3) and vehicles ride lane
edges, a fraction of samples sit just outside a lane (on the Saro roundabout ~12 %, but 94 %
of them within 1 m of a lane). Those frames do not contribute to the Edie sums, so on-road
density/flow carry a small low bias (~the off-lane fraction). This is a known limitation with
fix options (snap-to-nearest-lane; non-constant width) tracked in
[DEVELOPMENT.md](DEVELOPMENT.md).

---

## 6. Metrics — calculation and interpretation

Sections: **data_quality** (reliability), **geometry_agnostic** (need only the drivable
polygon), **network** (Edie state), **by_type**, **roads** (per-segment), **junctions**
(turning movements), and **roundabout**. All traffic metrics use the **kept-moving** samples
from §4b; `data_quality` describes the whole input. "On-road" means *assigned to a driving
lane*. Instantaneous per-frame quantities are summarized `{mean, peak, p95, min}` (`_summary`);
density/flow/speed use Edie's definitions (§6.1).

### 6.0 `data_quality` — how reliable the clip is (`compute_data_quality`)
`n_objects`, `n_static` (fully-parked, excluded), `n_short_tracks` and `n_teleport_samples`
(removed), `parking_vehicles` / `parking_samples` (in a mapped parking area, excluded),
`samples_kept_fraction`, `off_lane_fraction` (kept-moving samples off the **driving** lanes —
parking already excluded, so this flags genuine lane-coverage gaps), `unknown_type_objects`
(OSI "other"/"unknown" — often spurious), `track_duration_s` `{median, p10}`, the applied
`thresholds`, and a `removed` summary. High static/short/teleport counts or a low kept-fraction
flag a noisy (e.g. pseudo-labelled) clip whose traffic numbers should be read with caution.

### 6.1 Why Edie's generalized definitions (`_edie`)
Counting vehicles in a segment at one instant (`count/length`) is noisy on short segments and
gives no flow. Edie (1963) instead integrates over a space–time region A (a set of lanes of
total length `L` observed for a duration `T`). With fixed-`dt` trajectory samples, each
in-region vehicle-frame contributes `dt` of time and `v·dt` of distance, so:

```
k = total_time / (L·T)     = mean_count / L          [veh/km]   (per lane-km)
v = total_distance / total_time = mean sample speed  [km/h]
q = total_distance / (L·T) = k · v                   [veh/h]    (per lane)
```

This is the standard way to derive macroscopic density/flow/speed from trajectory/drone data
(NGSIM, highD, **rounD**), and `q = k·v` holds exactly. A useful identity: Edie's `q` over a
one-way arm equals the **cross-section flow** `N/T` (vehicles that traversed it ÷ duration),
so the approach Edie flow *is* the entry/exit flow — no fragile counting line needed. All
densities/flows are **per lane** (normalized by total lane-length), matching the HCM
veh/km/lane convention.

### 6.2 `geometry_agnostic` — topology-free indicators
```
area_occupancy(frame) = Σ(Lᵢ·Wᵢ for on-road vehicles) / drivable_area_m²   [fraction 0..1]
veh_per_m2(frame)     = on_road_vehicle_count(frame) / drivable_area_m²     [veh/m²]
```
These need no centreline, lane count, or length — only the drivable polygon — so they
generalize to **any topology** and are the **best cross-scene comparators**. The occupancy
fraction is the share of tarmac physically covered by vehicle footprints.

**Interpretation:** ≈0 is empty; ≈0.10–0.15 is dense; the jam ceiling is well below 1.0
because of inter-vehicle gaps. *Reference scene:* mean ≈ **0.029** (3 % covered) — sparse.
*Caveat:* occupancy is a dimensionless proxy, not directly comparable to literature veh/km,
and is sensitive to drone altitude/footprint scaling across datasets.

### 6.3 `network` — whole-network traffic state
- `edie` = Edie `{density_veh_per_km, flow_veh_per_h, speed_kmh}` over all driving lanes.
- `instantaneous_density_veh_per_lane_km` = `{summary}` of the per-frame `count/total_lane_km`
  (its mean equals the Edie density — a built-in consistency check).
- `vehicles_on_road` = `{summary}` of the raw per-frame count.

**Interpretation (per-lane-km reference points):** <11 free flow; 11–28 stable→near-capacity;
~25–40 capacity; >28 congested; 120–180 jam. *Reference scene:* Edie density ≈ **7.3
veh/km/lane**, flow ≈ **237 veh/h/lane**, speed ≈ **33 km/h** — light, free-flowing.

### 6.3a `by_type` — per vehicle class (`compute_by_type`)
For each coarse class present on-road (car/van/truck/bus/motorcycle/bicycle/other): the object
`count` and an Edie `{density, flow, speed}` over the network. Heavy vehicles contribute
disproportionately to occupancy, and the class mix characterizes the scene.

### 6.3b `roads` — per-segment density (`compute_roads`)
Edie `{density, flow, speed}` for **each driving road** (grouped by OpenDRIVE `road_id`),
sorted by flow. This is the generic per-segment metric for any scene — for a straight-road or
crossing clip it gives the through-road's density/flow directly (the roundabout's ring and
approaches are themselves roads, so they also appear here).

### 6.3c `junctions` — turning movements (`compute_turning_movements`)
For each OpenDRIVE junction, the count and veh/h of each `from_road → to_road` movement. Each
kept-moving vehicle's ordered road sequence is collapsed (off-road gaps dropped); whenever it
crosses a junction connector road, the roads immediately before and after are the movement.
Through-movements that never enter a connector are not separately counted; most useful on
junctions with real turning traffic.

### 6.4 `roundabout` — the operational analysis
Emitted only when a ring was detected (§3.4); otherwise `{"detected": false}`. This follows
how roundabouts are actually assessed (HCM-7 / TRL): **flow and delay**, not freeway density.

`circulating_direction` (`CCW`/`CW`) is the sign of ring vehicles' net angular sweep
(`_circulating_direction`) — robust and identical across sources.

**`ring`** — the circulatory roadway (plain ring roads **plus** the through-going junction
connectors, §3.4; the entry/exit slips are excluded):
- `edie` density/flow/speed (the **flow is the mean circulating flow per lane**),
- `area_occupancy_fraction` = ring footprint / ring polygon area,
- `control_delay_s` and delay-based `level_of_service` (§6.5).

**`approaches`** — one per radial arm, the classic per-approach table:
- `direction` — `entry`/`exit` from the sign of mean radial velocity (`_arm_direction`).
- `edie` — the arm's density/flow/speed; its **flow = the entry/exit cross-section flow**.
- `circulating_flow_veh_per_h` — **total** flow crossing the ring at this arm's angular
  position (`_circulating_flow`, an angular cross-section count over both ring lanes). This is
  the **conflicting flow** that governs entry capacity in roundabout theory — the single most
  important quantity for sizing an entry.
- `control_delay_s` and `level_of_service`.

*Reference scene:* 3 entries (roads 45/46/52) + 3 exits (15/50/51); ring mean circulating flow
≈ **160 veh/h/lane**, conflicting flows ≈ **150–300 veh/h** per arm, all delays < 2 s → **LOS A**
(under-saturated, free-flowing).

### 6.5 Level of Service — delay-based (`_control_delay`, `_los_grade`)
Roundabout/unsignalized LOS is graded by **control delay**, not density. Per region the tool
measures delay directly from trajectories: with a free-flow speed `v_ff` = the region's
85th-percentile speed (`_FREE_FLOW_PERCENTILE`), each in-region sample loses
`dt·(1 − v/v_ff)` seconds vs free flow; per-vehicle losses are summed, clamped at 0, and
averaged. Grades (HCM, s/veh): `A ≤10, B ≤15, C ≤25, D ≤35, E ≤50, F >50`.

> This is measured (uncongested-reference) delay, not a capacity-model control delay; it
> captures slow-down through the junction. Override the thresholds with `--los-spec`
> (`[[grade, upper_delay_s], …]`).

---

## 7. Output structure

```jsonc
{
  "provenance":       { tool, version, source, source_format, opendrive, map_crs,
                        frames, sampling_hz, lanes, roundabout_detected },
  "data_quality":     { n_objects, n_static, n_short_tracks, n_teleport_samples,
                        parking_vehicles, parking_samples, samples_kept_fraction,
                        off_lane_fraction, unknown_type_objects,
                        track_duration_s{median,p10}, thresholds{…}, removed{…} },
  "geometry_agnostic":{ drivable_area_m2, area_occupancy_fraction{…}, veh_per_m2{…},
                        [time_series_occupancy] },
  "network":          { total_lane_km, edie{density_veh_per_km, flow_veh_per_h, speed_kmh},
                        instantaneous_density_veh_per_lane_km{…}, vehicles_on_road{…},
                        [time_series_density] },
  "by_type":          { car:{count, edie{…}}, truck:{…}, bus:{…}, … },
  "roads":            [ { road, lane_km, edie{…} } … ],          // sorted by flow
  "junctions":        [ { junction_id, name, movements:[{from,to,count,veh_per_h}] } … ],
  "roundabout":       { detected, circulating_direction,
                        ring: { lane_km, edie{…}, area_occupancy_fraction{…},
                                control_delay_s, level_of_service },
                        approaches: [ { road, lane_km, direction, edie{…},
                                        circulating_flow_veh_per_h, control_delay_s,
                                        level_of_service } … ] }
}
```
Each `{…}` is a `{mean, peak, p95, min}` summary; each `edie{…}` is a density/flow/speed
triple. A run that fails to compute returns `{"error": "<type>: <message>"}` instead of
aborting.

---

## 8. Cross-source validation

Running both trajectory sources for the same scene against the same map yields closely matching
results — independent confirmation that geometry, alignment, lane assignment, the
finite-difference velocity, and the flow/direction logic are correct:

| Metric | OmegaPrime MCAP | OpenLABEL JSON |
|---|---|---|
| network Edie density (veh/km/lane) | 7.29 | 7.19 |
| network Edie speed (km/h) | 32.6 | 32.6 |
| ring Edie flow / circulating (veh/h/lane) | 159.5 | 157.4 |
| circulating direction | CCW | CCW |
| approach 15 cross-section flow (veh/h) | 476 | 470 |
| approach 15 conflicting flow (veh/h) | 176 | 174 |

---

## 9. Key parameters & assumptions

| Constant | Value | Meaning |
|---|---|---|
| `_LANE_SAMPLE_STEP_M` | 1.0 m | Reference-line sampling resolution. |
| `_RING_CIRC_MIN` | 0.75 | Min circumferential fraction for a lane to count as ring. |
| `_RING_SECTOR_COVERAGE` | 0.75 | Fraction of 10° sectors a real ring loop must cover (false-positive guard). |
| `_MIN_TRACK_S` | 0.5 s | Tracks shorter than this are dropped (`--min-track-s`). |
| `_STATIC_DISP_M` | 2.0 m | Lifetime displacement below this ⇒ parked, excluded (`--static-disp-m`). |
| `_MAX_SPEED_MS` | 55 m/s | Samples faster than this are teleports, dropped (`--max-speed-ms`). |
| `_FREE_FLOW_PERCENTILE` | 85 | Region speed percentile used as the free-flow reference for delay. |
| `_DELAY_LOS_TABLE` | HCM delay LOS | Control-delay→LOS breakpoints (override with `--los-spec`). |

**Assumptions / limitations** (deferred fixes tracked in [DEVELOPMENT.md](DEVELOPMENT.md))
- Trajectory and map CRS share one UTM zone (no reprojection between zones).
- Lane widths are **constant** (only the `<width>` `a` term); only `line` and `paramPoly3`
  geometry are evaluated. With edge-riding this leaves a small off-lane fraction, so on-road
  density/flow are biased slightly low (§5).
- Only `type="driving"` lanes count as road; other lane types are ignored. Parking is handled
  separately via `<object type="parking">` outlines.
- Delay is a measured slow-down vs an 85th-percentile free-flow speed, not a capacity-model
  control delay; under light traffic it reads near zero (LOS A).
- Approaches are classified geometrically; an arm carries either entering or exiting traffic
  (labelled by radial-velocity sign), not both.
- Turning movements count only connector-crossing turns, not straight-through movements.
