# Development notes — deferred work

Known limitations and design options for the tools, captured for later. Nothing here is
implemented yet. The focus is `traffic_density.py` (see [TRAFFIC_DENSITY.md](TRAFFIC_DENSITY.md)
for current behaviour).

---

## 1. Off-lane edge-clipping → density under-count (`traffic_density.py`)

### Problem
Lane assignment (`assign_lanes`) is **per sample**: a vehicle's frame is "on-lane" only if its
centre point falls inside a driving-lane polygon, else `lane_idx = -1` and that frame is
excluded from the on-road metrics (`network`, `roads`, `by_type`, `roundabout`, area-occupancy).
The vehicle is never removed wholesale — its on-lane frames still count — but the off-lane
frames contribute nothing.

On the Saro roundabout this makes **12.3 % of kept samples off-lane**, and they sit right on the
lane edge: median 0.30 m, **94 % within 1 m, 100 % within 2 m** of a lane (per-vehicle off-lane
fraction: median 9 %, max 89 %; 0 vehicles fully removed). Causes:
- vehicles riding / cutting the circulatory edge;
- the **constant-width lane approximation** — `_build_driving_lanes` uses only the `<width>`
  `a` term, but ~half the lanes in the newer maps carry non-zero `b/c/d` (width varies along
  the road), so the built polygon is narrower/offset from the true lane.

### Consequence
Those frames are absent from the Edie sums, so on-road **density/flow are under-counted by
roughly the off-lane fraction** (~12 % on the roundabout, ~0 elsewhere). It is consistent
across the tool but a real low bias, and it inflates `data_quality.off_lane_fraction` for
vehicles that are legitimately on the road.

### Option A — snap to nearest lane within a tolerance (recommended)
In `assign_lanes`, if a point is not inside any lane, assign it to the nearest lane polygon when
that distance is below a tolerance (e.g. `--snap-tol-m`, default ~1.0 m). Implementation: after
the `contains` pass, for still-unassigned points query the STRtree by an expanded box (or
`nearest`) and accept if `polygon.distance(point) <= tol`.
- **Pros:** cheap; cuts off-lane 12 % → ~1 % and removes the density bias; parking is already
  handled separately so the snap won't pull parked cars onto lanes.
- **Cons:** a too-large tolerance could attach genuinely off-road samples to a lane; keep it
  small (≤ ~1.0 m, < half a lane width) and configurable.
- **Effect on numbers:** raises on-road densities/flows by ~the off-lane fraction (~12 % on the
  roundabout, negligible elsewhere); shrinks `off_lane_fraction`.

### Option B — evaluate non-constant lane width
Extend `_build_driving_lanes` to evaluate the full width polynomial
`w(ds) = a + b·ds + c·ds² + d·ds³` per sampled station (and support multiple `<width>` records
per lane by `sOffset`). More faithful geometry.
- **Pros:** fixes the *cause* for tapering/widening lanes; also improves drivable-area and
  area-occupancy accuracy.
- **Cons:** more code; does **not** fix vehicles that genuinely ride the lane edge, so a small
  off-lane residual remains. Best combined with a small snap (Option A).

### Recommendation
Do Option A first (small configurable snap) — it directly removes the bias for little code.
Consider Option B later if precise lane geometry is needed for other reasons.

---

## 2. Static-vehicle rule now partly redundant with parking areas (`clean_trajectories`)

`--static-disp-m` excludes a vehicle whose whole-track bounding box is < 2 m (parked). Since
parking areas are now parsed (`_parking_areas`) and in-parking samples are excluded per-sample,
the displacement rule mostly fires on the same parked cars. It is still useful for a car parked
on a **driving lane with no parking polygon** (e.g. roadside), so it is kept as a complement.
Option: drop it and rely solely on parking areas + a sustained-stationary (run-based) detector;
decide once more scenes are available.

---

## 3. Turning movements omit through-movements (`compute_turning_movements`)

Only movements that cross a junction **connector** road are counted (the actual turns). A
vehicle that goes straight through on the main road never enters a connector, so through-traffic
is not represented in the per-junction movement table. For a complete OD/turning matrix, also
infer the through movement (main-in → main-out) from the road sequence at each junction.

---

## 4. Single-UTM-zone assumption (coordinate alignment)

Trajectories and the map are assumed to share one UTM zone; alignment is a constant offset
(`map_local = global_utm − proj(lon_0, lat_0)`), not a full reprojection. If a dataset ever
mixes zones, reproject trajectory coordinates into the map CRS with `pyproj` before assignment.
