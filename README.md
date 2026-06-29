# data-metrics

Quality metrics for **video** (MP4) and **OmegaPrime** (MCAP) datasets, plus a
dataset-level summary tool.

The package provides three standalone command-line tools:

| Tool | Purpose |
|---|---|
| `data_metrics.py` | Compute quality metrics for a **single** video or OmegaPrime MCAP file and write the result as JSON. |
| `data_metrics_summary.py` | Aggregate many such JSON files under a directory into a **dataset-level** summary (tables + optional JSON). |
| `traffic_density.py` | Compute **traffic density / speed / flow / LOS** for a scene from vehicle trajectories (OmegaPrime MCAP or OpenLABEL JSON) grounded on an OpenDRIVE map. |

`data_metrics.py` depends on numpy, OpenCV, PyAV (video) and pandas + mcap
(OmegaPrime). `data_metrics_summary.py` is pure Python standard library.
`traffic_density.py` depends on pyproj + shapely (and pandas + mcap for the
OmegaPrime source).

---

## Installation

```bash
pip install .            # or: uv pip install .
```

This installs three console commands, `data-metrics`, `data-metrics-summary` and
`traffic-density`. You can also run the scripts directly (`./data_metrics.py`,
`./data_metrics_summary.py`, `./traffic_density.py`).

---

## Quick start

```bash
# Per-file metrics
data-metrics --video clip.mp4 --json clip_qm.json
data-metrics --omegaprime scene.mcap --json scene_qm.json

# Both at once, with a campaign-specific class spec
data-metrics --video clip.mp4 --omegaprime scene.mcap \
             --dataset-spec spec.json --json combined_qm.json

# Dataset-level summary over a tree of metric JSON files
data-metrics-summary --root ./results --d31 --json summary.json
```

---

## `data_metrics.py` — per-file metrics

```
data-metrics [--video FILE] [--omegaprime FILE]
             [--expected-hz HZ] [--stride N] [--max-frames N]
             [--dataset-spec FILE] [--no-role-check] [--json FILE]
```

| Option | Meaning |
|---|---|
| `--video FILE` | Compute video metrics for an MP4 file. |
| `--omegaprime FILE` | Compute OmegaPrime metrics for an MCAP file. |
| `--expected-hz HZ` | Expected sampling frequency. If omitted, the target frame interval is inferred from the data. |
| `--stride N` | Sample every N-th frame for the image-based video metrics (default 5). |
| `--max-frames N` | Cap on frames loaded into memory for video metrics (default 500). |
| `--dataset-spec FILE` | JSON file declaring expected/required object classes (see [Dataset spec](#dataset-spec-file)). |
| `--no-role-check` | Exclude `role` from the Class Completeness diversity check. |
| `--json FILE` | Write the result JSON to FILE (also printed to stdout). |

At least one of `--video` / `--omegaprime` is required.

### Output schema

The result is a single JSON object with a `video` and/or `omegaprime` section.
Each metric is stored under the name of the function that produced it:

```jsonc
{
  "video": {
    "compute_temporal_metrics":            { "Temporal Completeness (%)": .., "Temporal Distortion Score (TDS)": .. },
    "compute_duplicate_record_rate_video": { "Duplicate Record Rate (%)": .. },
    "compute_sensor_consistency":          { "Composite SCI (0..1)": .., "Metrics": { .. } },
    "compute_sensor_degradation":          { "Partial Blockage Score (0..1)": .., "FOV Change Score (0..1)": .. }
  },
  "omegaprime": {
    "file_metadata":          { "passes": .. },
    "attribute_completeness": { "completeness_pct": .., "passes_threshold": .. },
    "record_completeness":    { "completeness_pct": .., "passes_threshold": .. },
    "class_completeness":     { "case2_passes": .., "case2_checks": { .. } },
    "format_consistency":     { "consistency_pct": .., "passes_threshold": .. },
    "duplicate_rate":         { "duplicate_rate_pct": .., "passes_threshold": .. },
    "temporal_completeness":  { "completeness_pct": .., "passes_threshold": .. },
    "object_type_coverage":   { "passes": .. },
    "trajectory_plausibility":{ "implausible_fraction": .., "passes_threshold": .. }
  }
}
```

A metric that fails to compute is replaced by `{ "error": "<message>" }` rather than
aborting the run.

---

## `data_metrics_summary.py` — dataset summary

```
data-metrics-summary --root DIR [--pattern GLOB ...] [--d31] [--json FILE]
```

| Option | Meaning |
|---|---|
| `--root DIR` | Directory scanned **recursively** for metric JSON files. |
| `--pattern GLOB` | Filename glob to match (repeatable; default `*.json`). |
| `--d31` | Show the full metric set with formal names and the extra OmegaPrime pass/fail columns (class completeness, object type coverage, trajectory plausibility, file metadata). |
| `--json FILE` | Also write the aggregated summary as JSON. |

Each matched file is auto-classified by content: a file contributes a **video** row
if it has a top-level `video` key and an **OmegaPrime** row if it has an `omegaprime`
key (a single file may contribute both). Files with neither are skipped. The tool
prints aggregate statistics (mean / min / max / n) and a per-file breakdown, with
sub-threshold values flagged.

---

## `traffic_density.py` — traffic density / flow / roundabout metrics

```
traffic-density (--omegaprime FILE | --openlabel FILE) [--opendrive MAP.xodr]
                [--expected-hz HZ] [--los-spec FILE] [--time-series]
                [--min-track-s S] [--static-disp-m M] [--max-speed-ms V] [--json FILE]
```

| Option | Meaning |
|---|---|
| `--opendrive FILE` | OpenDRIVE `.xodr` map supplying lane geometry, lengths and the drivable surface. Optional with `--omegaprime`: if omitted, the map embedded in the MCAP (`ground_truth_map`) is used. Required for `--openlabel`. |
| `--omegaprime FILE` | OmegaPrime MCAP trajectory source (has velocity). |
| `--openlabel FILE` | OpenLABEL JSON trajectory source (world cuboids); velocity is finite-differenced. May be plain or gzip/zip/bz2/xz compressed (auto-detected). |
| `--expected-hz HZ` | Frame rate used for OpenLABEL velocity (default 30). |
| `--los-spec FILE` | JSON list `[[grade, upper_control_delay_s], …]` overriding the default HCM delay-based LOS thresholds. |
| `--min-track-s S` | Drop tracks shorter than this (s); default 0.5 — removes pseudo-label flicker. |
| `--static-disp-m M` | Exclude vehicles whose centre moves less than this over their life, i.e. **parked** (m); default 2.0. |
| `--max-speed-ms V` | Drop samples faster than this as teleports / ID-switches (m/s); default 55. |
| `--time-series` | Include per-frame density/occupancy arrays in the output. |
| `--json FILE` | Write the result JSON to FILE (also printed to stdout). |

Exactly one of `--omegaprime` / `--openlabel` is required. An OmegaPrime MCAP embeds its
own OpenDRIVE map, so `--opendrive` may be omitted for an MCAP source; an OpenLABEL source
always needs `--opendrive`. Vehicle positions (global UTM) are aligned to the map's local
frame using the map `geoReference`, then each vehicle is assigned to the driving lane that
contains it.

**Trajectory cleaning.** Before any metric, tracks are cleaned (uniformly across sources, no
confidence field exists in the data): short tracks are dropped (flicker), teleport/ID-switch
samples are removed, and **parked/static vehicles are excluded from the traffic metrics** —
they are not traffic and otherwise dominate counts (queued vehicles move eventually and are
kept). Vehicles inside a mapped **parking area** (`<object type="parking">` outlines, which the
tool parses into polygons) are likewise treated as parking, not road traffic — so a car that is
parked and later drives off contributes only its on-road portion. What was filtered is reported
in `data_quality`; thresholds are CLI-configurable.

Density, flow and speed are reported with **Edie's generalized (space–time) definitions**
over each region (`k = mean_count/L`, `v = mean sample speed`, `q = k·v`), which are robust
to short segments. Per-frame summaries (mean / peak / p95 / min) are given for the
instantaneous quantities; the full per-frame series is available under `--time-series`.

See [TRAFFIC_DENSITY.md](TRAFFIC_DENSITY.md) for a detailed explanation of the pipeline,
the geometry, and how each metric is calculated and interpreted.

### Sections produced

| Section | Meaning | How |
|---|---|---|
| `data_quality` | How clean/reliable the (pseudo-)labelled clip is. | object count, static/short/teleport counts, **parking** vehicles/samples, off-lane fraction, unknown-type objects, track-duration stats, what was removed. |
| `geometry_agnostic` | Topology-free congestion indicators — the best cross-scene comparators (need only the drivable polygon). | area-occupancy fraction = Σ(L·W of on-road vehicles)/drivable area; veh/m². |
| `network` | Whole drivable network: Edie density/flow/speed + an instantaneous-density summary. | Edie over all driving lanes; instantaneous = on-road count / total lane-km. |
| `by_type` | Per coarse vehicle class (car/van/truck/bus/…): count + Edie density/flow/speed. | OSI subtype / OpenLABEL type grouped, Edie over the network. |
| `roads` | Per driving-road density/flow/speed — generic per-segment metric for any scene. | Edie per `road_id`, sorted by flow. |
| `junctions` | Turning movements (in-road → out-road counts, veh/h) per OpenDRIVE junction. | vehicle road sequences crossing each junction's connector roads. |
| `roundabout` | Per-approach and ring analysis (only when a real roundabout loop is detected). | see below. |

**`roundabout`** contains the circulating direction, the **ring** (Edie density/flow/speed,
area occupancy, control delay → LOS), and one entry per **approach** arm with: `direction`
(`entry`/`exit`, from radial-velocity sign), Edie density/flow/speed (the arm's cross-section
flow, since Edie `q = N/T`), `circulating_flow_veh_per_h` (total flow crossing the ring at
that arm — the conflicting flow that drives entry capacity), and control delay → LOS.
A roundabout is reported only when the circumferential lanes form a **near-complete loop**
(angular-sector coverage ≥ 75 %), so a short curved connector on a straight road is not
mistaken for a ring.

**LOS:** grades are the HCM **roundabout / unsignalized** scale by **control delay**
(s/veh: A ≤10, B ≤15, C ≤25, D ≤35, E ≤50, F >50) — the correct scale for interrupted flow,
not freeway density. Control delay is measured from trajectories (time lost vs the region's
85th-percentile free-flow speed). Override the thresholds with `--los-spec`.

---

## Metrics reference

Each metric below lists **what it means**, **how it is computed**, and a **flag
threshold** — the level at or below which `data_metrics_summary` highlights the value.
Video thresholds are empirical and may be tuned; OmegaPrime thresholds are stored in
each metric's JSON as `passes_threshold`.

Metrics marked **[stability/usability]** are heuristic indicators rather than
correctness measures.

### Video metrics

#### Temporal Completeness
**Meaning:** fraction of expected time steps actually present — detects dropped frames.
**How:** extract per-frame PTS timestamps (PyAV, falling back to PyAV `frame.time`).
With `duration = last − first` and a target interval `Δt` (`1/expected_hz` if given,
else the mean inter-frame interval):
`expected = round(duration / Δt) + 1`, `completeness% = actual_frames / expected × 100`.
**Flag:** < 95%.
**Limitation:** when `--expected-hz` is not supplied, `Δt` is inferred from the mean
interval, so a uniformly dropped-frame file understates the gap.

#### Temporal Distortion Score (TDS) — **[stability/usability]**
**Meaning:** *regularity* of the inter-frame interval (jitter). A clip can be 100%
complete yet have a poor TDS if frames are unevenly spaced.
**How:** with relative interval error `e(i) = (Δt(i) − Δt_target) / Δt_target`,
`δ_rms = √(mean(e²))` and `TDS = max(0, 1 − δ_rms)`, clamped to [0, 1].
**Flag:** < 0.95.
**Limitation:** self-references the observed mean when `--expected-hz` is unknown, so
it will not detect a systematic offset in capture rate.

#### Duplicate Record Rate
**Meaning:** fraction of duplicated frames (serialization/merge error).
**How:** counts frames whose PTS timestamp (ms) appears more than once. (An optional
32×32 grayscale thumbnail MD5 hash check exists but is off by default.)
`rate% = duplicates / total_frames × 100`.
**Flag:** > 1% (the summary reports the inverted *Unique Record Rate* `= 100 − rate`,
flagged below 99%).

#### Sensor Consistency Index (SCI) — **[stability/usability]**
**Meaning:** frame-to-frame photometric stability (brightness, color, sharpness, noise,
flicker, histogram) — detects exposure swings, AGC artefacts, flicker.
**How:** six sub-metrics on sampled frames, each mapped to a stability score in [0, 1]
via a clamp threshold, then combined as a weighted sum:

| Sub-metric | Raw value | Clamp | Weight |
|---|---|---|---|
| Brightness | Mean relative deviation of per-frame mean luminance from median | 0.35 | 0.22 |
| Color | RMS of per-channel relative deviation from per-channel median | 0.35 | 0.22 |
| Sharpness | CoV of Laplacian variance across frames | 0.35 | 0.15 |
| Noise | CoV of per-frame noise std in low-gradient regions | 0.40 | 0.15 |
| Flicker | RMS brightness-change rate normalized by mean brightness | 0.40 | 0.18 |
| Histogram | Mean Bhattacharyya distance from median 64-bin grayscale histogram | 0.40 | 0.08 |

**Flag:** SCI < 0.70.
**Limitation:** clamp thresholds and weights are empirically chosen, not validated
against a labelled dataset.

#### Partial Blockage Score — **[stability/usability]**
**Meaning:** whether the lens is partially blocked/obscured, producing frames unsuitable
for annotation or training.
**How:** five penalty signals per frame, averaged over time: low-intensity coverage
(pixels ≤ 30/255), low-texture coverage (21×21 local variance ≤ 50), edge Gini across a
4×4 tile grid, mean tile entropy, and border occlusion (darkest 6%-wide border band).
Each is normalized to [0, 1]; `score = 1 − mean_penalty`.
**Flag:** < 0.70.
**Limitation:** all five signals fire on legitimately dark or featureless scenes (night,
fog, tunnels).

#### FOV Change Score — **[stability/usability]**
**Meaning:** detects zoom or letterboxing/pillarboxing that invalidates the camera
calibration assumed during annotation.
**How:** (1) ORB + RANSAC affine scale estimation every 5 frames against the first frame;
(2) per-frame border constant-fraction in each 6%-wide band. Three penalty terms (scale
deviation, scale CoV, border constant fraction) are normalized and averaged;
`score = 1 − penalty`.
**Flag:** < 0.70.
**Limitation:** ORB matching is skipped on textureless frames; the border check fires on
pillarboxed content even when nothing changes.

### OmegaPrime (MCAP) metrics

These read each OmegaPrime MCAP into a per-object DataFrame. Mandatory per-record columns
are: `total_nanos`, `idx`, `x/y/z`, `vel_x/y/z`, `acc_x/y/z`, `length/width/height`,
`roll/pitch/yaw`, `type`, `subtype`, `role`. Mandatory file-level fields are
`country_code`, `proj_string`, `host_vehicle_id`, `version`.

#### File Metadata
**Meaning:** the file-level header fields are present and valid.
**How:** read the first `GroundTruth` message; check `country_code` is a 3-digit ISO 3166-1
code (100–999), `proj_string` is non-empty, and `host_vehicle_id` is present.
**Result:** `passes = true` iff no issue.

#### Attribute Completeness
**Meaning:** fraction of mandatory **schema attributes** present — a structural check.
**How:** `(present per-frame columns + present file-level fields) / total mandatory × 100`.
**Threshold:** 100% (`passes_threshold` true iff nothing is missing).
**Limitation:** protobuf fields always carry a default value, so the per-frame part reads
100% unless the schema itself is wrong; file-level fields are the discriminating part.

#### Record Completeness
**Meaning:** fraction of records where every mandatory field is populated (non-null).
**How:** `rows_with_no_NaN / total_rows × 100`; per-field NaN fraction is also reported.
**Threshold:** ≥ 95%.
Because protobuf zero-defaults are indistinguishable from real zeros, the result also
carries a `default_value_issues` block with five supplementary checks — zero dimensions,
constant timestamp, non-monotonic timestamps, objects fixed at the origin, and all-unknown
type. `default_value_issues.passes` is true iff none trigger.

#### Class Completeness
**Meaning:** the expected object classes are present across all six enumerated fields
(`moving_object.type`, `vehicle_classification.type/role`, and the three
`traffic_light.classification` fields), where present.
**How — two modes:**
- **Case 1 (expected set known):** per-field `coverage_pct = present / expected × 100`,
  using the `expected_types/subtypes/roles` from the dataset spec.
- **Case 2 (default):** each field must contain ≥ 2 distinct *non-unknown* classes
  (the OSI `*_UNKNOWN` zero-value is excluded as a non-semantic default). The result
  carries `case2_checks` (per-field bool) and `case2_passes` (all-pass bool).

Traffic-light fields appear only when the MCAP contains at least one traffic light.
`role` can be excluded from Case 2 with `--no-role-check`.
**Threshold:** Case 2 — every checked field has ≥ 2 distinct non-unknown classes.

#### Data Format Consistency
**Meaning:** fraction of records conforming to type, enum, range, and vocabulary rules —
catches encoding/unit errors and invalid enum values.
**How:** five per-record checks; any failure marks the record invalid:
1. mandatory columns parse as int/float;
2. `type ∈ [0,4]`, `subtype ∈ [0,22]`, `role ∈ [0,10]` (OSI enum ranges);
3. `roll/pitch/yaw ∈ [−π, π]` (values outside indicate degrees);
4. `length/width/height` strictly > 0;
5. string label vocabulary (only when string label columns are present).

`consistency% = (total − invalid) / total × 100`.
**Threshold:** ≥ 95%.

#### Duplicate Record Rate
**Meaning:** fraction of exact duplicate records (serialization error).
**How:** count rows where the `(total_nanos, idx)` pair appears more than once;
`rate% = duplicates / total × 100`.
**Threshold:** ≤ 1%.
**Limitation:** catches exact duplicates only — not near-duplicates from float rounding.

#### Temporal Completeness
**Meaning:** within-track frame completeness — detects per-object annotation gaps.
**How:** infer the frame interval `Δt` as the median of within-track consecutive intervals
(or `1/expected_hz`). For each object (`idx`):
`n_expected = round((t_last − t_first) / Δt) + 1`, `n_actual = unique timestamps`; a gap is
flagged when an interval exceeds `1.5 × Δt`. `completeness% = Σ n_actual / Σ n_expected × 100`.
Globally empty frames are excluded — only within-track gaps count.
**Threshold:** ≥ 98%.
**Limitation:** unreliable when objects legitimately appear/disappear, since the first→last
span overstates the expected frame count.

#### Object Type Coverage
**Meaning:** whether the required object types are present in `moving_object.type`.
**How — two outputs:**
- **Primary (requires `required_types` from the spec):** binary presence check;
  `coverage_pct = matched / required × 100`, `passes` true iff none missing. Omitted when
  no `required_types` is supplied.
- **Informational (always):** per-type frame density, detections-per-frame stats, and
  unique object count.

**Threshold:** all required types present.
Distinct from Class Completeness, which covers all six enumerated fields with a Case-2
fallback; this checks only `moving_object.type` against a user-defined required set.

#### Trajectory Plausibility — **[stability/usability]**
**Meaning:** flags physically impossible motion (gross coordinate/velocity errors). It does
**not** compare against a reference — only against a speed limit.
**How — two per-object checks:** (1) velocity magnitude `√(vx² + vy² + vz²)` exceeding
`max_speed_ms` (default 50 m/s); (2) implied speed `Δposition / Δt` between consecutive
frames exceeding the same limit (using x/y). `implausible_fraction = implausible_objects / total_objects`.
**Threshold:** ≤ 2% of objects.
**Limitation:** the speed limit is fixed and type-agnostic; displacement ignores the z axis.

---

## Dataset spec file

Two metrics — Object Type Coverage and Class Completeness (Case 1) — need a
campaign-specific declaration of the expected classes. Supply it with `--dataset-spec`:

```json
{
  "required_types":    ["TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_ANIMAL"],
  "expected_types":    ["TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_ANIMAL"],
  "expected_subtypes": ["TYPE_CAR", "TYPE_COMPACT_CAR", "TYPE_VAN",
                        "TYPE_MOTORCYCLE", "TYPE_BICYCLE", "TYPE_BUS"],
  "expected_roles":    ["ROLE_UNKNOWN", "ROLE_CIVIL"]
}
```

All fields are optional; values may be OSI enum name strings or integers. Without the
spec, Object Type Coverage reports only the informational density section, and Class
Completeness runs in Case 2.

| Field | Valid string values |
|---|---|
| `required_types` / `expected_types` | `TYPE_UNKNOWN` `TYPE_OTHER` `TYPE_VEHICLE` `TYPE_PEDESTRIAN` `TYPE_ANIMAL` |
| `expected_subtypes` | `TYPE_UNKNOWN` `TYPE_OTHER` `TYPE_SMALL_CAR` `TYPE_COMPACT_CAR` `TYPE_CAR` `TYPE_LUXURY_CAR` `TYPE_VAN` `TYPE_HEAVY_TRUCK` `TYPE_SEMITRAILER` `TYPE_TRAILER` `TYPE_MOTORCYCLE` `TYPE_BICYCLE` `TYPE_BUS` `TYPE_TRAM` `TYPE_TRAIN` `TYPE_WHEELCHAIR` `TYPE_SEMITRACTOR` `TYPE_STANDUP_SCOOTER` `TYPE_MICROMOBILITY_DEVICE` `TYPE_WORK_MACHINE` `TYPE_WATERCRAFT` `TYPE_AIRCRAFT` `TYPE_LAND_VEHICLE` |
| `expected_roles` | `ROLE_UNKNOWN` `ROLE_OTHER` `ROLE_CIVIL` `ROLE_AMBULANCE` `ROLE_FIRE` `ROLE_POLICE` `ROLE_PUBLIC_TRANSPORT` `ROLE_ROAD_ASSISTANCE` `ROLE_GARBAGE_COLLECTION` `ROLE_ROAD_CONSTRUCTION` `ROLE_MILITARY` |

---

## Development

Known limitations and deferred design options are tracked in
[DEVELOPMENT.md](DEVELOPMENT.md).

---

## License

MIT — see [LICENSE](LICENSE).
