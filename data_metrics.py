#!/usr/bin/env python3
import json
import cv2
import numpy as np
import hashlib
import av
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


EPS = 1e-9

# Hard cap on frames loaded by compute_sensor_* functions. Prevents OOM when
# callers omit max_frames on long/high-resolution videos. Pass max_frames=None
# explicitly to read_video_frames() directly if you need unlimited frames.
_DEFAULT_MAX_FRAMES = 500

def compute_duplicate_record_rate_video(video_path, threshold=1.0, use_frame_hash=False):
    cap = cv2.VideoCapture(video_path)
    timestamps = []
    frame_hashes = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get timestamp in milliseconds
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamps.append(ts)
        
        if use_frame_hash:
            # Compute hash for frame content
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (32, 32))
            frame_hash = hashlib.md5(resized.tobytes()).hexdigest()
            frame_hashes.append(frame_hash)
    
    cap.release()
    
    total_records = len(timestamps)
    if total_records == 0:
        raise ValueError("No frames found in video.")
    
    # Detect duplicates by timestamp
    duplicate_by_ts = total_records - len(set(timestamps))
    
    # Detect duplicates by frame hash (optional)
    duplicate_by_hash = 0
    if use_frame_hash:
        duplicate_by_hash = total_records - len(set(frame_hashes))
    
    # Combine duplicates (timestamp OR hash)
    duplicate_count = max(duplicate_by_ts, duplicate_by_hash)
    
    duplicate_rate = (duplicate_count / total_records) * 100
    
    return {
        "Duplicate Record Rate (%)": duplicate_rate,
        "Duplicate Count": duplicate_count,
        "Total Frames": total_records,
        "Threshold (%)": threshold,
        "Status": "PASS" if duplicate_rate <= threshold else "FAIL"
    }


def compute_temporal_metrics(video_path, expected_frequency=None):

    container = av.open(video_path)
    stream = container.streams.video[0]

    # Time base converts integer PTS to seconds: seconds = pts * time_base
    tb = stream.time_base  # Fraction

    timestamps = []

    for frame in container.decode(stream):
        # Prefer PTS; fall back to frame.time if needed
        if frame.pts is not None:
            ts_sec = float(frame.pts * tb)
        else:
            # frame.time is PyAV's best effort timestamp (already in seconds), may be None
            ts_sec = frame.time if frame.time is not None else None

        if ts_sec is not None:
            timestamps.append(ts_sec)

    container.close()

    # Basic validation
    if len(timestamps) < 2:
        raise ValueError("Not enough frames to compute temporal metrics.")

    t0 = timestamps[0]
    timestamps = [t - t0 for t in timestamps]

    # Compute Δt(i)
    deltas = np.diff(timestamps)
    N = len(deltas)

    # Compute Δt_target (seconds per frame)
    if expected_frequency:
        delta_target = 1.0 / float(expected_frequency)
    else:
        # Use mean interval for variable frame rate streams
        delta_target = float(np.mean(deltas))

    # Root-mean-square of relative interval error
    delta_rms = float(np.sqrt(np.mean(((deltas - delta_target) / delta_target) ** 2)))

    # Temporal Distortion Score (clamped to [0, 1])
    TDS = float(max(0.0, 1.0 - delta_rms))

    # Temporal Completeness (%): expected number of samples based on duration and target Δt
    duration = timestamps[-1] - timestamps[0]
    expected_samples = int(round(duration / delta_target)) + 1
    completeness = float((len(timestamps) / expected_samples) * 100.0)

    return {
        "Temporal Distortion Score (TDS)": TDS,
        "delta_RMS": delta_rms,
        "Temporal Completeness (%)": completeness,
        "Δt_target": delta_target,
        "Number of consecutive frame pairs": N
    }
   

def _relative_deviation(x: np.ndarray, ref: float) -> np.ndarray:
    ref_safe = ref if abs(ref) > EPS else EPS
    return np.abs((x - ref) / ref_safe)

def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    ma = (c[w:] - c[:-w]) / float(w)
    # Pad to original length
    pad_left = w // 2
    pad_right = len(x) - len(ma) - pad_left
    return np.pad(ma, (pad_left, pad_right), mode='edge')

def _bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    # p and q must be normalized histograms
    bc = np.sum(np.sqrt(p * q + EPS))
    return float(np.sqrt(max(0.0, 1.0 - bc)))

def _coefficient_of_variation(x: np.ndarray) -> float:
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    denom = abs(mu) if abs(mu) > EPS else (abs(sigma) + EPS)
    return sigma / denom

def _safe_mean(x: List[float]) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def read_video_frames(
    video_path: str,
    max_frames: Optional[int] = None,
    sample_stride: int = 2,
    prefer_pyav: bool = True
) -> Tuple[List[np.ndarray], List[Optional[float]]]:

    frames: List[np.ndarray] = []
    timestamps: List[Optional[float]] = []

    used_pyav = False
    if prefer_pyav:
        try:
            import av
            container = av.open(video_path)
            stream = container.streams.video[0]
            tb = stream.time_base  # Fraction
            i = 0
            for frame in container.decode(stream):
                if (i % sample_stride) == 0:
                    # Timestamp
                    if frame.pts is not None:
                        ts = float(frame.pts * tb)
                    else:
                        ts = float(frame.time) if frame.time is not None else None
                    # Frame array
                    frm = frame.to_ndarray(format='bgr24')
                    frames.append(frm)
                    timestamps.append(ts)
                    if max_frames is not None and len(frames) >= max_frames:
                        break
                i += 1
            container.close()
            used_pyav = True
        except Exception:
            used_pyav = False

    if not used_pyav:
        cap = cv2.VideoCapture(video_path)
        i = 0
        while True:
            ok, frm = cap.read()
            if not ok:
                break
            if (i % sample_stride) == 0:
                frames.append(frm)
                ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                timestamps.append(float(ts))
                if max_frames is not None and len(frames) >= max_frames:
                    break
            i += 1
        cap.release()

    # Normalize timestamps to start ~0 if available
    if timestamps and timestamps[0] is not None:
        t0 = timestamps[0]
        timestamps = [(t - t0) if t is not None else None for t in timestamps]

    return frames, timestamps

@dataclass
class MetricResult:
    value: float           
    stability: float       # 0..1 mapped "stability"
    details: Dict[str, float]

@dataclass
class SensorConsistencyResult:
    brightness: MetricResult
    color: MetricResult
    sharpness: MetricResult
    noise: MetricResult
    flicker: MetricResult
    histogram: MetricResult
    composite_SCI: float
    weights: Dict[str, float]
    counts: Dict[str, int]  # frames used, etc.

def _compute_brightness_metrics(
    frames_bgr: List[np.ndarray],
    variability_threshold: float = 0.1
) -> MetricResult:
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]
    brightness = np.array([float(g.mean()) for g in grays], dtype=float)
    ref = float(np.median(brightness))
    deviations = _relative_deviation(brightness, ref)
    mean_dev = float(np.mean(deviations))
    # Stability mapped with linear clamp: 1 at 0 dev, 0 at threshold or worse
    stability = float(max(0.0, 1.0 - mean_dev / max(variability_threshold, EPS)))
    return MetricResult(
        value=mean_dev,
        stability=stability,
        details={
            "reference": ref,
            "median_deviation": float(np.median(deviations)),
            "mean_deviation": mean_dev
        }
    )

def _compute_color_metrics(
    frames_bgr: List[np.ndarray],
    variability_threshold: float = 0.15
) -> MetricResult:
    # Mean per channel (B,G,R), then relative deviation w.r.t. each channel's median
    means = []
    for f in frames_bgr:
        ch_means = np.mean(f.reshape(-1, 3), axis=0)  # [B, G, R]
        means.append(ch_means)
    means = np.array(means, dtype=float)  
    refs = np.median(means, axis=0)  # per-channel reference
    devs = np.abs((means - refs) / (np.where(np.abs(refs) > EPS, refs, EPS)))
    
    per_frame_rms = np.sqrt(np.mean(devs**2, axis=1))
    mean_rms = float(np.mean(per_frame_rms))
    stability = float(max(0.0, 1.0 - mean_rms / max(variability_threshold, EPS)))
    return MetricResult(
        value=mean_rms,
        stability=stability,
        details={
            "median_B": float(refs[0]),
            "median_G": float(refs[1]),
            "median_R": float(refs[2]),
            "mean_rms_channel_deviation": mean_rms
        }
    )

def _compute_sharpness_metrics(
    frames_bgr: List[np.ndarray],
    variability_threshold: float = 0.20
) -> MetricResult:
    # Laplacian variance as sharpness proxy; stability = low CoV across frames
    lap_vars = []
    for f in frames_bgr:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        lap_vars.append(float(lap.var()))
    lap_vars = np.array(lap_vars, dtype=float)
    cov = _coefficient_of_variation(lap_vars)
    # Map: 1 at CoV=0, 0 at CoV>=threshold
    stability = float(max(0.0, 1.0 - cov / max(variability_threshold, EPS)))
    return MetricResult(
        value=cov,
        stability=stability,
        details={
            "median_laplacian_variance": float(np.median(lap_vars)),
            "mean_laplacian_variance": float(np.mean(lap_vars)),
            "cov_laplacian_variance": cov
        }
    )

def _compute_noise_metrics(
    frames_bgr: List[np.ndarray],
    low_gradient_percentile: float = 20.0,
    variability_threshold: float = 0.20,
    downsample_factor: int = 2,
) -> MetricResult:
    """Estimate noise stability as CoV of per-frame noise std in flat (low-gradient) regions.

    Frames are downsampled by `downsample_factor` before Sobel computation — flat regions
    are large-scale features so resolution loss is negligible, but computation drops ~4x
    at the default factor of 2.
    """
    noise_stds = []
    for f in frames_bgr:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if downsample_factor > 1:
            H, W = gray.shape
            gray = cv2.resize(
                gray,
                (W // downsample_factor, H // downsample_factor),
                interpolation=cv2.INTER_AREA,
            )
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        thr = np.percentile(mag, low_gradient_percentile)
        mask = mag <= thr
        noise_stds.append(float(np.std(gray[mask])) if mask.any() else float(np.std(gray)))
    noise_stds = np.array(noise_stds, dtype=float)
    cov = _coefficient_of_variation(noise_stds)
    stability = float(max(0.0, 1.0 - cov / max(variability_threshold, EPS)))
    return MetricResult(
        value=cov,
        stability=stability,
        details={
            "median_noise_std": float(np.median(noise_stds)),
            "mean_noise_std": float(np.mean(noise_stds)),
            "cov_noise_std": cov
        }
    )

def _compute_flicker_metrics(
    brightness: np.ndarray,
    timestamps: Optional[List[Optional[float]]] = None,
    smoothing_window: int = 15,
    variability_threshold: float = 0.20
) -> MetricResult:
    
    ##Flicker index via high-pass brightness normalized by mean.
    
    b = brightness.astype(float)
    mu = float(np.mean(b))
    mu_safe = mu if mu > EPS else EPS

    detrended = b - _moving_average(b, smoothing_window)

    if timestamps is not None and all(t is not None for t in timestamps) and len(b) >= 2:
        t = np.array(timestamps, dtype=float)
        dt = np.diff(t)
        
        dt = np.where(dt > 1e-6, dt, 1e-6)
        db = np.diff(b)
        rate = db / dt  # brightness units per second
        rate_norm = rate / mu_safe
        flicker_value = float(np.sqrt(np.mean(rate_norm**2)))
    else:
        flicker_value = float(np.std(detrended) / mu_safe)

    # Map to stability: lower flicker_value -> higher stability.
    stability = float(max(0.0, 1.0 - flicker_value / max(variability_threshold, EPS)))

    return MetricResult(
        value=flicker_value,
        stability=stability,
        details={
            "mean_brightness": mu,
            "std_detrended": float(np.std(detrended)),
            "uses_timebase": 1.0 if (timestamps is not None and all(t is not None for t in timestamps)) else 0.0
        }
    )

def _compute_histogram_stability(
    frames_bgr: List[np.ndarray],
    bins: int = 64,
    variability_threshold: float = 0.10
) -> MetricResult:
    # Compare each frame's grayscale histogram to a reference (median) using Bhattacharyya.
    hists = []
    for f in frames_bgr:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).flatten().astype(float)
        hist /= (hist.sum() + EPS)
        hists.append(hist)
    hists = np.stack(hists, axis=0)
    # Reference: median histogram 
    ref = np.median(hists, axis=0)
    ref = ref / (ref.sum() + EPS)

    distances = np.array([_bhattacharyya_distance(h, ref) for h in hists], dtype=float)
    mean_dist = float(np.mean(distances))
    stability = float(max(0.0, 1.0 - mean_dist / max(variability_threshold, EPS)))

    return MetricResult(
        value=mean_dist,
        stability=stability,
        details={
            "mean_bhattacharyya_distance": mean_dist,
            "median_bhattacharyya_distance": float(np.median(distances))
        }
    )

# Composite consistency metric

def compute_sensor_consistency(
    video_path: str,
    variability_thresholds: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    max_frames: Optional[int] = _DEFAULT_MAX_FRAMES,
    sample_stride: int = 1,
    prefer_pyav: bool = True
) -> SensorConsistencyResult:
    """
    Compute a multi-metric Sensor Consistency Index (SCI) from an input video.

    Args:
        video_path: path to input video.
        variability_thresholds: dict of thresholds mapping metric deviation to stability (default below).
        weights: dict of metric weights for composite SCI (default below).
        max_frames: cap on frames loaded into RAM (default: _DEFAULT_MAX_FRAMES). Pass None only if
            you are certain the video fits in available memory.
        sample_stride: process every N-th frame (>=1).
        prefer_pyav: if True, use PyAV for real PTS timestamps when available.

    Returns:
        SensorConsistencyResult with per-metric details and composite SCI.
    """

    # Default thresholds 
    th = {
        "brightness": 0.35,
        "color": 0.35,
        "flicker": 0.40,
        "histogram": 0.40,
        "noise": 0.40,
        "sharpness": 0.35
    }
    if variability_thresholds:
        th.update(variability_thresholds)

    # Default weights 
    w = {
        "brightness": 0.22,
        "color": 0.22,
        "flicker": 0.18,
        "histogram": 0.08,
        "noise": 0.15,
        "sharpness": 0.15
    }
    if weights:
        w.update(weights)

    frames, timestamps = read_video_frames(
        video_path,
        max_frames=max_frames,
        sample_stride=sample_stride,
        prefer_pyav=prefer_pyav
    )
    if len(frames) < 2:
        raise ValueError("Not enough frames to compute sensor consistency metrics.")

    # Precompute brightness series for flicker (fast, needed before parallel dispatch)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    brightness = np.array([float(g.mean()) for g in grays], dtype=float)

    with ThreadPoolExecutor(max_workers=5) as ex:
        f_brightness = ex.submit(_compute_brightness_metrics, frames, th["brightness"])
        f_color      = ex.submit(_compute_color_metrics,      frames, th["color"])
        f_sharpness  = ex.submit(_compute_sharpness_metrics,  frames, th["sharpness"])
        f_noise      = ex.submit(_compute_noise_metrics,      frames, th["noise"])
        f_hist       = ex.submit(_compute_histogram_stability, frames, 64, th["histogram"])
        # flicker is already vectorized and cheap — run on this thread while others work
        m_flicker = _compute_flicker_metrics(
            brightness,
            timestamps=timestamps,
            variability_threshold=th["flicker"]
        )
    m_brightness = f_brightness.result()
    m_color      = f_color.result()
    m_sharpness  = f_sharpness.result()
    m_noise      = f_noise.result()
    m_hist       = f_hist.result()

    # Normalize weights
    keys = ["brightness", "color", "sharpness", "noise", "flicker", "histogram"]
    wvec = np.array([w[k] for k in keys], dtype=float)
    wvec = np.clip(wvec, 0.0, None)
    wsum = float(wvec.sum())
    wnorm = wvec / (wsum if wsum > EPS else 1.0)

    stabilities = np.array([
        m_brightness.stability,
        m_color.stability,
        m_sharpness.stability,
        m_noise.stability,
        m_flicker.stability,
        m_hist.stability
    ], dtype=float)

    composite_SCI = float(np.sum(wnorm * stabilities))

    return SensorConsistencyResult(
        brightness=m_brightness,
        color=m_color,
        sharpness=m_sharpness,
        noise=m_noise,
        flicker=m_flicker,
        histogram=m_hist,
        composite_SCI=composite_SCI,
        weights={k: float(wn) for k, wn in zip(keys, wnorm)},
        counts={
            "frames_used": len(frames),
            "has_true_timestamps": int(timestamps is not None and all(t is not None for t in timestamps))
        }
    )


def summarize_sensor_consistency(result: SensorConsistencyResult) -> Dict:
    """Return a plain dict summary (easy to print or log as JSON)."""
    out = {
        "Composite SCI (0..1)": result.composite_SCI,
        "Weights": result.weights,
        "Metrics": {
            "Brightness": asdict(result.brightness),
            "Color": asdict(result.color),
            "Sharpness": asdict(result.sharpness),
            "Noise": asdict(result.noise),
            "Flicker": asdict(result.flicker),
            "Histogram": asdict(result.histogram)
        }
    }
    out["Counts"] = result.counts
    return out



def boxvar(img: np.ndarray, ksize: int = 15) -> np.ndarray:
    """
    Local variance via box filter: var = E[x^2] - (E[x])^2
    img should be float32 [0..255].
    """
    k = (ksize, ksize)
    ex = cv2.boxFilter(img, ddepth=-1, ksize=k, normalize=True)
    ex2 = cv2.boxFilter(img * img, ddepth=-1, ksize=k, normalize=True)
    var = np.maximum(0.0, ex2 - ex * ex)
    return var

def tile_stats(arr: np.ndarray, tiles_xy: Tuple[int, int] = (4, 4)) -> np.ndarray:
    """Compute mean per spatial tile for an image-like array."""
    H, W = arr.shape[:2]
    tx, ty = tiles_xy
    tile_h = H // ty
    tile_w = W // tx
    vals = []
    for j in range(ty):
        for i in range(tx):
            y0, y1 = j * tile_h, (j + 1) * tile_h if j < ty - 1 else H
            x0, x1 = i * tile_w, (i + 1) * tile_w if i < tx - 1 else W
            tile = arr[y0:y1, x0:x1]
            vals.append(float(np.mean(tile)))
    return np.array(vals, dtype=float)

def entropy_gray(gray: np.ndarray, bins: int = 64) -> np.ndarray:
    """Local entropy via histogram over tiles; returns tile entropies as a 1D array."""
    H, W = gray.shape
    ty, tx = 4, 4
    tile_h = H // ty
    tile_w = W // tx
    ents = []
    for j in range(ty):
        for i in range(tx):
            y0, y1 = j * tile_h, (j + 1) * tile_h if j < ty - 1 else H
            x0, x1 = i * tile_w, (i + 1) * tile_w if i < tx - 1 else W
            tile = gray[y0:y1, x0:x1]
            hist = cv2.calcHist([tile], [0], None, [bins], [0, 256]).flatten().astype(float)
            p = hist / (hist.sum() + EPS)
            ents.append(float(-np.sum(p * np.log(p + EPS))))
    return np.array(ents, dtype=float)

def gini_coeff(x: np.ndarray) -> float:
    """Gini coefficient (0 uniform, 1 very unequal) for nonnegative vector."""
    x = np.array(x, dtype=float).flatten()
    x = x - np.min(x)
    s = np.sum(x) + EPS
    idx = np.arange(1, len(x) + 1)
    x_sorted = np.sort(x)
    return float((np.sum((2 * idx - len(x) - 1) * x_sorted)) / (len(x) * s))

# ==============================
# Metrics for degradation
# ==============================

@dataclass
class DegradationMetrics:
    partial_blockage_score: float      # 0..1 (1 = no blockage, 0 = heavy blockage)
    fov_change_score: float            # 0..1 (1 = no FOV change, 0 = large change)
    flags: Dict[str, bool]             
    details: Dict[str, float]          

def compute_partial_blockage_metrics(
    frames_bgr: List[np.ndarray],
    intensity_abs: float = 30.0,       # pixel value (0–255) below which a pixel is "dark"
    texture_abs: float = 50.0,         # local variance (σ≈7) below which a patch is "flat"
    border_band_frac: float = 0.06,    # 6% border bands
    tiles_xy: Tuple[int, int] = (4, 4)
) -> Dict[str, float]:
    """
    Detect partial blockage through:
    - Low-intensity coverage (dark regions)
    - Low-texture coverage (flat/low variance regions)
    - Border occlusion percentages
    - Edge density inequality 
    - Tile entropy (low entropy implies blockage)
    """
    low_int_coverages = []
    low_tex_coverages = []
    edge_ginis = []
    entropy_means = []

    border_lefts, border_rights, border_tops, border_bottoms = [], [], [], []

    for f in frames_bgr:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        H, W = gray.shape

        # Low-intensity: pixels genuinely below absolute threshold
        thr_int = intensity_abs
        low_int_mask = gray <= thr_int
        low_int_coverages.append(float(np.mean(low_int_mask)))

        # Low-texture: patches genuinely below absolute variance threshold
        var = boxvar(gray, ksize=21)
        thr_tex = texture_abs
        low_tex_mask = var <= thr_tex
        low_tex_coverages.append(float(np.mean(low_tex_mask)))

        # Border band occlusion (constant/dark likely)
        bw_l = int(max(1, border_band_frac * W))
        bh_t = int(max(1, border_band_frac * H))
        left_band = gray[:, :bw_l]
        right_band = gray[:, W - bw_l:]
        top_band = gray[:bh_t, :]
        bottom_band = gray[H - bh_t:, :]

        # Criterion: proportion of pixels both low intensity and low texture in band
        band_l_occ = float(np.mean((left_band <= thr_int)))
        band_r_occ = float(np.mean((right_band <= thr_int)))
        band_t_occ = float(np.mean((top_band <= thr_int)))
        band_b_occ = float(np.mean((bottom_band <= thr_int)))

        border_lefts.append(band_l_occ)
        border_rights.append(band_r_occ)
        border_tops.append(band_t_occ)
        border_bottoms.append(band_b_occ)

        # Edge density inequality (tiles)
        edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
        tile_edge_means = tile_stats(edges, tiles_xy=tiles_xy)
        edge_ginis.append(gini_coeff(tile_edge_means))

        # Tile entropy mean
        ents = entropy_gray(gray, bins=64)
        entropy_means.append(float(np.mean(ents)))

    # Aggregate across time
    return {
        "low_intensity_coverage_mean": float(np.mean(low_int_coverages)),
        "low_texture_coverage_mean": float(np.mean(low_tex_coverages)),
        "edge_gini_mean": float(np.mean(edge_ginis)),
        "tile_entropy_mean": float(np.mean(entropy_means)),
        "border_left_occ_mean": float(np.mean(border_lefts)),
        "border_right_occ_mean": float(np.mean(border_rights)),
        "border_top_occ_mean": float(np.mean(border_tops)),
        "border_bottom_occ_mean": float(np.mean(border_bottoms)),
    }

def estimate_fov_change_orb_affine(
    ref_bgr: np.ndarray,
    cur_bgr: np.ndarray
) -> Tuple[Optional[float], Optional[float], int]:
    """
    Estimate relative scale and rotation via ORB + RANSAC affine.
    Returns (scale, rotation_deg, inliers) 
    """
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(cur_gray, None)
    if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
        return None, None, 0

    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 20:
        return None, None, len(good)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return None, None, len(good)
    # Extract scale & rotation from affine
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    # Isotropic scale approx (average of row norms)
    s1 = np.sqrt(a * a + b * b)
    s2 = np.sqrt(c * c + d * d)
    scale = float((s1 + s2) / 2.0)
    rotation = float(np.degrees(np.arctan2(b, a)))
    inlier_count = int(np.sum(inliers)) if inliers is not None else len(good)
    return scale, rotation, inlier_count

def compute_fov_change_metrics(
    frames_bgr: List[np.ndarray],
    border_band_frac: float = 0.06,
    orb_stride: int = 5,
) -> Dict[str, float]:
    """FOV change via feature-based scale estimation and border constant bands.

    ORB+RANSAC is run every `orb_stride` frames (expensive: ~110 ms/frame at 1080p).
    Border constant-fraction check (letterboxing/pillarboxing) runs on every frame.
    """
    if len(frames_bgr) < 2:
        return {
            "scale_mean": 1.0,
            "scale_cov": 0.0,
            "rotation_mean_deg": 0.0,
            "border_constant_fraction_mean": 0.0
        }

    ref = frames_bgr[0]
    scales, rots, inliers = [], [], []
    const_fracs = []

    for i, f in enumerate(frames_bgr):
        # ORB scale/rotation — subsampled to avoid O(N * ORB_cost) dominance
        if i % orb_stride == 0:
            s, r, k = estimate_fov_change_orb_affine(ref, f)
            if s is not None:
                scales.append(s)
                rots.append(r)
                inliers.append(k)

        # Border constant fraction — cheap, run every frame
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        H, W = gray.shape
        bw = int(max(1, border_band_frac * W))
        bh = int(max(1, border_band_frac * H))
        bands = [
            gray[:, :bw], gray[:, W - bw:],
            gray[:bh, :], gray[H - bh:, :]
        ]
        const_frac = sum(float(np.mean(np.abs(b - np.mean(b)) < 2.0)) for b in bands) / 4.0
        const_fracs.append(const_frac)

    scales = np.array(scales, dtype=float) if scales else np.array([1.0])
    rots = np.array(rots, dtype=float) if rots else np.array([0.0])
    const_fracs = np.array(const_fracs, dtype=float)

    return {
        "scale_mean": float(np.mean(scales)),
        "scale_cov": float(np.std(scales) / (np.mean(scales) + EPS)),
        "rotation_mean_deg": float(np.mean(rots)),
        "border_constant_fraction_mean": float(np.mean(const_fracs)),
        "inliers_median": float(np.median(inliers)) if inliers else 0.0
    }

# ==============================
# Top-level degradation function
# ==============================

def compute_sensor_degradation(
    video_path: str,
    max_frames: Optional[int] = _DEFAULT_MAX_FRAMES,
    sample_stride: int = 1,
    prefer_pyav: bool = True,
    thresholds: Optional[Dict[str, float]] = None
) -> DegradationMetrics:
    """
    Focused detection of sensor degradation:
      - Partial blockage / occlusion
      - FOV change (zoom/crop/letterboxing)
    Returns scores in [0,1] where 1 = healthy, 0 = severe degradation.
    """
    frames, timestamps = read_video_frames(
        video_path,
        max_frames=max_frames,
        sample_stride=sample_stride,
        prefer_pyav=prefer_pyav
    )
    if len(frames) < 2:
        return DegradationMetrics(
            partial_blockage_score=1.0,
            fov_change_score=1.0,
            flags={"insufficient_frames": True},
            details={}
        )

    # Default thresholds (can be tuned)
    T = {
        # Partial blockage thresholds
        "low_intensity_coverage_warn": 0.08,     # >8% dark area suggests occlusion
        "low_texture_coverage_warn": 0.12,       # >12% flat area (local variance low)
        "edge_gini_warn": 0.35,                  # edge inequality across tiles (0..1)
        "tile_entropy_warn": 3.5,                # low entropy means blockage 
        "border_occ_warn": 0.08,                 # >8% of a border band is dark/constant

        # FOV thresholds
        "scale_change_warn": 0.05,               # 5% scale change from ref
        "scale_cov_warn": 0.10,                  # high variability of scale
        "border_constant_fraction_warn": 0.08    # letter/pillar boxing or hard crop
    }
    if thresholds:
        T.update(thresholds)

    # Compute metrics in parallel — independent of each other
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_pb = ex.submit(compute_partial_blockage_metrics, frames, 30.0, 50.0, 0.06, (4, 4))
        f_fv = ex.submit(compute_fov_change_metrics, frames)
    pb = f_pb.result()
    fv = f_fv.result()

    # Partial blockage score mapping (lower coverage -> higher score)
    # Penalty components normalized by thresholds (clamped)
    pb_penalty = 0.0
    pb_penalty += min(1.0, pb["low_intensity_coverage_mean"] / (T["low_intensity_coverage_warn"] + EPS))
    pb_penalty += min(1.0, pb["low_texture_coverage_mean"] / (T["low_texture_coverage_warn"] + EPS))
    pb_penalty += min(1.0, pb["edge_gini_mean"] / (T["edge_gini_warn"] + EPS))
    # Entropy
    entropy_shortfall = max(0.0, (T["tile_entropy_warn"] - pb["tile_entropy_mean"]) / max(T["tile_entropy_warn"], EPS))
    pb_penalty += min(1.0, entropy_shortfall)
    # Border occlusion
    border_max = max(pb["border_left_occ_mean"], pb["border_right_occ_mean"], pb["border_top_occ_mean"], pb["border_bottom_occ_mean"])
    pb_penalty += min(1.0, border_max / (T["border_occ_warn"] + EPS))
    # Average 5 penalties
    partial_blockage_score = float(max(0.0, 1.0 - (pb_penalty / 5.0)))

    # FOV change score mapping
    # Penalize scale deviation from 1 (zoom), scale variability, and constant border fraction
    scale_dev = abs(fv["scale_mean"] - 1.0)
    fov_penalty = 0.0
    fov_penalty += min(1.0, scale_dev / (T["scale_change_warn"] + EPS))
    fov_penalty += min(1.0, fv["scale_cov"] / (T["scale_cov_warn"] + EPS))
    fov_penalty += min(1.0, fv["border_constant_fraction_mean"] / (T["border_constant_fraction_warn"] + EPS))
    fov_change_score = float(max(0.0, 1.0 - (fov_penalty / 3.0)))

    flags = {
        "partial_blockage_suspected": partial_blockage_score < 0.7,
        "fov_change_suspected": fov_change_score < 0.7,
        "border_intrusion": border_max >= T["border_occ_warn"],
        "scale_change": scale_dev >= T["scale_change_warn"]
    }

    details = {
        # Partial blockage details
        **pb,
        # FOV details
        **fv,
        # Scores
        "partial_blockage_score": partial_blockage_score,
        "fov_change_score": fov_change_score,
    }

    return DegradationMetrics(
        partial_blockage_score=partial_blockage_score,
        fov_change_score=fov_change_score,
        flags=flags,
        details=details
    )


def summarize_degradation(result: DegradationMetrics) -> Dict:
    out = {
        "Partial Blockage Score (0..1)": result.partial_blockage_score,
        "FOV Change Score (0..1)": result.fov_change_score,
        "Flags": result.flags,
        "Details": result.details
    }
    return out



_DATA_METRICS_VERSION = "1.0.0"


def write_qm_provenance(
    video_path: str,
    qm_path: str,
    metrics_summary: Dict,
    input_prov_path: Optional[str] = None,
) -> Optional[str]:
    """Write a dataprov provenance sidecar for a QM metrics JSON file.

    Creates DPR_<qm_stem>.json alongside the QM file, recording that
    data_metrics computed quality metrics from the given video.

    Returns the path to the provenance file, or None if dataprov is unavailable.
    """
    try:
        from dataprov import ProvenanceChain
    except ImportError:
        return None

    qm_file = Path(qm_path)
    prov_path = qm_file.parent / f"DPR_{qm_file.stem}.json"

    video_file = Path(video_path)
    location = video_file.parent.parent.parent.name   # session root -> location
    session = video_file.parent.parent.name
    stem = video_file.stem
    entity_id = f"{location}_{session}_{stem}_quality_metrics"

    chain = ProvenanceChain.create(
        entity_id=entity_id,
        initial_source=str(video_file.parent),
        description=f"Video quality metrics computed from {video_file.name}",
        tags=["drone", "video", "quality", "metrics"],
    )

    now = datetime.now(timezone.utc).isoformat()
    warnings_str = ""
    flags = metrics_summary.get("Degradation", {}).get("Flags", {})
    if any(flags.values()):
        warnings_str = "; ".join(k for k, v in flags.items() if v)

    input_prov_files = [input_prov_path] if input_prov_path else None

    chain.add(
        started_at=now,
        ended_at=now,
        tool_name="data_metrics",
        tool_version=_DATA_METRICS_VERSION,
        operation="video quality metrics computation",
        inputs=[str(video_file)],
        input_formats=["MP4"],
        outputs=[str(qm_file)],
        output_formats=["JSON"],
        input_provenance_files=input_prov_files,
        warnings=warnings_str,
        drl=4,
    )
    chain.save(str(prov_path))
    return str(prov_path)



_RGB_STREAMS = frozenset({"D", "V"})


def compute_sensor_type_coverage(
    seq_dir: str,
    required_streams: Optional[List[str]] = None,
) -> Dict:
    """Check which sensor stream types are present in a concatenated sequence directory.

    D and V streams are both RGB cameras and treated as equivalent — if either is
    present, any RGB stream requirement (D or V) is satisfied.

    Args:
        seq_dir: Path to a sequence directory under drone_concatenated/.
        required_streams: Stream type letters that must be present.
            Default: ["D"]. Use ["D", "T"] for thermal campaigns.
    """
    if required_streams is None:
        required_streams = ["D"]

    seq_path = Path(seq_dir)
    found_types: set = set()
    for mp4 in seq_path.glob("VIDc_*.mp4"):
        parts = mp4.stem.rsplit("_", 1)
        if len(parts) == 2:
            found_types.add(parts[-1].upper())

    missing = []
    for req in required_streams:
        req_upper = req.upper()
        if req_upper in _RGB_STREAMS:
            if not (found_types & _RGB_STREAMS):
                missing.append(req_upper)
        else:
            if req_upper not in found_types:
                missing.append(req_upper)

    present = sorted(found_types)
    n_required = len(required_streams)
    n_covered = n_required - len(missing)
    coverage_pct = round((n_covered / n_required * 100) if n_required > 0 else 100.0, 1)

    return {
        "coverage_pct": coverage_pct,
        "present_streams": present,
        "missing_streams": missing,
        "required_streams": [s.upper() for s in required_streams],
    }


def compute_temporal_coverage(
    srt_path: str,
    required_windows: List[Tuple[str, str]],
) -> Dict:
    """Compute recording coverage of required time-of-day windows from an SRT sidecar.

    Reads the first and last absolute timestamps from a DJI-style SRT subtitle file
    to determine the recording's time-of-day span, then computes overlap with each
    user-specified required window. Assumes recordings do not span midnight.

    Args:
        srt_path: Path to the VIDc_*.srt (or .SRT) sidecar file.
        required_windows: List of ("HH:MM", "HH:MM") tuples for required
            time-of-day intervals (24-hour clock).
    """
    import re

    if not required_windows:
        return {"error": "No required windows specified"}

    srt_file = Path(srt_path)
    if not srt_file.exists():
        return {"error": f"SRT file not found: {srt_path}"}

    _TS_RE = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)')
    timestamps = []
    try:
        with open(srt_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _TS_RE.search(line)
                if m:
                    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in m.group(1) else "%Y-%m-%d %H:%M:%S"
                    try:
                        timestamps.append(datetime.strptime(m.group(1), fmt))
                    except ValueError:
                        continue
    except Exception as exc:
        return {"error": f"SRT read failed: {exc}"}

    if not timestamps:
        return {"error": "No timestamps found in SRT file"}

    rec_start, rec_end = timestamps[0], timestamps[-1]

    def _to_minutes(dt: datetime) -> float:
        return dt.hour * 60 + dt.minute + dt.second / 60 + dt.microsecond / 60_000_000

    def _parse_hhmm(s: str) -> float:
        h, m = s.split(":")
        return int(h) * 60 + int(m)

    rec_start_m = _to_minutes(rec_start)
    rec_end_m = _to_minutes(rec_end)

    windows_result = []
    total_overlap = 0.0
    total_required = 0.0

    for win_start_str, win_end_str in required_windows:
        win_start_m = _parse_hhmm(win_start_str)
        win_end_m = _parse_hhmm(win_end_str)
        win_dur = win_end_m - win_start_m
        if win_dur <= 0:
            continue
        overlap = max(0.0, min(rec_end_m, win_end_m) - max(rec_start_m, win_start_m))
        windows_result.append({
            "window": f"{win_start_str}-{win_end_str}",
            "coverage_pct": round(overlap / win_dur * 100, 1),
            "overlap_minutes": round(overlap, 2),
        })
        total_overlap += overlap
        total_required += win_dur

    total_coverage_pct = round(total_overlap / total_required * 100, 1) if total_required > 0 else 0.0

    return {
        "recording_start": rec_start.strftime("%H:%M:%S"),
        "recording_end": rec_end.strftime("%H:%M:%S"),
        "windows": windows_result,
        "total_coverage_pct": total_coverage_pct,
    }


def compute_traffic_presence(
    video_path: str,
    sample_stride: int = 3,
    max_bg_frames: int = 120,
    threshold: float = 12.0,
    alpha: float = 0.75,
    max_workers: int = 4,
) -> Dict:
    """Estimate traffic presence using background subtraction and edge-weighted frame diff.

    Builds a median background from the first `max_bg_frames` sampled frames, then scores
    each subsequent frame by a weighted combination of pixel-level difference from that
    background and local edge density. Frames exceeding `threshold` are counted as "active".

    Args:
        video_path: Path to the input video file.
        sample_stride: Process every N-th decoded frame (reduces compute).
        max_bg_frames: Number of frames used to build the median background model.
        threshold: Score above which a frame is considered to contain traffic.
        alpha: Weight for edge-weighted diff vs. plain diff (0=plain, 1=edge-only).
        max_workers: Thread-pool size for parallel frame processing.

    Returns:
        Dict with keys: Traffic Presence Ratio, Active Frames, Total Frames, Mean Score.
    """
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    container = av.open(video_path)
    stream = container.streams.video[0]

    bg_buffer: List[np.ndarray] = []
    early_frames: List[Tuple[np.ndarray, np.ndarray]] = []  # (gray, edge)
    bg: Optional[np.ndarray] = None

    traffic_scores: List[float] = []
    active_count = 0

    q: queue.Queue = queue.Queue(maxsize=200)

    def _process_frame(data: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        gray, img = data
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        edge = cv2.magnitude(gx, gy)
        return gray, edge

    def _decoder() -> None:
        for i, frame in enumerate(container.decode(stream)):
            if i % sample_stride != 0:
                continue
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (256, 144)).astype(np.float32)
            q.put((gray, img))
        q.put(None)

    def _score(gray: np.ndarray, edge: np.ndarray, bg: np.ndarray) -> float:
        diff = np.abs(gray - bg)
        ew = edge / (np.mean(edge) + EPS)
        return float(alpha * np.mean(diff * ew) + (1 - alpha) * np.mean(diff))

    threading.Thread(target=_decoder, daemon=True).start()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        def _flush_future(f) -> None:
            nonlocal bg, active_count
            gray, edge = f.result()
            if bg is None:
                if len(bg_buffer) < max_bg_frames:
                    bg_buffer.append(gray)
                    early_frames.append((gray, edge))
                if len(bg_buffer) == max_bg_frames:
                    bg = np.median(np.stack(bg_buffer), axis=0)
                    for g, e in early_frames:
                        s = _score(g, e, bg)
                        traffic_scores.append(s)
                        if s > threshold:
                            active_count += 1
                    early_frames.clear()
            else:
                s = _score(gray, edge, bg)
                traffic_scores.append(s)
                if s > threshold:
                    active_count += 1

        while True:
            item = q.get()
            if item is None:
                break
            futures.append(executor.submit(_process_frame, item))
            pending = []
            for f in futures:
                if f.done():
                    _flush_future(f)
                else:
                    pending.append(f)
            futures = pending

        for f in as_completed(futures):
            _flush_future(f)

    container.close()

    scores_arr = np.array(traffic_scores)
    return {
        "Traffic Presence Ratio": float(active_count / max(len(scores_arr), 1)),
        "Active Frames": int(active_count),
        "Total Frames": int(len(scores_arr)),
        "Mean Score": float(np.mean(scores_arr)) if len(scores_arr) else 0.0,
    }


# ---------------------------------------------------------------------------
# OmegaPrime CSV metrics (D3.1-aligned)
# ---------------------------------------------------------------------------

# Mandatory columns in the OmegaPrime CSV export.
# File-level mandatory fields that must be present once per MCAP (not per-frame).
_OP_MANDATORY_FILE_FIELDS: List[str] = [
    "country_code", "proj_string", "host_vehicle_id", "version",
]

_OP_MANDATORY_COLUMNS: List[str] = [
    "total_nanos", "idx",
    "x", "y", "z",
    "vel_x", "vel_y", "vel_z",
    "acc_x", "acc_y", "acc_z",
    "length", "width", "height",
    "roll", "pitch", "yaw",
    "type", "subtype", "role",
]

# OSI integer enum valid ranges (inclusive), sourced from ASAM OSI spec.
# osi3::MovingObject::Type
_OSI_MOVING_OBJECT_TYPE: Dict[int, str] = {
    0: "TYPE_UNKNOWN", 1: "TYPE_OTHER", 2: "TYPE_VEHICLE",
    3: "TYPE_PEDESTRIAN", 4: "TYPE_ANIMAL",
}
# osi3::MovingObject::VehicleClassification::Type
_OSI_VEHICLE_SUBTYPE: Dict[int, str] = {
    0: "TYPE_UNKNOWN",  1: "TYPE_OTHER",       2: "TYPE_SMALL_CAR",
    3: "TYPE_COMPACT_CAR", 4: "TYPE_CAR",      5: "TYPE_LUXURY_CAR",
    6: "TYPE_VAN",      7: "TYPE_HEAVY_TRUCK", 8: "TYPE_SEMITRAILER",
    9: "TYPE_TRAILER", 10: "TYPE_MOTORCYCLE", 11: "TYPE_BICYCLE",
   12: "TYPE_BUS",     13: "TYPE_TRAM",       14: "TYPE_TRAIN",
   15: "TYPE_WHEELCHAIR", 16: "TYPE_SEMITRACTOR", 17: "TYPE_STANDUP_SCOOTER",
   18: "TYPE_MICROMOBILITY_DEVICE", 19: "TYPE_WORK_MACHINE",
   20: "TYPE_WATERCRAFT", 21: "TYPE_AIRCRAFT", 22: "TYPE_LAND_VEHICLE",
}
# osi3::MovingObject::VehicleClassification::Role
_OSI_VEHICLE_ROLE: Dict[int, str] = {
    0: "ROLE_UNKNOWN",  1: "ROLE_OTHER",       2: "ROLE_CIVIL",
    3: "ROLE_AMBULANCE", 4: "ROLE_FIRE",       5: "ROLE_POLICE",
    6: "ROLE_PUBLIC_TRANSPORT", 7: "ROLE_ROAD_ASSISTANCE",
    8: "ROLE_GARBAGE_COLLECTION", 9: "ROLE_ROAD_CONSTRUCTION",
   10: "ROLE_MILITARY",
}

# osi3::TrafficLight::Classification enums — verified against installed schema
_OSI_TL_COLOR: Dict[int, str] = {
    0: "COLOR_UNKNOWN", 1: "COLOR_OTHER", 2: "COLOR_RED",
    3: "COLOR_YELLOW",  4: "COLOR_GREEN", 5: "COLOR_BLUE", 6: "COLOR_WHITE",
}
_OSI_TL_MODE: Dict[int, str] = {
    0: "MODE_UNKNOWN", 1: "MODE_OTHER", 2: "MODE_OFF",
    3: "MODE_CONSTANT", 4: "MODE_FLASHING", 5: "MODE_COUNTING",
}
_OSI_TL_ICON: Dict[int, str] = {
     0: "ICON_UNKNOWN",                         1: "ICON_OTHER",
     2: "ICON_NONE",                            3: "ICON_ARROW_STRAIGHT_AHEAD",
     4: "ICON_ARROW_LEFT",                      5: "ICON_ARROW_DIAG_LEFT",
     6: "ICON_ARROW_STRAIGHT_AHEAD_LEFT",       7: "ICON_ARROW_RIGHT",
     8: "ICON_ARROW_DIAG_RIGHT",                9: "ICON_ARROW_STRAIGHT_AHEAD_RIGHT",
    10: "ICON_ARROW_LEFT_RIGHT",               11: "ICON_ARROW_DOWN",
    12: "ICON_ARROW_DOWN_LEFT",                13: "ICON_ARROW_DOWN_RIGHT",
    14: "ICON_ARROW_CROSS",                    15: "ICON_PEDESTRIAN",
    16: "ICON_WALK",                           17: "ICON_DONT_WALK",
    18: "ICON_BICYCLE",                        19: "ICON_PEDESTRIAN_AND_BICYCLE",
    20: "ICON_COUNTDOWN_SECONDS",              21: "ICON_COUNTDOWN_PERCENT",
    22: "ICON_TRAM",                           23: "ICON_BUS",
    24: "ICON_BUS_AND_TRAM",
}

# OmegaPrime taxonomy string names as exported by the pipeline (used by class
# completeness and vocabulary checks). These are pipeline-specific labels and
# may differ from OSI enum names (e.g. pipeline exports "DELIVERY_VAN" and
# "MOVING" rather than OSI's "VAN" / "ROLE_CIVIL").
_OP_OBJECT_TYPES: List[str] = ["OTHER", "VEHICLE", "PEDESTRIAN", "ANIMAL"]
_OP_VEHICLE_SUBTYPES: List[str] = [
    "OTHER", "SMALL_CAR", "COMPACT_CAR", "CAR", "MEDIUM_CAR", "LUXURY_CAR",
    "DELIVERY_VAN", "HEAVY_TRUCK", "BUS", "COACH", "TRAILER",
    "MOTORBIKE", "CYCLIST", "BICYCLE",
    "POLICE_CAR", "AMBULANCE", "FIRE_TRUCK",
]
_OP_ROLES: List[str] = ["OTHER", "CIVILIAN", "AMBULANCE", "FIRE", "POLICE", "PUBLIC_TRANSPORT", "MOVING"]

# Expected numeric dtype per mandatory column ("int" or "float").
# Used by format_consistency to detect non-parseable values.
_OP_COLUMN_TYPES: Dict[str, str] = {
    "total_nanos": "int",
    "idx":         "int",
    "x": "float", "y": "float", "z": "float",
    "vel_x": "float", "vel_y": "float", "vel_z": "float",
    "acc_x": "float", "acc_y": "float", "acc_z": "float",
    "length": "float", "width": "float", "height": "float",
    "roll": "float", "pitch": "float", "yaw": "float",
    "type":    "int",
    "subtype": "int",
    "role":    "int",
}

# Radian fields: valid range [-π, π]. Values outside indicate unit error (degrees).
_OP_RADIAN_COLUMNS: List[str] = ["roll", "pitch", "yaw"]
_PI = 3.141592653589793

# Dimension fields must be strictly positive.
_OP_POSITIVE_COLUMNS: List[str] = ["length", "width", "height"]


def _load_omegaprime_mcap(mcap_path: str):
    """Load an OmegaPrime MCAP file and return a pandas DataFrame.

    Iterates all GroundTruth messages on the 'ground_truth' topic, expands
    each moving_object into one row. Column names match _OP_MANDATORY_COLUMNS.
    """
    import pandas as pd
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    rows = []
    with open(mcap_path, "rb") as f:
        for _, _, _, gt in make_reader(f, decoder_factories=[DecoderFactory()]) \
                .iter_decoded_messages(topics=["ground_truth"]):
            total_nanos = gt.timestamp.seconds * 1_000_000_000 + gt.timestamp.nanos
            for mo in gt.moving_object:
                b = mo.base
                vc = mo.vehicle_classification
                rows.append({
                    "total_nanos": total_nanos,
                    "idx":         int(mo.id.value),
                    "x": b.position.x,    "y": b.position.y,    "z": b.position.z,
                    "vel_x": b.velocity.x, "vel_y": b.velocity.y, "vel_z": b.velocity.z,
                    "acc_x": b.acceleration.x, "acc_y": b.acceleration.y, "acc_z": b.acceleration.z,
                    "length": b.dimension.length, "width": b.dimension.width, "height": b.dimension.height,
                    "roll": b.orientation.roll, "pitch": b.orientation.pitch, "yaw": b.orientation.yaw,
                    "type":    int(mo.type),
                    "subtype": int(vc.type),
                    "role":    int(vc.role),
                })
    return pd.DataFrame(rows, columns=_OP_MANDATORY_COLUMNS)


def compute_omegaprime_file_metadata(mcap_path: str) -> Dict:
    """Check file-level MCAP fields: country_code, proj_string, host_vehicle_id.

    These are only accessible in the MCAP protobuf — not in CSV exports.
    country_code must be a 3-digit ISO numeric code (100–999).
    """
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    country_code = proj_string = host_vehicle_id = None
    with open(mcap_path, "rb") as f:
        for _, _, _, gt in make_reader(f, decoder_factories=[DecoderFactory()]) \
                .iter_decoded_messages(topics=["ground_truth"]):
            country_code = gt.country_code
            proj_string = gt.proj_string
            host_vehicle_id = int(gt.host_vehicle_id.val) if gt.HasField("host_vehicle_id") else None
            break  # file-level fields are identical in every frame

    issues = []
    if not country_code:
        issues.append("country_code missing or empty")
    elif not (country_code.isdigit() and 100 <= int(country_code) <= 999):
        issues.append(f"country_code not a valid 3-digit ISO numeric: {country_code!r}")
    if not proj_string:
        issues.append("proj_string missing or empty")
    if host_vehicle_id is None:
        issues.append("host_vehicle_id missing")

    return {
        "country_code": country_code,
        "proj_string": proj_string,
        "host_vehicle_id": host_vehicle_id,
        "issues": issues,
        "passes": len(issues) == 0,
    }


def compute_omegaprime_attribute_completeness(mcap_path: str) -> Dict:
    """Check that all mandatory OmegaPrime attributes are present.

    Covers both per-frame moving-object columns (_OP_MANDATORY_COLUMNS) and
    file-level fields that must appear once per MCAP (_OP_MANDATORY_FILE_FIELDS).
    Maps to D3.1 Attribute Completeness (mandatory). Threshold: 100%.
    """
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    # Per-frame columns
    df = _load_omegaprime_mcap(mcap_path)
    present_cols = set(df.columns)
    missing_cols = [c for c in _OP_MANDATORY_COLUMNS if c not in present_cols]
    n_frame_present = len(_OP_MANDATORY_COLUMNS) - len(missing_cols)

    # File-level mandatory fields (read from first frame only)
    file_present: List[str] = []
    file_missing: List[str] = []
    with open(mcap_path, "rb") as f:
        for _, _, _, gt in make_reader(f, decoder_factories=[DecoderFactory()]) \
                .iter_decoded_messages(topics=["ground_truth"]):
            for field in _OP_MANDATORY_FILE_FIELDS:
                if field == "country_code":
                    (file_present if gt.country_code else file_missing).append(field)
                elif field == "proj_string":
                    (file_present if gt.proj_string else file_missing).append(field)
                elif field == "host_vehicle_id":
                    try:
                        present = gt.HasField("host_vehicle_id")
                    except ValueError:
                        present = False
                    (file_present if present else file_missing).append(field)
                elif field == "version":
                    v = gt.version
                    present = bool(v.version_major or v.version_minor or v.version_patch)
                    (file_present if present else file_missing).append(field)
            break

    n_total = len(_OP_MANDATORY_COLUMNS) + len(_OP_MANDATORY_FILE_FIELDS)
    n_present_total = n_frame_present + len(file_present)
    return {
        "completeness_pct": round(n_present_total / n_total * 100, 2),
        "passes_threshold": n_present_total == n_total,
        "per_frame": {
            "mandatory_columns": _OP_MANDATORY_COLUMNS,
            "missing_columns": missing_cols,
        },
        "file_level": {
            "mandatory_fields": _OP_MANDATORY_FILE_FIELDS,
            "present_fields": file_present,
            "missing_fields": file_missing,
        },
        "extra_columns": sorted(present_cols - set(_OP_MANDATORY_COLUMNS)),
    }


def compute_omegaprime_record_completeness(mcap_path: str) -> Dict:
    """Fraction of records with all mandatory fields populated (no NaN).

    Also reports default_value_issues: fields that are present but implausibly zero,
    which protobuf default-value semantics make indistinguishable from NaN.
    Maps to D3.1 Record Completeness (mandatory). Threshold: ≥95%.
    """
    df = _load_omegaprime_mcap(mcap_path)
    mandatory_present = [c for c in _OP_MANDATORY_COLUMNS if c in df.columns]
    total = len(df)
    if total == 0:
        return {"completeness_pct": 0.0, "total_records": 0, "complete_records": 0,
                "nan_fraction_per_field": {}, "default_value_issues": {}}

    nan_fractions = {c: round(float(df[c].isna().mean()), 6) for c in mandatory_present}
    complete_mask = df[mandatory_present].notna().all(axis=1)
    n_complete = int(complete_mask.sum())

    # Detect implausible protobuf default zeros the NaN check cannot catch.
    dim_cols = [c for c in ("length", "width", "height") if c in df.columns]
    n_zero_dim = int((df[dim_cols] == 0).any(axis=1).sum()) if dim_cols else 0

    frame_ts = sorted(df["total_nanos"].unique()) if "total_nanos" in df.columns else []
    ts_constant = len(frame_ts) == 1 and total > 1
    ts_nonmonotonic = any(frame_ts[i] >= frame_ts[i + 1] for i in range(len(frame_ts) - 1))

    pos_cols = [c for c in ("x", "y", "z") if c in df.columns]
    if pos_cols and "idx" in df.columns:
        pos_zero = (df[pos_cols] == 0).all(axis=1)
        obj_all_zero = df.groupby("idx").apply(lambda g: pos_zero.loc[g.index].all())
        n_zero_pos = int(obj_all_zero.sum())
        total_objects = len(obj_all_zero)
    else:
        n_zero_pos = total_objects = 0

    all_type_unknown = bool("type" in df.columns and (df["type"] == 0).all())

    default_issues = {
        "zero_dimension_records": n_zero_dim,
        "timestamp_constant": ts_constant,
        "timestamp_nonmonotonic": ts_nonmonotonic,
        "objects_all_zero_position": n_zero_pos,
        "total_objects": total_objects,
        "all_type_unknown": all_type_unknown,
        "passes": not any([n_zero_dim > 0, ts_constant, ts_nonmonotonic,
                           n_zero_pos > 0, all_type_unknown]),
    }

    return {
        "completeness_pct": round(n_complete / total * 100, 2),
        "total_records": total,
        "complete_records": n_complete,
        "nan_fraction_per_field": nan_fractions,
        "threshold_pct": 95.0,
        "passes_threshold": (n_complete / total * 100) >= 95.0,
        "default_value_issues": default_issues,
    }


def _read_traffic_light_enums(mcap_path: str) -> Optional[Dict[str, set]]:
    """Return sets of observed traffic light enum integers, or None if no TLs found."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    colors: set = set()
    modes: set = set()
    icons: set = set()
    found_any = False
    with open(mcap_path, "rb") as f:
        for _, _, _, gt in make_reader(f, decoder_factories=[DecoderFactory()]) \
                .iter_decoded_messages(topics=["ground_truth"]):
            for tl in gt.traffic_light:
                found_any = True
                cl = tl.classification
                colors.add(int(cl.color))
                modes.add(int(cl.mode))
                icons.add(int(cl.icon))
    return {"colors": colors, "modes": modes, "icons": icons} if found_any else None


def compute_omegaprime_class_completeness(
    mcap_path: str,
    expected_types: Optional[List[int]] = None,
    expected_subtypes: Optional[List[int]] = None,
    expected_roles: Optional[List[int]] = None,
    check_role: bool = True,
) -> Dict:
    """Report class diversity across all 6 OmegaPrime enumerated fields.

    Maps to D3.1 Class Completeness (two cases):
    Case 1 (expected known): coverage_pct = present/expected × 100 per field.
    Case 2 (expected unknown): each field must contain ≥2 distinct classes.
    Traffic light fields (color, icon, mode) are included when TLs are present.
    """
    df = _load_omegaprime_mcap(mcap_path)
    observed_type_ints = set(int(v) for v in df["type"].dropna().unique()) if "type" in df.columns else set()
    observed_subtype_ints = set(int(v) for v in df["subtype"].dropna().unique()) if "subtype" in df.columns else set()
    observed_role_ints = set(int(v) for v in df["role"].dropna().unique()) if "role" in df.columns else set()

    observed_types = {_OSI_MOVING_OBJECT_TYPE.get(i, str(i)) for i in observed_type_ints}
    observed_subtypes = {_OSI_VEHICLE_SUBTYPE.get(i, str(i)) for i in observed_subtype_ints}
    observed_roles = {_OSI_VEHICLE_ROLE.get(i, str(i)) for i in observed_role_ints}

    ref_type_names = set(({i: _OSI_MOVING_OBJECT_TYPE.get(i, str(i)) for i in expected_types} if expected_types is not None else _OSI_MOVING_OBJECT_TYPE).values())
    ref_subtype_names = set(({i: _OSI_VEHICLE_SUBTYPE.get(i, str(i)) for i in expected_subtypes} if expected_subtypes is not None else _OSI_VEHICLE_SUBTYPE).values())
    ref_role_names = set(({i: _OSI_VEHICLE_ROLE.get(i, str(i)) for i in expected_roles} if expected_roles is not None else _OSI_VEHICLE_ROLE).values())

    def _field(observed: set, ref: set) -> Dict:
        pct = round(len(observed & ref) / len(ref) * 100, 1) if ref else 0.0
        return {"observed": sorted(observed), "absent": sorted(ref - observed), "coverage_pct": pct}

    tl_enums = _read_traffic_light_enums(mcap_path)

    # Exclude OSI zero-value defaults from diversity count — UNKNOWN (int 0) is the
    # unset/fallback value and should not count as a meaningful distinct class.
    case2_type_ints    = observed_type_ints    - {0}
    case2_subtype_ints = observed_subtype_ints - {0}
    case2_role_ints    = observed_role_ints    - {0}

    case2_checks: Dict[str, bool] = {
        "type":    len(case2_type_ints)    >= 2,
        "subtype": len(case2_subtype_ints) >= 2,
    }
    if check_role:
        case2_checks["role"] = len(case2_role_ints) >= 2
    if tl_enums is not None:
        case2_checks["tl_color"] = len(tl_enums["colors"]) >= 2
        case2_checks["tl_icon"] = len(tl_enums["icons"]) >= 2
        case2_checks["tl_mode"] = len(tl_enums["modes"]) >= 2

    result: Dict = {
        "custom_taxonomy": any(x is not None for x in [expected_types, expected_subtypes, expected_roles]),
        "check_role": check_role,
        "type": _field(observed_types, ref_type_names),
        "subtype": _field(observed_subtypes, ref_subtype_names),
        "role": _field(observed_roles, ref_role_names),
        "case2_checks": case2_checks,
        "case2_passes": all(case2_checks.values()),
    }

    if tl_enums is not None:
        def _tl_field(observed_ints: set, taxonomy: Dict[int, str]) -> Dict:
            observed_names = {taxonomy.get(i, str(i)) for i in observed_ints}
            all_names = set(taxonomy.values())
            return {"observed": sorted(observed_names), "absent": sorted(all_names - observed_names),
                    "coverage_pct": round(len(observed_names & all_names) / len(all_names) * 100, 1)}
        result["traffic_lights"] = {
            "color": _tl_field(tl_enums["colors"], _OSI_TL_COLOR),
            "mode":  _tl_field(tl_enums["modes"],  _OSI_TL_MODE),
            "icon":  _tl_field(tl_enums["icons"],  _OSI_TL_ICON),
        }

    return result


def compute_omegaprime_format_consistency(mcap_path: str) -> Dict:
    """Check that fields conform to expected data types and controlled vocabulary.

    Four checks applied per record:
    1. Numeric type conformance — each mandatory column must be parseable as
       int/float (per _OP_COLUMN_TYPES). Non-parseable cells are violations.
    2. OSI integer enum conformance — type, subtype, role must be valid OSI
       integer enum values per _OSI_MOVING_OBJECT_TYPE / _OSI_VEHICLE_SUBTYPE /
       _OSI_VEHICLE_ROLE.
    3. Radian range — roll, pitch, yaw must be in [-π, π]. Values outside this
       range indicate a unit error (degrees passed instead of radians).
    4. Positive dimensions — length, width, height must be > 0.
    5. Vocabulary conformance — type_name, subtype_name, role_name (optional
       string label columns) must match OmegaPrime enum vocabulary.

    Maps to D3.1 Data Format Consistency (mandatory). Threshold: ≥95%.
    """
    import pandas as pd
    df = _load_omegaprime_mcap(mcap_path)
    total = len(df)
    if total == 0:
        return {"consistency_pct": 100.0, "total_records": 0, "invalid_records": 0,
                "invalid_counts_per_field": {}}

    invalid_counts: Dict[str, int] = {}
    invalid_mask = np.zeros(total, dtype=bool)

    def _flag(col: str, bad_mask):
        n_bad = int(bad_mask.sum())
        if n_bad > 0:
            invalid_counts[col] = invalid_counts.get(col, 0) + n_bad
            invalid_mask.__ior__(bad_mask.values if hasattr(bad_mask, "values") else bad_mask)

    # 1. Numeric type conformance
    for col in _OP_COLUMN_TYPES:
        if col not in df.columns:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        _flag(col, coerced.isna() & ~df[col].isna())

    # 2. OSI integer enum conformance
    for col, valid_ints in [
        ("type",    set(_OSI_MOVING_OBJECT_TYPE)),
        ("subtype", set(_OSI_VEHICLE_SUBTYPE)),
        ("role",    set(_OSI_VEHICLE_ROLE)),
    ]:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        _flag(col, ~numeric.isin(valid_ints) & numeric.notna())

    # 3. Radian range [-π, π]
    for col in _OP_RADIAN_COLUMNS:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        _flag(col, (vals.abs() > _PI) & vals.notna())

    # 4. Positive dimensions
    for col in _OP_POSITIVE_COLUMNS:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        _flag(col, (vals <= 0) & vals.notna())

    # 5. String enum vocabulary (optional label columns)
    for col, valid_set in [
        ("type_name",    set(_OP_OBJECT_TYPES)),
        ("subtype_name", set(_OP_VEHICLE_SUBTYPES)),
        ("role_name",    set(_OP_ROLES)),
    ]:
        if col not in df.columns:
            continue
        _flag(col, ~df[col].astype(str).str.upper().isin(valid_set))

    n_invalid = int(invalid_mask.sum())
    consistency_pct = round((total - n_invalid) / total * 100, 2)
    return {
        "consistency_pct": consistency_pct,
        "total_records": total,
        "invalid_records": n_invalid,
        "invalid_counts_per_field": invalid_counts,
        "threshold_pct": 95.0,
        "passes_threshold": consistency_pct >= 95.0,
    }


def compute_omegaprime_duplicate_rate(mcap_path: str) -> Dict:
    """Fraction of duplicate (total_nanos, idx) pairs in the OmegaPrime CSV.

    Maps to D3.1 Duplicate Record Rate (mandatory). Threshold: ≤1%.
    """
    df = _load_omegaprime_mcap(mcap_path)
    total = len(df)
    if total == 0:
        return {"duplicate_rate_pct": 0.0, "total_records": 0, "duplicate_records": 0}

    n_duplicates = int(df.duplicated(subset=["total_nanos", "idx"]).sum())
    dup_rate_pct = round(n_duplicates / total * 100, 4)
    return {
        "duplicate_rate_pct": dup_rate_pct,
        "total_records": total,
        "duplicate_records": n_duplicates,
        "threshold_pct": 1.0,
        "passes_threshold": dup_rate_pct <= 1.0,
    }


def compute_omegaprime_temporal_completeness(
    mcap_path: str,
    expected_hz: Optional[float] = None,
) -> Dict:
    """Per-object track temporal completeness for OmegaPrime data.

    For each object's track, checks for gaps between its first and last observed
    frame. Globally empty periods (frames where no objects exist) are NOT counted
    as missing — only within-track gaps are flagged.

    This correctly handles annotation-derived data (e.g. pAn) where stationary
    or off-screen objects are legitimately absent, unlike sensor streams that
    record every frame.

    Maps to D3.1 Temporal Completeness (trajectory data variant). Threshold: ≥98%.
    completeness_pct = total actual object-frames / total expected object-frames
    where expected is computed per-object from first→last appearance.
    """
    df = _load_omegaprime_mcap(mcap_path)
    n_global_frames = df["total_nanos"].nunique()

    if n_global_frames < 2:
        return {
            "completeness_pct": 100.0,
            "n_objects": df["idx"].nunique(),
            "n_global_frames": n_global_frames,
            "objects_with_gaps": [],
            "threshold_pct": 98.0,
            "passes_threshold": True,
        }

    # Infer target Δt from within-track consecutive intervals (not global, to
    # avoid large cross-track jumps distorting the estimate).
    per_obj_diffs = []
    for _, grp in df.groupby("idx"):
        obj_ts = np.sort(grp["total_nanos"].unique())
        if len(obj_ts) >= 2:
            per_obj_diffs.extend(np.diff(obj_ts).tolist())

    if expected_hz is not None:
        target_dt_ns = 1e9 / expected_hz
    elif per_obj_diffs:
        target_dt_ns = float(np.median(per_obj_diffs))
    else:
        # Fallback: global median
        target_dt_ns = float(np.median(np.diff(np.sort(df["total_nanos"].unique()))))

    inferred_hz = round(1e9 / target_dt_ns, 2) if target_dt_ns > 0 else None
    gap_threshold_ns = target_dt_ns * 1.5

    total_actual = 0
    total_expected = 0
    objects_with_gaps = []

    for obj_idx, grp in df.groupby("idx"):
        obj_ts = np.sort(grp["total_nanos"].unique())
        n_actual = len(obj_ts)

        if n_actual < 2:
            total_actual += n_actual
            total_expected += n_actual
            continue

        obj_diffs = np.diff(obj_ts)
        n_expected = int(round((obj_ts[-1] - obj_ts[0]) / target_dt_ns)) + 1

        total_actual += n_actual
        total_expected += n_expected

        gap_mask = obj_diffs > gap_threshold_ns
        if gap_mask.any():
            gap_indices = np.where(gap_mask)[0]
            obj_missing = int(np.round(obj_diffs[gap_mask] / target_dt_ns - 1).sum())
            obj_gaps = [
                {
                    "at_nanos": int(obj_ts[gi]),
                    "gap_ns": int(obj_diffs[gi]),
                    "missing_frames": int(round(obj_diffs[gi] / target_dt_ns - 1)),
                }
                for gi in gap_indices[:5]  # cap details per object
            ]
            objects_with_gaps.append({
                "idx": int(obj_idx),
                "n_frames": n_actual,
                "n_expected": n_expected,
                "n_gaps": int(gap_mask.sum()),
                "missing_frames": obj_missing,
                "gaps": obj_gaps,
            })

    completeness_pct = round(total_actual / total_expected * 100, 2) if total_expected > 0 else 100.0
    n_objects = df["idx"].nunique()

    return {
        "completeness_pct": completeness_pct,
        "n_objects": n_objects,
        "n_objects_with_gaps": len(objects_with_gaps),
        "n_global_frames": n_global_frames,
        "total_actual_object_frames": total_actual,
        "total_expected_object_frames": total_expected,
        "inferred_hz": inferred_hz,
        "expected_hz": expected_hz,
        "threshold_pct": 98.0,
        "passes_threshold": completeness_pct >= 98.0,
        "objects_with_gaps": objects_with_gaps,
    }



def compute_omegaprime_object_type_coverage(
    mcap_path: str,
    required_types: Optional[List[int]] = None,
) -> Dict:
    """Binary presence of required object types (D3.1) plus temporal frame density.

    D3.1 formula: (Matched Required Object Types / Total Required Object Types) × 100.
    required_types must be provided as OSI integer enum values to compute the D3.1
    coverage_pct; without it only the informational frame density section is returned.
    """
    df = _load_omegaprime_mcap(mcap_path)
    total_frames = df["total_nanos"].nunique()
    if total_frames == 0:
        return {"total_frames": 0}

    observed_type_ints = set(int(v) for v in df["type"].dropna().unique()) if "type" in df.columns else set()

    result: Dict = {"total_frames": total_frames}

    if required_types is not None:
        req = {_OSI_MOVING_OBJECT_TYPE.get(i, str(i)): i for i in required_types}
        matched = {name for name, i in req.items() if i in observed_type_ints}
        missing = set(req.keys()) - matched
        result["required_types"] = sorted(req.keys())
        result["matched_types"] = sorted(matched)
        result["missing_types"] = sorted(missing)
        result["coverage_pct"] = round(len(matched) / len(req) * 100, 1) if req else 0.0
        result["passes"] = len(missing) == 0

    # Informational: fraction of frames in which each observed type appears
    frame_density: Dict[str, float] = {}
    for type_int, grp in df.groupby("type"):
        label = _OSI_MOVING_OBJECT_TYPE.get(int(type_int), str(type_int))
        frame_density[label] = round(grp["total_nanos"].nunique() / total_frames * 100, 2)
    counts_per_frame = df.groupby("total_nanos").size()
    result["frame_density_by_type"] = frame_density
    result["detections_per_frame"] = {
        "mean": round(float(counts_per_frame.mean()), 2),
        "min": int(counts_per_frame.min()),
        "max": int(counts_per_frame.max()),
        "median": float(counts_per_frame.median()),
    }
    result["unique_objects"] = int(df["idx"].nunique())
    return result


def compute_omegaprime_trajectory_plausibility(
    mcap_path: str,
    max_speed_ms: float = 50.0,
    max_implausible_fraction: float = 0.02,
) -> Dict:
    """Check per-object tracks for physically implausible speed or position jumps.

    Uses velocity vectors already in the CSV. Also cross-checks with positional
    displacement between consecutive frames per object. Maps loosely to D3.1
    Range Accuracy and Non-default valued attributes accuracy.

    Args:
        mcap_path: Path to OmegaPrime MCAP file.
        max_speed_ms: Maximum plausible speed in m/s (default 50 = 180 km/h).
    """
    df = _load_omegaprime_mcap(mcap_path)
    if df.empty:
        return {"n_objects": 0, "implausible_objects": [], "implausible_fraction": 0.0}

    df = df.sort_values(["idx", "total_nanos"])
    v_mag = np.sqrt(df["vel_x"] ** 2 + df["vel_y"] ** 2 + df["vel_z"] ** 2)
    df = df.copy()
    df["_v_mag"] = v_mag

    implausible_objects = []
    n_objects = df["idx"].nunique()

    for obj_id, grp in df.groupby("idx"):
        issues = []

        # Speed check from velocity field
        v = grp["_v_mag"]
        n_fast = int((v > max_speed_ms).sum())
        if n_fast > 0:
            issues.append({
                "type": "velocity_too_high",
                "max_speed_ms": round(float(v.max()), 2),
                "n_frames": n_fast,
            })

        # Position jump check: displacement between consecutive frames
        if len(grp) >= 2:
            dt_s = np.diff(grp["total_nanos"].values) / 1e9
            dx = np.diff(grp["x"].values)
            dy = np.diff(grp["y"].values)
            dist = np.sqrt(dx ** 2 + dy ** 2)
            # Avoid division by zero for same-timestamp rows
            valid = dt_s > 0
            if valid.any():
                implied_speed = np.where(valid, dist / np.where(valid, dt_s, 1.0), 0.0)
                n_jumps = int((implied_speed > max_speed_ms).sum())
                if n_jumps > 0:
                    issues.append({
                        "type": "position_jump",
                        "max_implied_speed_ms": round(float(implied_speed.max()), 2),
                        "n_jumps": n_jumps,
                    })

        if issues:
            implausible_objects.append({"idx": int(obj_id), "issues": issues})

    n_implausible = len(implausible_objects)
    return {
        "n_objects": n_objects,
        "n_implausible": n_implausible,
        "implausible_fraction": round(n_implausible / n_objects, 4) if n_objects > 0 else 0.0,
        "max_speed_ms_threshold": max_speed_ms,
        "max_implausible_fraction_threshold": max_implausible_fraction,
        "implausible_objects": implausible_objects,
        "passes_threshold": (n_implausible / n_objects if n_objects > 0 else 0.0) <= max_implausible_fraction,
    }


# ---------------------------------------------------------------------------
# Convenience runners + metric registry
# ---------------------------------------------------------------------------

METRIC_CATEGORIES: Dict[str, List[str]] = {
    "video": [
        "compute_temporal_metrics",
        "compute_duplicate_record_rate_video",
        "compute_sensor_consistency",
        "compute_sensor_degradation",
        "compute_sensor_type_coverage",
        "compute_temporal_coverage",
        "compute_traffic_presence",
        "compute_partial_blockage_metrics",
        "compute_fov_change_metrics",
    ],
    "omegaprime": [
        "compute_omegaprime_file_metadata",
        "compute_omegaprime_attribute_completeness",
        "compute_omegaprime_record_completeness",
        "compute_omegaprime_class_completeness",
        "compute_omegaprime_format_consistency",
        "compute_omegaprime_duplicate_rate",
        "compute_omegaprime_temporal_completeness",
        "compute_omegaprime_object_type_coverage",
        "compute_omegaprime_trajectory_plausibility",
    ],
}


def run_video_metrics(
    video_path: str,
    expected_frequency: Optional[float] = None,
    sample_stride: int = 5,
    max_frames: int = _DEFAULT_MAX_FRAMES,
) -> Dict:
    """Run all video quality metrics for a single video file and return a combined dict.

    Keys match the function names, values are the raw metric dicts.
    """
    results: Dict[str, object] = {}
    results["compute_temporal_metrics"] = compute_temporal_metrics(
        video_path, expected_frequency=expected_frequency)
    results["compute_duplicate_record_rate_video"] = compute_duplicate_record_rate_video(video_path)
    try:
        sc = compute_sensor_consistency(video_path, sample_stride=sample_stride, max_frames=max_frames)
        results["compute_sensor_consistency"] = summarize_sensor_consistency(sc)
    except Exception as e:
        results["compute_sensor_consistency"] = {"error": str(e)}
    try:
        sd = compute_sensor_degradation(video_path, sample_stride=sample_stride, max_frames=max_frames)
        results["compute_sensor_degradation"] = summarize_degradation(sd)
    except Exception as e:
        results["compute_sensor_degradation"] = {"error": str(e)}
    return results


def run_omegaprime_metrics(
    mcap_path: str,
    expected_hz: Optional[float] = None,
    max_speed_ms: float = 50.0,
    expected_types: Optional[List[int]] = None,
    expected_subtypes: Optional[List[int]] = None,
    expected_roles: Optional[List[int]] = None,
    required_types: Optional[List[int]] = None,
    check_role: bool = True,
) -> Dict:
    """Run all OmegaPrime quality metrics for a single MCAP file and return a combined dict.

    Keys match the function names, values are the raw metric dicts.
    """
    results: Dict[str, object] = {}
    try:
        results["file_metadata"] = compute_omegaprime_file_metadata(mcap_path)
    except Exception as e:
        results["file_metadata"] = {"error": str(e)}

    for fn_name, fn in [
        ("attribute_completeness", compute_omegaprime_attribute_completeness),
        ("record_completeness", compute_omegaprime_record_completeness),
        ("class_completeness", compute_omegaprime_class_completeness),
        ("format_consistency", compute_omegaprime_format_consistency),
        ("duplicate_rate", compute_omegaprime_duplicate_rate),
        ("temporal_completeness", compute_omegaprime_temporal_completeness),
        ("object_type_coverage", compute_omegaprime_object_type_coverage),
    ]:
        try:
            if fn_name == "temporal_completeness":
                results[fn_name] = fn(mcap_path, expected_hz=expected_hz)
            elif fn_name == "class_completeness":
                results[fn_name] = fn(mcap_path, expected_types=expected_types,
                                      expected_subtypes=expected_subtypes,
                                      expected_roles=expected_roles,
                                      check_role=check_role)
            elif fn_name == "object_type_coverage":
                results[fn_name] = fn(mcap_path, required_types=required_types)
            else:
                results[fn_name] = fn(mcap_path)
        except Exception as e:
            results[fn_name] = {"error": str(e)}

    try:
        results["trajectory_plausibility"] = compute_omegaprime_trajectory_plausibility(
            mcap_path, max_speed_ms=max_speed_ms)
    except Exception as e:
        results["trajectory_plausibility"] = {"error": str(e)}

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute quality metrics for a video file or OmegaPrime MCAP."
    )
    parser.add_argument("--video", metavar="FILE", help="Path to video file")
    parser.add_argument("--omegaprime", metavar="FILE", help="Path to OmegaPrime MCAP file")
    parser.add_argument("--expected-hz", type=float, default=None,
                        help="Expected sampling frequency in Hz (temporal metrics)")
    parser.add_argument("--stride", type=int, default=5,
                        help="Frame sample stride for video metrics (default: 5)")
    parser.add_argument("--max-frames", type=int, default=_DEFAULT_MAX_FRAMES,
                        help=f"Max frames to sample for video metrics (default: {_DEFAULT_MAX_FRAMES})")
    parser.add_argument("--dataset-spec", metavar="FILE",
                        help="JSON file defining expected/required object classes for D3.1 metrics")
    parser.add_argument("--json", dest="json_out", metavar="FILE",
                        help="Write JSON output to FILE")
    parser.add_argument("--no-role-check", action="store_true",
                        help="Exclude role from class completeness Case 2 diversity check")
    args = parser.parse_args()

    if not args.video and not args.omegaprime:
        parser.error("At least one of --video or --omegaprime is required")

    # Load dataset spec and resolve string enum names to OSI integers
    required_types = expected_types = expected_subtypes = expected_roles = None
    if args.dataset_spec:
        with open(args.dataset_spec) as f:
            spec = json.load(f)
        _name_to_int = {
            "type":    {v: k for k, v in _OSI_MOVING_OBJECT_TYPE.items()},
            "subtype": {v: k for k, v in _OSI_VEHICLE_SUBTYPE.items()},
            "role":    {v: k for k, v in _OSI_VEHICLE_ROLE.items()},
        }

        def _resolve(key: str, lookup_key: str) -> Optional[List[int]]:
            values = spec.get(key)
            if not values:
                return None
            lookup = _name_to_int[lookup_key]
            result = []
            for v in values:
                if isinstance(v, int):
                    result.append(v)
                elif isinstance(v, str):
                    if v not in lookup:
                        parser.error(
                            f"dataset-spec '{key}': unknown name {v!r}. "
                            f"Valid names: {sorted(lookup)}"
                        )
                    result.append(lookup[v])
                else:
                    parser.error(f"dataset-spec '{key}': expected str or int, got {v!r}")
            return result or None

        required_types   = _resolve("required_types",   "type")
        expected_types   = _resolve("expected_types",   "type")
        expected_subtypes = _resolve("expected_subtypes", "subtype")
        expected_roles   = _resolve("expected_roles",   "role")

    result = {}

    if args.video:
        print(f"Computing video metrics: {args.video}")
        result["video"] = run_video_metrics(
            args.video,
            expected_frequency=args.expected_hz,
            sample_stride=args.stride,
            max_frames=args.max_frames,
        )

    if args.omegaprime:
        print(f"Computing OmegaPrime metrics: {args.omegaprime}")
        result["omegaprime"] = run_omegaprime_metrics(
            args.omegaprime,
            expected_hz=args.expected_hz,
            required_types=required_types,
            expected_types=expected_types,
            expected_subtypes=expected_subtypes,
            expected_roles=expected_roles,
            check_role=not args.no_role_check,
        )

    out = json.dumps(result, indent=2, default=str)
    print(out)

    if args.json_out:
        Path(args.json_out).write_text(out)
        print(f"\nJSON written to {args.json_out}", flush=True)


if __name__ == "__main__":
    main()

