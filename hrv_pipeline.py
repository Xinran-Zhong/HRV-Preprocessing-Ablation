"""
HRV preprocessing ablation pipeline.

Computes time-domain HRV metrics from PhysioNet RR intervals under a
cumulative four-level preprocessing scheme. Each level adds exactly one
processing step, so the contribution of each component can be attributed:

    L0  range filter only (naive concatenation of the retained RR intervals)
    L1  + remove R-peaks inside low-quality (Class C) annotation segments
    L2  + split into non-overlapping 5-minute windows
    L3  + moving-average artifact filter

Per-species parameters (RR bounds, moving-average threshold, pNN threshold,
window length) follow the PhysioZoo protocol. Reference medians from the
published table are included for deviation reporting.

Outputs one CSV per species with one row per ablation level, containing the
median HRV metrics, the corresponding reference values, and percentage
deviations.

Usage
-----
    python hrv_pipeline.py                 # both species
    python hrv_pipeline.py --species dog
    python hrv_pipeline.py --species human

Dependencies: numpy, wfdb
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import wfdb

# Resolve paths relative to this file so the script runs from any directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# PhysioZoo species parameters (Tables 2 and 3).
# ann_ext differs by database: MIT-BIH NSR uses .atr, the canine database
# uses manually corrected .qrs annotations.
# ma_thr is the relative-deviation cutoff for the moving-average filter.
CFG = {
    "human": dict(
        rr_min=0.28, rr_max=2.40, pnn_xx=50,
        win_sec=5 * 60, ann_ext="atr", ma_thr=0.25,
    ),
    "dog": dict(
        rr_min=0.25, rr_max=1.20, pnn_xx=25,
        win_sec=5 * 60, ann_ext="qrs", ma_thr=0.40,
    ),
}

# Human recordings excluded from the reference statistics because of
# inaccurate R-peak annotations (protocol footnote).
EXCLUDED_HUMAN = {"16773", "19093"}

# Reference medians from the published table.
PAPER = {
    "dog": dict(AVNN=482.63, SDNN=69.20, RMSSD=53.92, pNNxx=52.88),
    "human": dict(AVNN=786.53, SDNN=53.60, RMSSD=31.28, pNNxx=33.06),
}

METRICS = ["AVNN", "SDNN", "RMSSD", "pNNxx"]

LEVEL_LABELS = {
    0: "L0 range filter only",
    1: "L1 + drop low-quality segments",
    2: "L2 + 5-min non-overlapping windows",
    3: "L3 + moving-average filter",
}

# Moving-average window length in RR samples.
MA_WIN = 31


# ---------------------------------------------------------------------------
# Quality annotation
# ---------------------------------------------------------------------------
def load_quality_segments(rec_dir: str, rec_name: str):
    """Read Class-C bad-quality segments (seconds) from
    Data_Quality_Annotations_<rec>.txt.

    The source file may contain non-UTF-8 bytes, so decoding errors are
    ignored. Returns an empty list when the annotation file is absent, which
    is the case for the human database.
    """
    path = os.path.join(rec_dir, f"Data_Quality_Annotations_{rec_name}.txt")
    segs = []
    if not os.path.exists(path):
        return segs
    with open(path, errors="ignore") as fh:
        for line in fh:
            m = re.match(r"\s*([\d.]+)\s+([\d.]+)\s+([ABC])\s*$", line)
            if m:
                segs.append((float(m.group(1)), float(m.group(2)), m.group(3)))
    return segs


def drop_bad_quality_peaks(peaks_sec: np.ndarray, segs) -> np.ndarray:
    """Drop R-peaks falling inside Class-C segments. Class-A and Class-B
    segments are retained.
    """
    if not segs:
        return peaks_sec
    keep = np.ones(len(peaks_sec), dtype=bool)
    for begin, end, cls in segs:
        if cls.upper() == "C":
            keep &= ~((peaks_sec >= begin) & (peaks_sec <= end))
    return peaks_sec[keep]


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def range_filter(
    rr_sec: np.ndarray, rr_min: float, rr_max: float
) -> np.ndarray:
    """Return a boolean mask of RR intervals within [rr_min, rr_max]."""
    return (rr_sec >= rr_min) & (rr_sec <= rr_max)


def moving_average_filter(
    rr_sec: np.ndarray, species: str, win: int = MA_WIN
) -> np.ndarray:
    """Return a boolean mask (True = keep, False = artifact).

    An RR interval is flagged when its deviation from the local mean exceeds
    the species-specific threshold. The implementation follows the zero-phase
    moving-average approach of Mietus and Goldberger; edges are not cropped.

    The threshold is looked up from CFG[species]["ma_thr"] so that human and
    canine data can never be processed with each other's setting.
    """
    thr = CFG[species]["ma_thr"]
    keep = np.ones(len(rr_sec), dtype=bool)
    if len(rr_sec) == 0:
        return keep
    half = win // 2
    padded = np.pad(rr_sec, half, mode="edge")
    kernel = np.ones(win) / win
    local_mean = np.convolve(padded, kernel, mode="valid")
    local_mean = local_mean[: len(rr_sec)]
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.abs(rr_sec - local_mean) / local_mean
    keep[dev > thr] = False
    return keep


# ---------------------------------------------------------------------------
# Segmentation and metrics
# ---------------------------------------------------------------------------
def build_segments(rr_ms: np.ndarray, keep: np.ndarray):
    """Split the RR series into contiguous clean runs defined by keep.

    Rejected RR intervals must not be removed by concatenation, because that
    would create spurious adjacent pairs spanning an artifact and inflate
    short-term variability (RMSSD, pNNxx).
    """
    segs, cur = [], []
    for val, ok in zip(rr_ms, keep):
        if ok:
            cur.append(val)
        elif cur:
            segs.append(np.asarray(cur))
            cur = []
    if cur:
        segs.append(np.asarray(cur))
    return segs


def hrv_from_segments(segs, pnn_xx: float):
    """Compute time-domain metrics from clean segments.

    SDNN uses all NN intervals (sample standard deviation, ddof=1).
    RMSSD and pNNxx use only successive differences within a segment.
    Returns None when the sample is too small for a stable estimate.
    """
    nn = np.concatenate(segs) if segs else np.array([])
    dlist = [np.diff(s) for s in segs if len(s) > 1]
    diffs = np.concatenate(dlist) if dlist else np.array([])
    if len(nn) < 3 or len(diffs) < 1:
        return None
    return dict(
        AVNN=float(np.mean(nn)),
        SDNN=float(np.std(nn, ddof=1)),
        RMSSD=float(np.sqrt(np.mean(diffs ** 2))),
        pNNxx=float(np.mean(np.abs(diffs) > pnn_xx) * 100),
        n_nn=len(nn),
    )


def metrics_from_concatenated(rr_ms: np.ndarray, keep: np.ndarray, pnn_xx: float):
    """Compute metrics on the range-filtered RR intervals as one series.

    This is the naive baseline (L0): differences are taken across rejected
    intervals, which is retained deliberately so its effect can be quantified.
    """
    nn = rr_ms[keep]
    if len(nn) < 3:
        return None
    diffs = np.diff(nn)
    if len(diffs) < 1:
        return None
    return dict(
        AVNN=float(np.mean(nn)),
        SDNN=float(np.std(nn, ddof=1)),
        RMSSD=float(np.sqrt(np.mean(diffs ** 2))),
        pNNxx=float(np.mean(np.abs(diffs) > pnn_xx) * 100),
        n_nn=len(nn),
    )


# ---------------------------------------------------------------------------
# Record processing
# ---------------------------------------------------------------------------
def process_record(
    rec_dir: str, rec_name: str, species: str, level: int
):
    """Process one record end-to-end and return a list of metric dicts.

    Only the header and the R-peak annotations are read; the ECG waveform is
    not needed for time-domain HRV. For 24-hour human recordings this avoids
    loading the full ~44 MB signal file.
    """
    cfg = CFG[species]
    record = wfdb.rdheader(os.path.join(rec_dir, rec_name))
    ann = wfdb.rdann(os.path.join(rec_dir, rec_name), cfg["ann_ext"])
    fs = record.fs
    peaks_sec = ann.sample.astype(float) / fs

    if level >= 1:
        segs = load_quality_segments(rec_dir, rec_name)
        peaks_sec = drop_bad_quality_peaks(peaks_sec, segs)

    duration = record.sig_len / fs
    win_len = cfg["win_sec"]
    if level >= 2:
        n_win = int(np.floor(duration / win_len))
        windows = [(w * win_len, (w + 1) * win_len) for w in range(max(n_win, 0))]
        if not windows:
            windows = [(0.0, duration)]
    else:
        windows = [(0.0, duration)]

    out = []
    for t0, t1 in windows:
        pw = peaks_sec[(peaks_sec >= t0) & (peaks_sec < t1)]
        if len(pw) < 5:
            continue
        rr_sec = np.diff(pw)
        if len(rr_sec) < 3:
            continue
        rr_ms = rr_sec * 1000.0

        keep = range_filter(rr_sec, cfg["rr_min"], cfg["rr_max"])
        if level >= 3:
            keep &= moving_average_filter(rr_sec, species)

        if level == 0:
            metric = metrics_from_concatenated(rr_ms, keep, cfg["pnn_xx"])
        else:
            metric = hrv_from_segments(build_segments(rr_ms, keep), cfg["pnn_xx"])

        if metric is not None:
            metric["window_start_s"] = t0
            out.append(metric)
    return out


# ---------------------------------------------------------------------------
# Dataset discovery and ablation
# ---------------------------------------------------------------------------
def list_records(species: str):
    """Return a list of (record_directory, record_name) pairs."""
    species_dir = os.path.join(DATA_ROOT, species)
    if not os.path.isdir(species_dir):
        raise FileNotFoundError(f"Data directory not found: {species_dir}")
    records = []
    if species == "dog":
        for name in sorted(os.listdir(species_dir)):
            rec_dir = os.path.join(species_dir, name)
            if os.path.isdir(rec_dir) and name.lower().startswith("dog_"):
                records.append((rec_dir, name))
    else:
        for name in sorted(os.listdir(species_dir)):
            if name.endswith(".hea"):
                rec_name = name[:-4]
                if rec_name not in EXCLUDED_HUMAN:
                    records.append((species_dir, rec_name))
    return records


def run_ablation(species: str, max_records: int | None = None):
    """Run all ablation levels for one species and return output rows."""
    records = list_records(species)
    if max_records is not None:
        records = records[:max_records]
    print(f"\n{species.upper()}: {len(records)} records")

    rows = []
    for level in sorted(LEVEL_LABELS):
        per_window = []
        for rec_dir, rec_name in records:
            try:
                per_window.extend(process_record(rec_dir, rec_name, species, level))
            except Exception as exc:  # noqa: BLE001 - keep the pipeline running
                print(f"  skipped {rec_name}: {exc}")
        if not per_window:
            print(f"  L{level}: no valid windows")
            continue

        row = dict(
            species=species,
            level=level,
            description=LEVEL_LABELS[level],
            n_windows=len(per_window),
        )
        for metric in METRICS:
            value = float(np.median([m[metric] for m in per_window]))
            target = PAPER[species][metric]
            row[f"{metric}_ms" if metric != "pNNxx" else "pNNxx_pct"] = value
            row[f"{metric}_target"] = target
            row[f"{metric}_dev_pct"] = (value - target) / target * 100
        rows.append(row)

        dev = "  ".join(
            f"{m} {row[f'{m}_dev_pct']:+.1f}%" for m in METRICS
        )
        print(f"  L{level}: {len(per_window):4d} windows  |  {dev}")
    return rows


def write_csv(rows, species: str):
    """Write ablation results to results/hrv_ablation_<species>.csv."""
    if not rows:
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"hrv_ablation_{species}.csv")
    columns = [
        "species", "level", "description", "n_windows",
        "AVNN_ms", "SDNN_ms", "RMSSD_ms", "pNNxx_pct",
        "AVNN_target", "SDNN_target", "RMSSD_target", "pNNxx_target",
        "AVNN_dev_pct", "SDNN_dev_pct", "RMSSD_dev_pct", "pNNxx_dev_pct",
    ]
    import csv

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v
                             for k, v in row.items()})
    print(f"  written: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--species", choices=["dog", "human", "all"], default="all",
        help="species to process (default: all)",
    )
    parser.add_argument(
        "--max-records", type=int, default=None,
        help="limit the number of records per species (for testing)",
    )
    args = parser.parse_args()

    species_list = ["dog", "human"] if args.species == "all" else [args.species]
    for species in species_list:
        rows = run_ablation(species, max_records=args.max_records)
        write_csv(rows, species)


if __name__ == "__main__":
    main()
