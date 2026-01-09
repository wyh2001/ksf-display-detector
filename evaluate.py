import csv
import sys
import io
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict
from collections import defaultdict

import numpy as np

# Windows console UTF-8 support
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# region Config
INPUT_CSV = Path("all_raw_spectra.csv")  # Output from fetch.py
OUT_CSV = Path("all_classified.csv")
RUN_META_JSON = Path("all_classified.run_meta.json")

OUT_HEADER = [
    "hash",
    "display",
    "manufacturer",
    "instrument",
    "reference",
    "created",
    "year",
    "peak_nm",
    "fwhm_nm",
    "sharpness",
    "confidence",
    "likely_ksf",
    "maybe_wcg",
    "reason",
    "error",
]

# endregion

# region Thresholds
# KSF red peak is very stable, typically at 630-632nm
# Normal Wide Gamut LED (e.g., AdobeRGB) red peak is at 620-625nm and wider
PEAK_RANGE = (628.0, 638.0)  # KSF/PFS narrow red peak range (redder than normal WCG)
FWHM_MAX = 35.0  # FWHM threshold (KSF typical 20-30nm, normal LED 50-150nm)
SHARPNESS_MIN = 3.0  # Peak sharpness threshold (peak / shoulder mean)
SHARPNESS_HIGH = 7.0  # High sharpness threshold (can relax other conditions)

# Strict mode (enable if too many false positives)
STRICT_MODE = False
STRICT_FWHM_MAX = 25.0  # FWHM in strict mode

# endregion

# region Bands
# Feature extraction
MAIN_PEAK_BAND = (610.0, 670.0)
SHOULDER_BANDS = [(600.0, 610.0), (660.0, 700.0)]

# Red-channel selection
KSF_PICK_BAND = (625.0, 640.0)
KSF_PICK_SHOULDER_BANDS = [(600.0, 620.0), (650.0, 680.0)]
RED_ENERGY_BAND = (600.0, 700.0)

# Secondary flag: likely normal wide-gamut (non-KSF) LED
WCG_PEAK_MIN = 620.0
WCG_FWHM_MIN = 30.0
# endregion


# region Utils


def write_out_header(f) -> None:
    csv.writer(f).writerow(OUT_HEADER)


def read_existing_header() -> list[str] | None:
    if not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0:
        return None
    with OUT_CSV.open("r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        try:
            return next(r)
        except StopIteration:
            return None


def year_from_created(created: str) -> str:
    s = str(created or "")
    if len(s) >= 4 and s[:4].isdigit():
        return s[:4]
    return ""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_run_meta(meta: dict) -> None:
    RUN_META_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")


def load_processed_hashes() -> set:
    if not OUT_CSV.exists():
        return set()
    processed = set()
    with OUT_CSV.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            h = row.get("hash")
            if h:
                processed.add(h)
    return processed


# endregion


# region Rules
def classify_features(
    feat: Dict[str, float],
    *,
    peak_range: tuple[float, float],
    fwhm_limit: float,
    sharpness_min: float,
    sharpness_high: float,
) -> tuple[bool, bool, str]:
    """Pure classification logic.

    Returns: (likely_ksf, maybe_wcg, reason)
    """
    peak_nm = float(feat["peak_nm"])
    fwhm_nm = float(feat["fwhm_nm"])
    sharpness = float(feat["sharpness"])

    peak_ok = peak_range[0] <= peak_nm <= peak_range[1]
    fwhm_ok = fwhm_nm <= fwhm_limit
    sharp_ok = sharpness >= sharpness_min
    sharp_high_ok = sharpness >= sharpness_high

    # Keep behavior consistent with prior implementation:
    # require sharp_ok even when using sharp_high override.
    likely = bool(peak_ok and sharp_ok and (fwhm_ok or sharp_high_ok))

    maybe_wcg = bool(
        (WCG_PEAK_MIN <= peak_nm < peak_range[0]) and (fwhm_nm > WCG_FWHM_MIN)
    )

    reason = ""
    if likely:
        if peak_ok and sharp_ok and fwhm_ok:
            reason = "peak_ok+fwhm_ok+sharp_ok"
        elif peak_ok and sharp_high_ok:
            reason = "peak_ok+sharp_high"
        else:
            reason = "likely_ksf"
    elif maybe_wcg:
        reason = "maybe_wcg"

    return likely, maybe_wcg, reason


# endregion


# region Iterate
def iter_hash_entries(csv_path: Path, *, require_grouped: bool = True):
    """Stream entries from the long-format spectra CSV.

    By default, requires rows to be grouped by hash (i.e., all rows for the same hash are
    contiguous). This is typically true for fetch.py output.

    If the input CSV has been re-sorted/merged and hashes are no longer contiguous, the
    same hash can appear in multiple chunks, leading to incomplete spectra and incorrect
    classification. When require_grouped=True, this function detects such "hash backtrack"
    and raises a ValueError with a helpful message.

    Yields: (hash, meta_dict, samples_dict)
    """
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        current_hash = None
        meta = None
        samples = defaultdict(dict)
        seen_hashes = set()

        for row in reader:
            h = row.get("hash")
            if not h:
                continue

            if (
                require_grouped
                and current_hash is not None
                and h != current_hash
                and h in seen_hashes
            ):
                raise ValueError(
                    "Input CSV is not grouped by hash: saw a previously-seen hash appear again after other hashes. "
                    "This often happens if a previous run was interrupted and a later run appended new rows, causing the same hash to appear in multiple blocks. "
                    "Please sort/group the file by 'hash' (and typically 'sample_index', 'wavelength_nm') before running, "
                    "or re-run with --no-require-grouped if you understand the risk."
                )

            if current_hash is None:
                current_hash = h
                meta = None
                samples = defaultdict(dict)
            elif h != current_hash:
                yield current_hash, (meta or {}), dict(samples)
                seen_hashes.add(current_hash)
                current_hash = h
                meta = None
                samples = defaultdict(dict)

            if meta is None:
                meta = {
                    "display": row.get("display", ""),
                    "manufacturer": row.get("manufacturer", ""),
                    "instrument": row.get("instrument", ""),
                    "reference": row.get("reference", ""),
                    "created": row.get("created", ""),
                }

            try:
                sample_idx = int(row["sample_index"])
                wl = float(row["wavelength_nm"])
                val = float(row["value"])
            except Exception:
                continue

            samples[sample_idx][wl] = val

        if current_hash is not None:
            yield current_hash, (meta or {}), dict(samples)


# endregion


# region Load
def extract_spectrum_arrays(samples: Dict[int, Dict[float, float]]):
    """
    Convert samples dict to numpy arrays
    Returns: (wavelengths, [spectrum1, spectrum2, ...])
    """
    if not samples:
        raise ValueError("No samples")

    # Get wavelength list from first sample
    first_sample = samples[min(samples.keys())]
    wl = np.array(sorted(first_sample.keys()), dtype=float)

    spectra = []
    for si in sorted(samples.keys()):
        spd = np.array([samples[si][w] for w in wl], dtype=float)
        spectra.append(spd)

    return wl, spectra


# endregion


# region Math
def smooth(y: np.ndarray, win: int = 5) -> np.ndarray:
    # Simple moving average
    if win <= 1:
        return y
    k = np.ones(win) / win
    return np.convolve(y, k, mode="same")


def fwhm(x: np.ndarray, y: np.ndarray, peak_i: int) -> float:
    peak = y[peak_i]
    if peak <= 0:
        return 999.0
    half = peak * 0.5

    # Find left half-height crossing point
    li = None
    for i in range(peak_i, 0, -1):
        if (y[i] - half) * (y[i - 1] - half) <= 0:
            li = (i - 1, i)
            break
    # Find right half-height crossing point
    ri = None
    for i in range(peak_i, len(y) - 1):
        if (y[i] - half) * (y[i + 1] - half) <= 0:
            ri = (i, i + 1)
            break
    if li is None or ri is None:
        return 999.0

    def interp_x(i0, i1):
        x0, x1 = x[i0], x[i1]
        y0, y1 = y[i0], y[i1]
        if y1 == y0:
            return x0
        return x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    xl = interp_x(*li)
    xr = interp_x(*ri)
    return max(0.0, xr - xl)


# endregion


# region Features
def analyze_one_spectrum(wl: np.ndarray, spd: np.ndarray) -> Dict[str, float]:
    # Handle negative values: clamp to 0 (CCSS/measurements sometimes have slight negative noise)
    y = np.maximum(spd, 0.0)

    # Normalize
    m = float(np.max(y))
    if m <= 0:
        return {
            "peak_nm": np.nan,
            "fwhm_nm": 999.0,
            "sharpness": 0.0,
            "confidence": 0.0,
        }
    y = y / m

    # Adaptive smoothing window based on data resolution
    # High resolution (>80 points) uses win=5, low resolution uses win=3 or no smoothing
    n_points = len(wl)
    if n_points > 80:
        win = 5
    elif n_points > 50:
        win = 3
    else:
        win = 1  # Low resolution data: no smoothing to preserve narrow peaks
    y = smooth(y, win=win)

    # Find main peak in MAIN_PEAK_BAND range
    band = np.where((wl >= MAIN_PEAK_BAND[0]) & (wl <= MAIN_PEAK_BAND[1]))[0]
    if band.size == 0:
        return {
            "peak_nm": np.nan,
            "fwhm_nm": 999.0,
            "sharpness": 0.0,
            "confidence": 0.0,
        }

    peak_i = band[np.argmax(y[band])]
    peak_nm = float(wl[peak_i])
    width = float(fwhm(wl, y, peak_i))

    # Shoulder: mean of SHOULDER_BANDS regions
    shoulder_mask = np.zeros_like(wl, dtype=bool)
    for lo, hi in SHOULDER_BANDS:
        shoulder_mask |= (wl >= lo) & (wl <= hi)
    shoulder_idx = np.where(shoulder_mask)[0]
    shoulder = float(np.mean(y[shoulder_idx])) if shoulder_idx.size else 1e-6
    sharp = float(y[peak_i] / max(shoulder, 1e-6))

    # Confidence scoring (0~1): softening the three conditions into continuous scores
    # Peak score: 1 if within target range, decreasing with distance
    lo, hi = PEAK_RANGE
    if peak_nm < lo:
        peak_score = max(0.0, 1.0 - (lo - peak_nm) / 15.0)
    elif peak_nm > hi:
        peak_score = max(0.0, 1.0 - (peak_nm - hi) / 15.0)
    else:
        peak_score = 1.0

    # Width score: higher for narrower peaks
    width_score = max(0.0, 1.0 - (width - FWHM_MAX) / 20.0) if width > FWHM_MAX else 1.0

    # Sharpness score: higher for sharper peaks
    sharp_score = max(
        0.0, min(1.0, (sharp - 1.0) / 3.0)
    )  # sharp~1 is flat, >=4 is sharp

    confidence = float(
        np.clip(0.45 * peak_score + 0.35 * width_score + 0.20 * sharp_score, 0.0, 1.0)
    )

    return {
        "peak_nm": peak_nm,
        "fwhm_nm": width,
        "sharpness": sharp,
        "confidence": confidence,
    }


# endregion


# region Channel
def pick_reddest_spectrum(wl: np.ndarray, spectra: list) -> np.ndarray:
    """
    Select the spectrum most likely to be the red channel.
    Strategy: Prioritize spectra with highest peak in 625-640nm (KSF characteristic region)
    and sharpest peak shape. This is more accurate than simple "red region energy sum"
    because white spectra often have more total red region energy.
    """
    ksf_band = np.where((wl >= KSF_PICK_BAND[0]) & (wl <= KSF_PICK_BAND[1]))[0]
    shoulder_mask = np.zeros_like(wl, dtype=bool)
    for lo, hi in KSF_PICK_SHOULDER_BANDS:
        shoulder_mask |= (wl >= lo) & (wl <= hi)
    shoulder_band = np.where(shoulder_mask)[0]

    best = None
    best_score = -1.0

    for spd in spectra:
        y = np.maximum(spd, 0.0)
        if np.max(y) <= 0:
            continue
        y_norm = y / np.max(y)

        # KSF region peak
        ksf_peak = float(np.max(y_norm[ksf_band])) if ksf_band.size else 0.0
        # Shoulder mean
        shoulder_mean = (
            float(np.mean(y_norm[shoulder_band])) if shoulder_band.size else 1.0
        )
        # Sharpness score: KSF region peak / shoulder mean
        sharpness = ksf_peak / max(shoulder_mean, 0.01)

        # Combined score: high peak + high sharpness = more likely red channel KSF spectrum
        score = ksf_peak * sharpness

        if score > best_score:
            best_score = score
            best = spd

    # If no good match found, fall back to selecting highest red region energy
    if best is None:
        red = np.where((wl >= RED_ENERGY_BAND[0]) & (wl <= RED_ENERGY_BAND[1]))[0]
        best_e = -1.0
        for spd in spectra:
            y = np.maximum(spd, 0.0)
            e = float(np.sum(y[red])) if red.size else float(np.sum(y))
            if e > best_e:
                best_e = e
                best = spd

    return best if best is not None else spectra[0]


# endregion


# region Main
def main():
    parser = argparse.ArgumentParser(description="Classify displays as KSF/non-KSF")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output instead of overwriting (skips already-processed hashes)",
    )
    parser.add_argument(
        "--require-grouped",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Require input CSV to be grouped by hash (default: enabled). Disable with --no-require-grouped.",
    )
    parser.add_argument(
        "--peak-min",
        type=float,
        default=PEAK_RANGE[0],
        help=f"Lower bound for KSF peak detection in nm (default: {PEAK_RANGE[0]}).",
    )
    parser.add_argument(
        "--peak-max",
        type=float,
        default=PEAK_RANGE[1],
        help=f"Upper bound for KSF peak detection in nm (default: {PEAK_RANGE[1]}).",
    )
    parser.add_argument(
        "--fwhm-max",
        type=float,
        default=FWHM_MAX,
        help=f"Max allowed FWHM in nm (default: {FWHM_MAX}).",
    )
    parser.add_argument(
        "--sharpness-min",
        type=float,
        default=SHARPNESS_MIN,
        help=f"Min allowed sharpness (default: {SHARPNESS_MIN}).",
    )
    parser.add_argument(
        "--sharpness-high",
        type=float,
        default=SHARPNESS_HIGH,
        help=f"High sharpness override threshold (default: {SHARPNESS_HIGH}).",
    )
    parser.add_argument(
        "--strict-mode",
        default=STRICT_MODE,
        action=argparse.BooleanOptionalAction,
        help="Enable strict mode (tightens thresholds). Disable with --no-strict-mode.",
    )
    parser.add_argument(
        "--strict-fwhm-max",
        type=float,
        default=STRICT_FWHM_MAX,
        help=f"Max allowed FWHM in strict mode (default: {STRICT_FWHM_MAX}).",
    )
    args = parser.parse_args()

    peak_range = (float(args.peak_min), float(args.peak_max))
    if peak_range[0] >= peak_range[1]:
        raise ValueError("Invalid peak range: --peak-min must be < --peak-max")

    fwhm_limit = args.strict_fwhm_max if args.strict_mode else args.fwhm_max

    if not INPUT_CSV.exists():
        print(f"❌ Error: {INPUT_CSV} not found. Please run fetch.py first.")
        return

    run_started = utc_now_iso()

    processed_hashes = set()
    out_mode = "a" if args.append else "w"
    if args.append:
        processed_hashes = load_processed_hashes()

        existing_header = read_existing_header()
        if existing_header and existing_header != OUT_HEADER:
            raise ValueError(
                "Existing output CSV header is incompatible with the current schema. "
                "Please re-run without --append to regenerate all_classified.csv (overwrite), "
                "or delete the file and run again."
            )

    # Open output once
    with OUT_CSV.open(out_mode, newline="", encoding="utf-8") as out_f:
        if out_mode == "w" or OUT_CSV.stat().st_size == 0:
            write_out_header(out_f)

        print(f"Loading data from {INPUT_CSV}...")
        if processed_hashes:
            print(
                f"Append mode: skipping {len(processed_hashes)} already-processed hashes"
            )

        count_ksf = 0
        count_total = 0
        writer = csv.writer(out_f)

        for h, meta, samples in iter_hash_entries(
            INPUT_CSV, require_grouped=args.require_grouped
        ):
            if h in processed_hashes:
                continue

            try:
                wl, spectra = extract_spectrum_arrays(samples)

                spd = pick_reddest_spectrum(wl, spectra)
                feat = analyze_one_spectrum(wl, spd)

                likely, maybe_wcg, reason = classify_features(
                    feat,
                    peak_range=peak_range,
                    fwhm_limit=float(fwhm_limit),
                    sharpness_min=float(args.sharpness_min),
                    sharpness_high=float(args.sharpness_high),
                )

                if likely:
                    count_ksf += 1
                    display_name = str(meta.get("display", "Unknown"))[:40]
                    print(
                        f"✓ KSF: {display_name:<40} | Peak: {feat['peak_nm']:.1f}nm | FWHM: {feat['fwhm_nm']:.1f}nm"
                    )
                elif maybe_wcg:
                    display_name = str(meta.get("display", "Unknown"))[:40]
                    print(
                        f"? WCG: {display_name:<40} | Peak: {feat['peak_nm']:.1f}nm | FWHM: {feat['fwhm_nm']:.1f}nm"
                    )

                created = meta.get("created", "")
                year = year_from_created(created)

                writer.writerow(
                    [
                        h,
                        meta.get("display", ""),
                        meta.get("manufacturer", ""),
                        meta.get("instrument", ""),
                        meta.get("reference", ""),
                        created,
                        year,
                        f"{feat['peak_nm']:.2f}",
                        f"{feat['fwhm_nm']:.2f}",
                        f"{feat['sharpness']:.2f}",
                        f"{feat['confidence']:.2f}",
                        int(likely),
                        int(maybe_wcg),
                        reason,
                        "",
                    ]
                )
                count_total += 1

            except Exception as e:
                print(f"❌ Error processing {h[:8]}...: {e}")
                created = meta.get("created", "") if isinstance(meta, dict) else ""
                year = year_from_created(created)
                writer.writerow(
                    [
                        h,
                        meta.get("display", ""),
                        meta.get("manufacturer", ""),
                        meta.get("instrument", ""),
                        meta.get("reference", ""),
                        created,
                        year,
                        "",
                        "",
                        "",
                        "",
                        0,
                        0,
                        "error",
                        f"{e}",
                    ]
                )

        print(f"\n{'=' * 60}")
        print("✅ Analysis complete!")
        print(f"   Total displays analyzed: {count_total}")
        print(
            f"   Likely KSF/PFS displays: {count_ksf} ({100 * count_ksf / max(count_total, 1):.1f}%)"
        )
        print(f"   Results saved to: {OUT_CSV}")

    run_finished = utc_now_iso()
    write_run_meta(
        {
            "started_at": run_started,
            "finished_at": run_finished,
            "input_csv": str(INPUT_CSV),
            "output_csv": str(OUT_CSV),
            "require_grouped": bool(args.require_grouped),
            "append": bool(args.append),
            "thresholds": {
                "peak_range_nm": [peak_range[0], peak_range[1]],
                "fwhm_max_nm": float(args.fwhm_max),
                "sharpness_min": float(args.sharpness_min),
                "sharpness_high": float(args.sharpness_high),
                "strict_mode": bool(args.strict_mode),
                "strict_fwhm_max_nm": float(args.strict_fwhm_max),
            },
            "stats": {
                "hashes_analyzed": int(count_total),
                "likely_ksf": int(count_ksf),
            },
        }
    )


# endregion


if __name__ == "__main__":
    main()
