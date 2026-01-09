import csv
import sys
from pathlib import Path
import evaluate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


CASES = ROOT / "tests" / "test_cases.csv"
RAW = ROOT / "tests" / "test_raw_spectra.csv"


def to_bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y")


def classify_hashes_from_raw(raw_csv: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}

    peak_range = evaluate.PEAK_RANGE
    fwhm_max = evaluate.FWHM_MAX
    sharp_min = evaluate.SHARPNESS_MIN
    sharp_high = evaluate.SHARPNESS_HIGH
    strict_mode = evaluate.STRICT_MODE
    strict_fwhm_max = evaluate.STRICT_FWHM_MAX

    for h, meta, samples in evaluate.iter_hash_entries(raw_csv, require_grouped=True):
        try:
            wl, spectra = evaluate.extract_spectrum_arrays(samples)
            spd = evaluate.pick_reddest_spectrum(wl, spectra)
            feat = evaluate.analyze_one_spectrum(wl, spd)

            peak_ok = peak_range[0] <= feat["peak_nm"] <= peak_range[1]
            fwhm_limit = strict_fwhm_max if strict_mode else fwhm_max
            fwhm_ok = feat["fwhm_nm"] <= fwhm_limit
            sharp_ok = feat["sharpness"] >= sharp_min
            sharp_high_ok = feat["sharpness"] >= sharp_high

            likely_ksf = bool(peak_ok and ((fwhm_ok and sharp_ok) or sharp_high_ok))

            results[h] = {
                "hash": h,
                **meta,
                **feat,
                "likely_ksf": likely_ksf,
            }
        except Exception as e:
            results[h] = {
                "hash": h,
                **(meta or {}),
                "error": str(e),
                "likely_ksf": False,
            }

    return results


def main() -> None:
    if not CASES.exists():
        raise FileNotFoundError(
            f"Missing {CASES}. This repo should include test fixtures. "
            "If you need to refresh fixtures, run tests/build_test_set.py (requires pipeline outputs)."
        )
    if not RAW.exists():
        raise FileNotFoundError(
            f"Missing {RAW}. This repo should include test fixtures. "
            "If you need to refresh fixtures, run tests/build_test_set.py (requires pipeline outputs)."
        )

    predicted = classify_hashes_from_raw(RAW)

    total = 0
    failed = 0

    with CASES.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            case_id = row.get("case_id")
            h = (row.get("hash") or "").strip()
            expected = (row.get("expected") or "").strip().lower()

            if not h or h not in predicted:
                print(f"[FAIL] case {case_id}: missing hash in predictions")
                failed += 1
                continue

            pred = predicted[h]
            got = bool(pred.get("likely_ksf"))

            if expected not in ("ksf", "non_ksf"):
                print(f"[FAIL] case {case_id}: non-binary expected label: {expected!r}")
                failed += 1
                continue

            want = expected == "ksf"
            if got != want:
                print(
                    f"[FAIL] case {case_id}: expected {expected} got {'ksf' if got else 'non_ksf'} | "
                    f"peak={pred.get('peak_nm')} fwhm={pred.get('fwhm_nm')}"
                )
                failed += 1
            else:
                print(
                    f"[PASS] case {case_id}: {expected} | peak={pred.get('peak_nm'):.1f} fwhm={pred.get('fwhm_nm'):.1f}"
                )

    print("-")
    print(f"Total cases: {total}")
    print(f"Failures: {failed}")

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
