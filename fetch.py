import csv
import json
import re
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import requests

BASE = "https://colorimetercorrections.displaycal.net/"

# region Config

# Time slices to enumerate, avoiding overly large responses; extend years as needed
CREATED_SLICES = [
    "2025-*",
    "2024-*",
    "2023-*",
    "2022-*",
    "2021-*",
    "2020-*",
    "2019-*",
    "2018-*",
    "2017-*",
    "2016-*",
    "2015-*",
    "2014-*",
    "2013-*",
    "2012-*",
]

VISITED_PATH = Path("visited_hashes.json")
OUT_CSV = Path("all_raw_spectra.csv")

# Write buffering: flush after N new hashes to reduce memory peak
FLUSH_EVERY_HASHES = 100

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ccss-spd-dumper/1.0"})


# endregion


# region Visited


def load_visited() -> set:
    if VISITED_PATH.exists():
        return set(json.loads(VISITED_PATH.read_text("utf-8")))
    return set()


def save_visited(visited: set) -> None:
    VISITED_PATH.write_text(json.dumps(sorted(visited)), "utf-8")


# endregion


# region HTTP


def fetch_with_retry(params: Dict[str, Any], max_retries: int = 3) -> Any:
    """
    Request with retry mechanism (exponential backoff)
    """
    for attempt in range(max_retries):
        try:
            r = SESSION.get(BASE, params=params, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2**attempt  # 1s, 2s, 4s
                print(f"  ⚠️ Retry {attempt + 1}/{max_retries} in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def fetch_ccss_entries(created_pattern: str) -> List[Dict[str, Any]]:
    """
    Fetch CCSS entries within a time slice (JSON list)
    """
    params = {
        "get": "1",
        "type": "ccss",  # Only spectral data: CCSS is sufficient
        "display": "*",
        "created": created_pattern,
        "json": "1",
    }
    data = fetch_with_retry(params)
    if isinstance(data, list):
        return data
    # Handle possible wrapper structures
    for k in ("result", "results", "entries", "data"):
        if isinstance(data, dict) and isinstance(data.get(k), list):
            return data[k]
    raise ValueError(f"Unexpected JSON shape: {type(data)}")


# endregion


# region Parse


def parse_spec_from_cgats(cgats_text: str) -> Tuple[List[float], List[List[float]]]:
    """
    Parse spectral data from CGATS (CCSS):
      - Extract SPEC_* columns and their order from BEGIN_DATA_FORMAT
      - Read values for each row from BEGIN_DATA
    Returns: wavelengths_nm, spectra_rows
      spectra_rows: one spectrum per sample (list[float])
    """
    t = cgats_text.replace("\r\n", "\n").replace("\r", "\n")

    m_fmt = re.search(r"BEGIN_DATA_FORMAT\s+(.+?)\s+END_DATA_FORMAT", t, re.S)
    if not m_fmt:
        raise ValueError("Missing DATA_FORMAT block")
    cols = m_fmt.group(1).split()

    spec_cols = [c for c in cols if c.startswith("SPEC_")]
    if not spec_cols:
        raise ValueError("No SPEC_ columns")

    wavelengths = [float(c.split("_", 1)[1]) for c in spec_cols]
    first_spec_idx = cols.index(spec_cols[0])
    spec_count = len(spec_cols)

    m_data = re.search(r"BEGIN_DATA\s+(.+?)\s+END_DATA", t, re.S)
    if not m_data:
        raise ValueError("Missing DATA block")

    spectra = []
    for ln in m_data.group(1).splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < first_spec_idx + spec_count:
            raise ValueError("Row shorter than expected")
        spec_vals = list(
            map(float, parts[first_spec_idx : first_spec_idx + spec_count])
        )
        spectra.append(spec_vals)

    return wavelengths, spectra


# endregion


# region CSV


def ensure_csv_header():
    if OUT_CSV.exists():
        return
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Long format: one row per (hash, sample_index, wavelength_nm, value)
        w.writerow(
            [
                "hash",
                "created",
                "manufacturer",
                "display",
                "instrument",
                "reference",
                "sample_index",
                "wavelength_nm",
                "value",
            ]
        )


def build_spectra_rows(
    meta: Dict[str, Any], wavelengths: List[float], spectra: List[List[float]]
) -> List[List[Any]]:
    """Build rows to write (no IO), return row list"""
    rows = []
    h = meta.get("hash")
    created = meta.get("created")
    manufacturer = meta.get("manufacturer")
    display = meta.get("display")
    instrument = meta.get("instrument")
    reference = meta.get("reference")

    for si, spec in enumerate(spectra, start=1):
        for wl, val in zip(wavelengths, spec):
            rows.append(
                [h, created, manufacturer, display, instrument, reference, si, wl, val]
            )
    return rows


# endregion


# region Main


def main():
    ensure_csv_header()
    visited = load_visited()
    print(f"Already visited: {len(visited)} entries")

    total_new = 0
    total_skipped = 0

    for slice_idx, sl in enumerate(CREATED_SLICES, 1):
        print(f"\n[{slice_idx}/{len(CREATED_SLICES)}] Processing slice: {sl}")
        try:
            entries = fetch_ccss_entries(sl)
            print(f"  Found {len(entries)} entries")
        except Exception as e:
            print(f"  ❌ Fetch failed: {e}")
            continue

        # Polite delay after request (not during parsing)
        time.sleep(1.0)

        new_in_slice = 0
        parse_errors = 0
        pending_rows = []  # Buffer rows and flush periodically
        new_hashes_since_flush = 0

        # Open once per slice; flush buffered rows periodically
        with OUT_CSV.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)

            for ent in entries:
                h = ent.get("hash")
                if not h or h in visited:
                    total_skipped += 1
                    continue

                cgats = ent.get("cgats")
                if not isinstance(cgats, str) or not cgats.strip():
                    visited.add(h)
                    total_skipped += 1
                    continue

                try:
                    wavelengths, spectra = parse_spec_from_cgats(cgats)
                    pending_rows.extend(build_spectra_rows(ent, wavelengths, spectra))
                    new_in_slice += 1
                    total_new += 1
                    new_hashes_since_flush += 1

                    visited.add(h)

                    if new_hashes_since_flush >= FLUSH_EVERY_HASHES:
                        if pending_rows:
                            w.writerows(pending_rows)
                            pending_rows.clear()
                        new_hashes_since_flush = 0
                except Exception as e:
                    parse_errors += 1
                    if parse_errors <= 3:  # Only print first few errors
                        print(f"  ❌ Parse failed for {h[:8]}...: {e}")

                    # Mark parse failures as visited to avoid retry loops.
                    visited.add(h)

            if pending_rows:
                w.writerows(pending_rows)
                pending_rows.clear()

        # Save visited after each slice
        save_visited(visited)

        # Summary output (reduce logging)
        print(
            f"  ✓ New: {new_in_slice} | Skipped: {total_skipped} | Errors: {parse_errors}"
        )

    print(f"\n{'=' * 50}")
    print(f"✅ Complete! Total new entries: {total_new}")
    print(f"   Skipped (already visited): {total_skipped}")
    print(f"   Output: {OUT_CSV}")


# endregion


if __name__ == "__main__":
    main()
