import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
ALL_CLASSIFIED = ROOT / "all_classified.csv"
ALL_RAW = ROOT / "all_raw_spectra.csv"

OUT_DIR = ROOT / "tests"
OUT_CASES = OUT_DIR / "test_cases.csv"
OUT_RAW = OUT_DIR / "test_raw_spectra.csv"


@dataclass(frozen=True)
class SeedCase:
    display: str
    manufacturer_hint: str
    created_date: str  # YYYY-MM-DD
    instrument_hint: str
    expected: str  # ksf | non_ksf


SEED_CASES: list[SeedCase] = [
    SeedCase(
        display="LQ123Z1JX3X",
        manufacturer_hint="Sharp",
        created_date="2019-03-09",
        instrument_hint="i1 Pro 2",
        expected="ksf",
    ),
    SeedCase(
        display="LP129WT212166",
        manufacturer_hint="LG Display",
        created_date="2024-12-05",
        instrument_hint="i1 Pro 2",
        expected="ksf",
    ),
    SeedCase(
        display="LP123WQ112604",
        manufacturer_hint="LG Display",
        created_date="2020-06-30",
        instrument_hint="ColorMunki",
        expected="non_ksf",
    ),
    SeedCase(
        display="DELL U2715H",
        manufacturer_hint="Dell",
        created_date="2025-02-24",
        instrument_hint="ColorMunki",
        expected="non_ksf",
    ),
    SeedCase(
        display="B133QAN02.0",
        manufacturer_hint="AUO",
        created_date="2022-07-19",
        instrument_hint="i1 Pro 2",
        expected="ksf",
    ),
    SeedCase(
        display="VVX14P048M00",
        manufacturer_hint="Panasonic",
        created_date="2021-11-11",
        instrument_hint="ColorMunki",
        expected="ksf",
    ),
    SeedCase(
        display="LQ156D1JW42",
        manufacturer_hint="Sharp",
        created_date="2024-04-13",
        instrument_hint="ColorMunki",
        expected="ksf",
    ),
    SeedCase(
        display="B160QAN02.K",
        manufacturer_hint="AUO",
        created_date="2022-03-31",
        instrument_hint="ColorMunki",
        expected="non_ksf",
    ),
    SeedCase(
        display="B140QAN02.3",
        manufacturer_hint="AUO",
        created_date="2020-07-08",
        instrument_hint="i1 Pro 2",
        expected="non_ksf",
    ),
]


def norm(s: str) -> str:
    return (s or "").strip().lower()


def pick_candidate(rows: list[dict], seed: SeedCase) -> dict:
    # Prefer: exact display match + exact date prefix + instrument contains hint + no error
    date_prefix = seed.created_date
    inst_hint = norm(seed.instrument_hint)

    def score(r: dict) -> tuple:
        display_exact = (r.get("display") or "") == seed.display
        date_ok = (r.get("created") or "").startswith(date_prefix)
        inst_ok = inst_hint in norm(r.get("instrument") or "") if inst_hint else True
        no_error = not (r.get("error") or "").strip()
        # Higher is better; tuple sorts ascending, so invert booleans
        return (
            0 if display_exact else 1,
            0 if date_ok else 1,
            0 if inst_ok else 1,
            0 if no_error else 1,
        )

    rows_sorted = sorted(rows, key=score)
    return rows_sorted[0]


def find_match_in_classified(seed: SeedCase) -> dict:
    if not ALL_CLASSIFIED.exists():
        raise FileNotFoundError(f"Missing {ALL_CLASSIFIED}")

    candidates: list[dict] = []
    display_want = seed.display
    m_hint = norm(seed.manufacturer_hint)

    with ALL_CLASSIFIED.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            display = row.get("display") or ""
            if display != display_want:
                continue

            manu = norm(row.get("manufacturer") or "")
            if m_hint and m_hint not in manu:
                # allow empty manufacturer_hint match for CCSS oddities
                continue

            created = row.get("created") or ""
            if seed.created_date and not created.startswith(seed.created_date):
                continue

            candidates.append(row)

    # If strict matching fails, relax manufacturer/date constraints stepwise
    if not candidates:
        with ALL_CLASSIFIED.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("display") or "") != display_want:
                    continue
                if seed.created_date and not (row.get("created") or "").startswith(
                    seed.created_date
                ):
                    continue
                candidates.append(row)

    if not candidates:
        with ALL_CLASSIFIED.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("display") or "") != display_want:
                    continue
                candidates.append(row)

    if not candidates:
        raise ValueError(
            f"No match found in all_classified.csv for display={seed.display!r}"
        )

    return pick_candidate(candidates, seed)


def collect_hashes(rows: Iterable[dict]) -> set[str]:
    hashes = set()
    for r in rows:
        h = (r.get("hash") or "").strip()
        if h:
            hashes.add(h)
    return hashes


def export_raw_subset(hashes: set[str]) -> None:
    if not ALL_RAW.exists():
        raise FileNotFoundError(f"Missing {ALL_RAW}")

    with (
        ALL_RAW.open("r", newline="", encoding="utf-8") as src,
        OUT_RAW.open("w", newline="", encoding="utf-8") as dst,
    ):
        reader = csv.DictReader(src)
        if not reader.fieldnames:
            raise ValueError("all_raw_spectra.csv has no header")

        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            if (row.get("hash") or "").strip() in hashes:
                writer.writerow(row)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    resolved_rows: list[dict] = []

    for idx, seed in enumerate(SEED_CASES, 1):
        r = find_match_in_classified(seed)
        r = dict(r)
        r["case_id"] = str(idx)
        r["expected"] = seed.expected
        resolved_rows.append(r)

    # Write test_cases.csv
    # Minimal, stable contract for regression:
    # - Identify the raw fixture by hash
    # - Assert only the expected binary label
    fieldnames = ["case_id", "hash", "expected"]

    with OUT_CASES.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in resolved_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    export_raw_subset(collect_hashes(resolved_rows))

    print(f"Wrote: {OUT_CASES}")
    print(f"Wrote: {OUT_RAW}")


if __name__ == "__main__":
    main()
