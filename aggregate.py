import pandas as pd
import re
import argparse
from pathlib import Path


# region HRules
PANEL_PREFIX_RE = re.compile(
    r"^(ATNA|B\d{3}|N\d{3}|NV\d{3}|NE\d{3}|LP\d{3}|LM\d{3}|LTN\d{3}|LQ\d{3}|HB\d{3}|TV\d{3}|LEN\d{3})",
    re.IGNORECASE,
)
SIZE_CODES = [
    "101",
    "106",
    "116",
    "123",
    "125",
    "126",
    "133",
    "139",
    "140",
    "141",
    "144",
    "150",
    "154",
    "156",
    "160",
    "161",
    "170",
    "173",
    "184",
]
NOT_PANEL_PREFIX_RE = re.compile(
    r"^(DELL|ASUS|ROG|MSI|BenQ|AOC|ViewSonic|HP|LG\s|LG$|SAMSUNG|Odyssey|Philips|Eizo|NEC|Acer|Sony|TCL|Hisense|VIZIO|Apple|MacBook|iMac|Cintiq|OptiPlex|Thunderbolt|Smart\s|Mi\sTV|webOS)",
    re.IGNORECASE,
)

HIRES_KW = [
    "WUXGA",
    "WQXGA",
    "WQHD",
    "QHD",
    "UHD",
    "RETINA",
    "2880",
    "3200",
    "3840",
    "4096",
    "5120",
    "2160",
    "2400",
    "1600",
    "1440",
    "1800",
]
HIRES_KW_RE = re.compile("|".join(re.escape(k) for k in HIRES_KW), re.IGNORECASE)
HIRES_XK_RE = re.compile(r"(?<![A-Za-z0-9])([4568]K)(?![A-Za-z0-9])", re.IGNORECASE)
HIRES_CODE_TOKENS = ["QDM", "QHM", "QAN", "UAN", "QUM", "QHN", "ZAN"]
HIRES_CODE_RE = re.compile(r"(" + "|".join(HIRES_CODE_TOKENS) + r")", re.IGNORECASE)
SHARP_4K_RE = re.compile(r"^LQ\d{3}D[12]", re.IGNORECASE)


# endregion


# region HHelp
def is_laptop_panel(name: str) -> bool:
    if not isinstance(name, str):
        return False
    s = name.strip()
    if len(s) < 7 or len(s) > 30:
        return False
    if " " in s:
        return False
    if NOT_PANEL_PREFIX_RE.match(s):
        return False
    if not any(code in s for code in SIZE_CODES):
        return False
    if PANEL_PREFIX_RE.match(s) or ("-" in s) or ("." in s):
        return True
    return False


def hires_reason(name: str) -> str:
    s = (name or "").strip()
    m1 = HIRES_KW_RE.search(s)
    if m1:
        return f"keyword:{m1.group(0).upper()}"
    m_xk = HIRES_XK_RE.search(s)
    if m_xk:
        return f"keyword:{m_xk.group(1).upper()}"
    if SHARP_4K_RE.match(s):
        return "code:LQ*D1/D2 (Sharp 4K)"
    m2 = HIRES_CODE_RE.search(s)
    if m2:
        return f"code:{m2.group(0).upper()} (inferred)"
    return ""


# endregion


# region Parse
def extract_screen_size(display_name: str) -> float:
    """
    Extract screen size from display model name.
    Common laptop panel naming: B156HAN, LP140WF, NV173FHM, LQ156D1, etc.
    156 = 15.6", 140 = 14.0", 173 = 17.3", 133 = 13.3"
    """
    if pd.isna(display_name):
        return None

    name = str(display_name).upper()

    # Match size codes in panel model names: e.g., B156, LP140, NV173, LQ133
    # Format is typically letters + 3 digits, first two are inches, third is decimal
    patterns = [
        r"[A-Z]{1,3}(\d{3})[A-Z]",  # B156HAN, LP140WF, NV173FHM
        r"[A-Z]{2}(\d{3})[A-Z]",  # LQ156D1
        r"(\d{3})[A-Z]{2,}",  # 156WF6
    ]

    for p in patterns:
        m = re.search(p, name)
        if m:
            size_code = m.group(1)
            size = int(size_code[:2]) + int(size_code[2]) / 10
            if 10 <= size <= 20:  # Reasonable laptop/portable screen range
                return size

    return None


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in (
        "peak_nm",
        "fwhm_nm",
        "sharpness",
        "confidence",
        "likely_ksf",
        "maybe_wcg",
    ):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# endregion


# region Year
def ensure_year(
    df: pd.DataFrame,
    *,
    raw_spectra_path: Path = Path("all_raw_spectra.csv"),
) -> pd.DataFrame:
    """Ensure df has a usable 'year' column.

    Priority:
    1) Keep existing df['year'] when present
    2) Fill missing/empty years from df['created'] (YYYY... -> YYYY)
    3) If still entirely missing, try deriving from all_raw_spectra.csv by hash
    """
    df = df.copy()

    if "year" not in df.columns:
        df["year"] = pd.NA

    if "created" in df.columns:
        created_year = df["created"].astype(str).str[:4]
        df["year"] = df["year"].astype("string").fillna(created_year)

    # Fallback: derive year from all_raw_spectra.csv when still missing
    year_str = df["year"].astype(str)
    if df["year"].isna().all() or (year_str.str.len() == 0).all():
        if raw_spectra_path.exists():
            spectra = pd.read_csv(
                raw_spectra_path, usecols=["hash", "created"]
            ).drop_duplicates(subset="hash")
            spectra["year"] = spectra["created"].astype(str).str[:4]
            df = df.merge(
                spectra[["hash", "year"]], on="hash", how="left", suffixes=("", "_raw")
            )
            if "year_raw" in df.columns:
                df["year"] = df["year"].fillna(df["year_raw"])
                df = df.drop(columns=["year_raw"])
        else:
            df["year"] = pd.NA

    return df


# endregion


# region IO
def load_classified(path: str | Path = "all_classified.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = coerce_numeric(df)
    df = ensure_year(df)
    df["screen_size"] = df["display"].apply(extract_screen_size)
    return df


def add_vote_fields(vote_agg: pd.DataFrame) -> pd.DataFrame:
    vote_agg = vote_agg.copy()
    vote_agg["final_ksf"] = (
        vote_agg["ksf_votes"] > vote_agg["total_votes"] / 2
    ).astype(int)
    vote_agg["vote_ratio"] = vote_agg.apply(
        lambda r: f"{int(r['ksf_votes'])}/{int(r['total_votes'])}", axis=1
    )
    return vote_agg


def export_csv(
    src: pd.DataFrame,
    *,
    out_path: str,
    columns: list[str],
    rename: dict[str, str] | None = None,
    sort_by: list[str] | None = None,
    ascending: list[bool] | None = None,
) -> pd.DataFrame:
    out = src[columns].copy()
    if rename:
        out = out.rename(columns=rename)
    if sort_by:
        out = out.sort_values(sort_by, ascending=ascending)
    out.to_csv(out_path, index=False)
    return out


# endregion


# region Vote
def build_vote_agg(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["display", "manufacturer"]

    # Robust best_* selection: pick row with max sharpness in each group
    df = df.copy()
    df["_sharpness_for_best"] = pd.to_numeric(
        df.get("sharpness"), errors="coerce"
    ).fillna(-1e18)
    g = df.groupby(group_cols, sort=False)
    idx_best = g["_sharpness_for_best"].idxmax()
    best = df.loc[idx_best, group_cols + ["peak_nm", "fwhm_nm", "sharpness"]].copy()
    best = best.rename(
        columns={
            "peak_nm": "best_peak",
            "fwhm_nm": "best_fwhm",
            "sharpness": "best_sharpness",
        }
    )

    agg = g.agg(
        ksf_votes=("likely_ksf", "sum"),
        total_votes=("likely_ksf", "count"),
        year=("year", "max"),
        confidence=("confidence", "max"),
        screen_size=("screen_size", "first"),
    ).reset_index()
    vote_agg = agg.merge(best, on=group_cols, how="left")
    return vote_agg.drop(
        columns=[c for c in ("_sharpness_for_best",) if c in vote_agg.columns]
    )


# endregion


# region Run
def run_aggregate(stats: bool = False, debug: bool = False):
    df = load_classified()

    # KSF displays
    ksf = df[df["likely_ksf"] == 1]
    non_ksf = df[df["likely_ksf"] == 0]

    # ===== Majority Voting =====
    vote_agg = add_vote_fields(build_vote_agg(df))

    final_ksf = vote_agg[vote_agg["final_ksf"] == 1].copy()
    final_non_ksf = vote_agg[vote_agg["final_ksf"] == 0].copy()

    # ===== Default output: summary only =====
    print(f"Total: {len(df)} records -> {len(vote_agg)} unique models")
    print(f"  KSF: {len(final_ksf)} ({100 * len(final_ksf) / len(vote_agg):.1f}%)")
    print(
        f"  Non-KSF: {len(final_non_ksf)} ({100 * len(final_non_ksf) / len(vote_agg):.1f}%)"
    )

    # ===== --stats: by year/manufacturer/model =====
    if stats:
        print("\n" + "-" * 60)
        print("KSF Displays - By Manufacturer (Top 15)")
        print("-" * 60)
        mfr_counts = ksf["manufacturer"].value_counts().head(15)
        for mfr, cnt in mfr_counts.items():
            print(f"  {mfr:<30} {cnt:>5}")

        print("\n" + "-" * 60)
        print("KSF Displays - By Model (Top 20)")
        print("-" * 60)
        display_counts = ksf["display"].value_counts().head(20)
        for disp, cnt in display_counts.items():
            print(f"  {disp:<40} {cnt:>5}")

        print("\n" + "-" * 60)
        print("KSF vs Non-KSF - By Year")
        print("-" * 60)
        year_stats = df.groupby("year").agg({"likely_ksf": ["sum", "count"]}).round(1)
        year_stats.columns = ["KSF_Count", "Total"]
        year_stats["KSF%"] = (
            100 * year_stats["KSF_Count"] / year_stats["Total"]
        ).round(1).astype(str) + "%"
        print(year_stats.sort_index(ascending=False).to_string())

    # ===== --debug: distributions and disputed models =====
    if debug:
        print("\n" + "-" * 60)
        print("KSF Feature Statistics")
        print("-" * 60)
        print(
            f"  Peak (nm):   {ksf['peak_nm'].mean():.1f} ± {ksf['peak_nm'].std():.1f}"
        )
        print(
            f"  FWHM (nm):   {ksf['fwhm_nm'].mean():.1f} ± {ksf['fwhm_nm'].std():.1f}"
        )
        print(
            f"  Sharpness:   {ksf['sharpness'].mean():.1f} ± {ksf['sharpness'].std():.1f}"
        )

        print("\n" + "-" * 60)
        print("KSF vs Non-KSF Comparison")
        print("-" * 60)
        print(f"  {'Metric':<12} {'KSF':<25} {'Non-KSF':<25}")
        ksf_peak = f"{ksf['peak_nm'].mean():.1f} ± {ksf['peak_nm'].std():.1f}"
        non_peak = f"{non_ksf['peak_nm'].mean():.1f} ± {non_ksf['peak_nm'].std():.1f}"
        print(f"  {'Peak (nm)':<12} {ksf_peak:<25} {non_peak:<25}")
        ksf_fwhm = f"{ksf['fwhm_nm'].mean():.1f} ± {ksf['fwhm_nm'].std():.1f}"
        non_fwhm = f"{non_ksf['fwhm_nm'].mean():.1f} ± {non_ksf['fwhm_nm'].std():.1f}"
        print(f"  {'FWHM (nm)':<12} {ksf_fwhm:<25} {non_fwhm:<25}")
        ksf_sharp = f"{ksf['sharpness'].mean():.1f} ± {ksf['sharpness'].std():.1f}"
        non_sharp = (
            f"{non_ksf['sharpness'].mean():.1f} ± {non_ksf['sharpness'].std():.1f}"
        )
        print(f"  {'Sharpness':<12} {ksf_sharp:<25} {non_sharp:<25}")

        print("\n" + "-" * 60)
        print("KSF Displays - Peak Distribution")
        print("-" * 60)
        peak_bins = pd.cut(ksf["peak_nm"], bins=[628, 630, 632, 634, 636, 638])
        print(peak_bins.value_counts().sort_index().to_string())

        print("\n" + "-" * 60)
        print("KSF Displays - FWHM Distribution")
        print("-" * 60)
        fwhm_bins = pd.cut(ksf["fwhm_nm"], bins=[0, 10, 15, 20, 25])
        print(fwhm_bins.value_counts().sort_index().to_string())

        disputed = vote_agg[
            (vote_agg["ksf_votes"] > 0)
            & (vote_agg["ksf_votes"] < vote_agg["total_votes"])
        ]
        print("\n" + "-" * 60)
        print("Disputed Models (mixed classifications)")
        print("-" * 60)
        print(f"Total: {len(disputed)}")
        if len(disputed) > 0:
            for _, row in disputed.head(10).iterrows():
                result = "-> KSF" if row["final_ksf"] else "-> Non-KSF"
                print(f"  {row['display']}: {row['vote_ratio']} {result}")

    # ===== Export CSVs =====
    common_cols = [
        "display",
        "manufacturer",
        "year",
        "best_peak",
        "best_fwhm",
        "best_sharpness",
        "confidence",
        "vote_ratio",
    ]
    rename_best = {
        "best_peak": "peak_nm",
        "best_fwhm": "fwhm_nm",
        "best_sharpness": "sharpness",
    }

    ksf_export = export_csv(
        final_ksf,
        out_path="all_classified_ksf.csv",
        columns=common_cols,
        rename=rename_best,
        sort_by=["year", "confidence"],
        ascending=[False, False],
    )

    non_ksf_export = export_csv(
        final_non_ksf,
        out_path="all_classified_non_ksf.csv",
        columns=common_cols,
        rename=rename_best,
        sort_by=["year", "fwhm_nm"],
        ascending=[False, False],
    )

    print("\nExported:")
    print(f"  all_classified_ksf.csv ({len(ksf_export)} models)")
    print(f"  all_classified_non_ksf.csv ({len(non_ksf_export)} models)")


def run_laptop_export(stats: bool = False):
    """Filter and export laptop panels"""
    df = load_classified()

    # Group by display+manufacturer for majority voting
    vote_agg = add_vote_fields(build_vote_agg(df))

    # Filter laptop screens
    laptop_vote = vote_agg[vote_agg["screen_size"].notna()].copy()
    laptop_ksf = laptop_vote[laptop_vote["final_ksf"] == 1]
    laptop_non_ksf = laptop_vote[laptop_vote["final_ksf"] == 0]

    print(f"\nLaptop panels: {len(laptop_vote)} models")
    print(f"  KSF: {len(laptop_ksf)} ({100 * len(laptop_ksf) / len(laptop_vote):.1f}%)")
    print(
        f"  Non-KSF: {len(laptop_non_ksf)} ({100 * len(laptop_non_ksf) / len(laptop_vote):.1f}%)"
    )

    # --stats: distribution by size
    if stats:
        print("\n" + "-" * 60)
        print("Laptop Panels - By Size")
        print("-" * 60)
        size_stats = laptop_vote.groupby("screen_size").agg(
            {"final_ksf": ["sum", "count"]}
        )
        size_stats.columns = ["KSF", "Total"]
        size_stats["Non-KSF"] = size_stats["Total"] - size_stats["KSF"]
        size_stats["KSF%"] = (100 * size_stats["KSF"] / size_stats["Total"]).round(
            1
        ).astype(str) + "%"
        print(size_stats[["KSF", "Non-KSF", "Total", "KSF%"]].sort_index().to_string())

    laptop_cols = [
        "display",
        "manufacturer",
        "year",
        "screen_size",
        "best_peak",
        "best_fwhm",
        "best_sharpness",
        "vote_ratio",
    ]
    rename_best = {
        "best_peak": "peak_nm",
        "best_fwhm": "fwhm_nm",
        "best_sharpness": "sharpness",
    }

    laptop_ksf_export = export_csv(
        laptop_ksf,
        out_path="laptop_classified_ksf.csv",
        columns=laptop_cols,
        rename=rename_best,
        sort_by=["screen_size", "year"],
        ascending=[True, False],
    )

    laptop_non_ksf_export = export_csv(
        laptop_non_ksf,
        out_path="laptop_classified_non_ksf.csv",
        columns=laptop_cols,
        rename=rename_best,
        sort_by=["screen_size", "year"],
        ascending=[True, False],
    )

    print("\nExported:")
    print(f"  laptop_classified_ksf.csv ({len(laptop_ksf_export)} models)")
    print(f"  laptop_classified_non_ksf.csv ({len(laptop_non_ksf_export)} models)")


def run_hires_export():
    """Filter non-KSF laptop panels with resolution > 1080p"""
    # --- Process ---
    df = pd.read_csv("all_classified_non_ksf.csv")
    df["is_laptop_panel"] = df["display"].apply(is_laptop_panel)
    panels = df[df["is_laptop_panel"]].copy()
    panels["hires_reason"] = panels["display"].apply(hires_reason)
    gt1080 = panels[panels["hires_reason"] != ""].copy()

    gt1080.to_csv("laptop_classified_non_ksf_hires.csv", index=False)

    print(f"\nHigh-res (>1080p) non-KSF laptop panels: {len(gt1080)}")
    print("  -> laptop_classified_non_ksf_hires.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze KSF display results")
    parser.add_argument(
        "--laptop",
        action="store_true",
        help="Also export laptop panel classifications",
    )
    parser.add_argument(
        "--hires",
        action="store_true",
        help="Also filter high-res (>1080p) laptop panels (requires --laptop)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics by year/manufacturer/model",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug info (peak/fwhm distributions, disputed models)",
    )
    args = parser.parse_args()

    run_aggregate(stats=args.stats, debug=args.debug)
    if args.laptop:
        run_laptop_export(stats=args.stats)
    if args.hires:
        if not args.laptop:
            print("\nWarning: --hires requires --laptop, skipping high-res filter")
        else:
            run_hires_export()


# endregion
