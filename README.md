# KSF Display Detector

Fetches spectral data from [DisplayCAL Colorimeter Corrections Database](https://colorimetercorrections.displaycal.net/) via its public API and automatically identifies displays using **KSF narrow-band red phosphor (K₂SiF₆:Mn⁴⁺)** based on their spectral characteristics.

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python fetch.py                       # Fetch CCSS data
python evaluate.py                    # Classify displays as KSF/non-KSF
python aggregate.py                   # Export aggregated results
python aggregate.py --laptop          # Also export aggregated laptop panels
python aggregate.py --laptop --hires  # Also export aggregated high-res (>1080p) laptops
```

## Output Files

### fetch.py
- `all_raw_spectra.csv` - Raw spectral data fetched from DisplayCAL database
- `visited_hashes.json` - Cache of already visited hashes (for incremental fetching)

### evaluate.py
- `all_classified.csv` - Full results with all spectral data and classification

### aggregate.py
- `all_classified_ksf.csv` / `all_classified_non_ksf.csv` - KSF/non-KSF displays

### aggregate.py --laptop
- `laptop_classified_ksf.csv` / `laptop_classified_non_ksf.csv` - KSF/non-KSF laptop panels

### aggregate.py --laptop --hires
- `laptop_classified_non_ksf_hires.csv` - Non-KSF laptop panels with resolution > 1080p

## Disclaimer

Spectral data is obtained from the [DisplayCAL Colorimeter Corrections Database](https://colorimetercorrections.displaycal.net/) via its public API. As stated on the website, all data is user-contributed and in the public domain. They may contain inaccuracies or errors.

KSF classification/laptop panel filtering is heuristic-based and may not be accurate.

This repository was written with significant **AI assistance (GitHub Copilot)** for my personal use. It is provided as-is without any guarantees or support. Use at your own risk.

## License

[MIT](LICENSE)
