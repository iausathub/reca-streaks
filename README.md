# reca-streaks

Satellite streak detection and photometry for Dark Energy Camera (DECam)
images, developed as part of the Summer 2025 RECA research project.

This repository accompanies the paper
*Identifying and Measuring Satellite Streaks in DECam Images*.

## Repository Structure

```
reca-streaks/
├── reca_streaks/              # Python package
│   ├── __init__.py
│   ├── constants.py           # Physical & instrumental constants
│   ├── data.py                # NOIRLab archive data retrieval
│   └── photometry.py          # Aperture & PSF-fitting photometry
├── notebooks/
│   ├── analysis/
│   │   ├── Analyze_Streaks_V3.0.ipynb        # Full analysis pipeline
│   │   └── generate_paper_figures.ipynb       # Reproduce paper figures
│   ├── exploration/
│   │   ├── DECam-RECA-explore.ipynb           # DECam data exploration
│   │   └── HowBrightIsThatSatellite.ipynb     # Satellite brightness estimates
│   └── sandbox/
│       ├── Satellite_Streaks_Practice.ipynb    # Practice notebook
│       └── andres_sandbox/                     # Development notebooks
├── figures/                   # Output figures (PDFs)
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
# Clone and set up a virtual environment
git clone <repo-url> reca-streaks
cd reca-streaks
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the package and dependencies
pip install -e .

# For notebook extras (jupyterlab, opencv, etc.)
pip install -e ".[notebooks]"
```

## External Dependency: satmetrics

The line-detection pipeline relies on the
[satmetrics](https://github.com/andresplazas/satmetrics) package, which
must be cloned separately:

```bash
git clone https://github.com/andresplazas/satmetrics.git ../satmetrics
```

The notebooks add `satmetrics` to `sys.path` automatically. Adjust the
path in the import cell if your clone lives elsewhere.

## Quick Start

```python
from reca_streaks.data import retrieve_hdu_image
from reca_streaks.photometry import streak_photometry

# Download a DECam exposure
hdu_list = retrieve_hdu_image(expnum=1134933, detector=5)
image_data = hdu_list[1].data

# Run aperture photometry on a rotated streak cutout
result = streak_photometry(image_data, hdu_list=hdu_list)
print(result["sb_arcsec"], result["sb_arcsec_err"])
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `Analyze_Streaks_V3.0.ipynb` | End-to-end analysis: detection, rotation, photometry for multiple streaks |
| `generate_paper_figures.ipynb` | Reproduces Figures 2 and 3 from the paper as publication-quality PDFs |
| `DECam-RECA-explore.ipynb` | Exploratory look at DECam images and archive queries |
| `HowBrightIsThatSatellite.ipynb` | Brightness estimation exercises |
| `Satellite_Streaks_Practice.ipynb` | Introductory practice notebook |

## Reproducing Paper Figures

```bash
cd notebooks/analysis
jupyter lab generate_paper_figures.ipynb
```

Run all cells. Figures 2 and 3 will be saved as PDFs in the `figures/`
directory.

## Citation

Final report:
[Google Drive link](https://drive.google.com/file/d/197nayTaqmTiJN0onE_tskd_5oKVXJGbA/view)

## License

This project is released under the MIT License. See [LICENSE](LICENSE)
for details.
