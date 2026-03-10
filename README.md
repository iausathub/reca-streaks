# reca-streaks

Satellite streak photometry for Dark Energy Camera (DECam) images. This repository contains code and notebooks used in the paper "DECam_streaks_photometry_RECA2025_eSPECTRA_2026MAR10.pdf".

## Overview

This repository provides tools to:
- Download DECam FITS images from NOIRLab archive
- Detect satellite streaks using Hough transform
- Perform photometry on detected streaks using two methods:
  - Aperture photometry with Gaussian profile fitting
  - PSF-fitting photometry based on Veres+12 trail model

## Repository Structure

```
reca-streaks/
├── streak_photometry.py          # Main photometry functions
├── get_decam_data.py             # NOIRLab data retrieval
├── notebooks/
│   └── paper_notebooks/
│       ├── DECam_streaks_photometry-V2.ipynb  # Main analysis notebook
│       ├── get_data.ipynb                     # Data download 
│       └── simulated_streak.ipynb             # Streak simulations
└── README.md
```

The code also relies on the companion `satmetrics` package (located in `../satmetrics/`):
- `line_detection_updated.py` - Hough transform line detection
- `image_rotation.py` - Image rotation and streak isolation
- `satmetrics.py` - Main streak detection pipeline

## Installation

### Dependencies

```bash
pip install numpy matplotlib scipy astropy photutils scikit-image opencv-python scikit-learn requests pandas
```

You will also need access to the `satmetrics` package modules. Add them to your Python path:

```python
import sys
sys.path.append('/path/to/satmetrics')
```

## Getting DECam Data

DECam FITS files can be downloaded from the NOIRLab archive using the `get_decam_data.py` module or the examples in `notebooks/paper_notebooks/get_data.ipynb`.

### Example: Download a specific exposure and detector

```python
from get_decam_data import retrieve_hdu_image

# Download exposure 1103448, detector 4
expnum = 1103448
detector = 4
hdu_list = retrieve_hdu_image(expnum, detector)

# Access the data
header = hdu_list[0].header  # Primary header with MAGZERO, etc.
header_image = hdu_list[1].header  # Image header with GAIN, RDNOISE, etc.
image_data = hdu_list[1].data  # The image array
```

The `retrieve_hdu_image()` function:
1. Queries the NOIRLab archive API for the specified exposure number
2. Retrieves the MD5 hash for the multi-extension FITS file
3. Downloads only the requested detector HDU using the API
4. Returns an `astropy.io.fits.HDUList` object

**Note:** Images are cached automatically using `astropy.utils.data.download_file()`.

## Core Photometry Functions

The `streak_photometry.py` module provides the main analysis functions.

### 1. Aperture Photometry

`streak_photometry_aperture()` - Measures surface brightness using a rectangular aperture defined by the streak length and fitted Gaussian width.

**Method:**
1. Masks bright sources (stars) using sigma clipping
2. Collapses the image along the x-axis to create a 1D profile
3. Fits a Gaussian to determine the streak center and width (σ)
4. Defines on-streak region: ±3σ around center
5. Measures background in off-streak regions above/below the streak
6. Subtracts background and computes flux in aperture
7. Converts to surface brightness in counts/arcsec²

**Returns:** Surface brightness and uncertainty

**Example:**

```python
from streak_photometry import streak_photometry_aperture

# Assuming you have a cropped image containing a horizontal streak
sb, sb_err = streak_photometry_aperture(
    image_data,
    expnum=1103448,
    detector=4,
    sigma_mask=5,  # Mask sources > 5σ
    make_plots=True  # Generate diagnostic plots
)
```

**Output:**
```
=== Aperture Photometry Result ===
Streak center (y0): 48.25 px
Width: σ = 3.87 px, FWHM ≈ 9.11 px
Streak flux: 3029942.61 ADU ± 2660.20
Surface brightness: 2562.46 ± 2.25 counts/arcsec² [source-dominated]
SNR: 1138.99
Surface brightness: 18.935 ± 0.001 mag/arcsec²
```

### 2. PSF-Fitting Photometry

`streak_photometry_psf_fitting()` - Fits a 2D trail model to measure surface brightness.

**Method:**
- Implements the trail model from Veres et al. 2012, "Improved Asteroid Astrometry and Photometry with Trail Fitting" (Eq. 3)
- Model parameters:
  - `b`: background level
  - `phi`: total integrated flux
  - `L`: trail length
  - `sigma`: PSF width
  - `theta`: trail angle
  - `x0, y0`: trail center

**Returns:** Dictionary with photometric measurements and fit parameters

**Example:**

```python
from streak_photometry import streak_photometry_psf_fitting

results = streak_photometry_psf_fitting(
    image_data,
    expnum=1103448,
    detector=4,
    sigma_mask=5,
    make_plots=True
)

print(f"Surface brightness: {results['SB_mag']} mag/arcsec²")
print(f"SNR: {results['SNR']}")
```

**Output:**
```
=== Trail Fit Results (θ free) ===
Background: 269.80 counts/pixel
Total flux (phi): 3914172.06 ± 0.56 ADU
Trail length (L): 1977.48 px
PSF sigma: 3.42 px → FWHM = 8.06 px --> 2.12 arcsec
Trail angle θ: -0.21 deg
Center: x0 = 758.84, y0 = 48.31
Surface brightness: 2777.78 ± 0.00 counts/arcsec²
S/N (section 3): 1386.5
Chi²: 4302522.71
Reduced Chi²: 286.3576
Surface brightness: 18.529 ± 0.0002 mag/arcsec²
```

## Notebooks

### `DECam_streaks_photometry-V2.ipynb`

Main analysis notebook used to generate results in the paper.

**Workflow:**
1. Load DECam FITS files (assumes files are already downloaded)
2. Detect streaks using Hough transform
3. Rotate images to make streaks horizontal
4. Perform both aperture and PSF-fitting photometry
5. Generate comparison tables and plots

**Requirements:**
- DECam FITS files must be downloaded locally
- Update file paths in the notebook to point to your data directory

**Example cell:**

```python
import sys
sys.path.append('/home/plazas/WORK/science/satellites/satmetrics')
sys.path.append('/home/plazas/WORK/science/satellites/reca-streaks')

import line_detection_updated as ld
import image_rotation as ir
import streak_photometry
from astropy.io import fits

# Load image
expnum = 1103448
detector = 4
hdu_list = fits.open('path/to/your/fits/file')
image_data = hdu_list[1].data

# Detect and rotate streak
detections = streak_photometry.detect_lines_hough(image_data)
rotated_image, coords, angle = streak_photometry.rotate_streak_horizontal(
    image_data, detections["Lines"]
)

# Perform photometry
sb_aper, sb_aper_err = streak_photometry.streak_photometry_aperture(
    rotated_image, hdu_list=hdu_list
)

results_psf = streak_photometry.streak_photometry_psf_fitting(
    rotated_image, hdu_list=hdu_list
)
```

### `get_data.ipynb`

Demonstrates how to download DECam data from NOIRLab archive.

**Use this notebook to:**
- Learn the NOIRLab API query syntax
- Download specific exposures and detectors
- Inspect FITS headers and metadata

### `simulated_streak.ipynb`

Contains simulations and tests with synthetic streak data.

## Satmetrics Module Overview

The companion `satmetrics` package provides the low-level streak detection algorithms.

### `line_detection_updated.py`

**LineDetection class** - Implements the Hough transform-based line detection pipeline:

1. **Background removal:** Uses photutils Background2D with sigma clipping
2. **Brightness cuts:** Clips outliers to reduce noise
3. **Standardization:** Normalizes image to 0-255 range
4. **Binary thresholding:** Creates binary image based on brightness threshold
5. **Adaptive blurring:** Applies median blur with kernel size based on image noise
6. **Edge detection:** Canny edge detector
7. **Hough transform:** scikit-image hough_line and hough_line_peaks
8. **Clustering:** MeanShift clustering to group detected lines

**Key parameters:**
- `threshold`: Voting threshold as fraction of image diagonal (default: 0.075)
- `brightness_cuts`: Sigma clipping limits (default: (2, 2))
- `thresholding_cut`: Binary threshold in standard deviations (default: 0.5)
- `flux_prop_thresholds`: Thresholds for bright pixel fraction (default: [0.1, 0.2, 0.3, 1])
- `blur_kernel_sizes`: Kernel sizes for adaptive blurring (default: [3, 5, 9, 11])

### `image_rotation.py`

Functions for rotating images to align streaks horizontally:

- `get_edge_intersections()`: Converts polar coordinates to Cartesian edge points
- `determine_rotation_angle()`: Calculates rotation angle from line coordinates
- `rotate_image()`: Rotates and crops image around streak
- `complete_rotate_image()`: Full pipeline with Gaussian fitting validation

### `satmetrics.py`

Command-line tool for batch processing:

```bash
python satmetrics.py file1.fits file2.fits --config config.yaml --output results.yaml
```

## Example Workflow

Here's a complete example reproducing the paper's analysis:

```python
import sys
sys.path.append('/path/to/satmetrics')

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

import line_detection_updated as ld
import image_rotation as ir
import streak_photometry
from get_decam_data import retrieve_hdu_image

# Step 1: Download data
expnum = 1138498
detector = 23
hdu_list = retrieve_hdu_image(expnum, detector)
image = hdu_list[1].data

# Step 2: Detect streaks
detections = streak_photometry.detect_lines_hough(
    image,
    threshold=0.075,
    brightness_cuts=(2, 2)
)

print(f"Detected {len(detections['Lines'])} lines")

# Step 3: Rotate to horizontal
if len(detections["Lines"]) > 0:
    rotated_image, coords, angle = streak_photometry.rotate_streak_horizontal(
        image, detections["Lines"]
    )
    
    # Step 4: Perform photometry
    print("\n--- Aperture Photometry ---")
    sb_aper, sb_aper_err = streak_photometry.streak_photometry_aperture(
        rotated_image,
        hdu_list=hdu_list,
        make_plots=True
    )
    
    print("\n--- PSF Fitting Photometry ---")
    results = streak_photometry.streak_photometry_psf_fitting(
        rotated_image,
        hdu_list=hdu_list,
        make_plots=True
    )
    
    # Step 5: Display results
    print(f"\nAperture SB: {sb_aper:.2f} ± {sb_aper_err:.2f} counts/arcsec²")
    print(f"PSF Fit SB: {results['SB_counts']:.2f} counts/arcsec²")
    print(f"PSF Fit SB: {results['SB_mag']:.3f} mag/arcsec²")
```

## Key Parameters

### DECam Image Properties
- Pixel scale: 0.2634 arcsec/pixel
- Typical gain: ~3.4 e⁻/ADU (averaged over amplifiers A and B)
- Typical read noise: ~6 e⁻

### Photometry Settings
- **sigma_mask**: Sigma threshold for masking bright sources (default: 5)
- **Aperture width**: 3σ (cross-trail direction)
- **Background regions**: Adjacent regions of equal height to aperture

### Uncertainty Estimation
- Background-dominated: σ² = N_pix × (σ_bkg² + σ_read²)
- Source-dominated: σ² = Flux + N_pix × (σ_bkg² + σ_read²)
- Transition threshold: 5σ_bkg

## Citation

If you use this code, please cite:

 https://drive.google.com/file/d/197nayTaqmTiJN0onE_tskd_5oKVXJGbA/view

## References

- **Veres et al. 2012**: "Improved Asteroid Astrometry and Photometry with Trail Fitting"  
  https://iopscience.iop.org/article/10.1086/668616/pdf

- **NOIRLab Archive**: https://astroarchive.noirlab.edu


## License

See LICENSE file for details.

## Contact

For questions about this code, please open an issue on GitHub or contact the authors.
Andrés Alejandro Plazas Malagón (plazasmalagon@gmail.com)
Meredith Rawls (mrawls@uw.edu)

