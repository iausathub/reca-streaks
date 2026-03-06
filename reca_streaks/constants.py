"""Physical and instrumental constants for DECam streak photometry."""

# DECam pixel scale
DECAM_PIXEL_SCALE_ARCSEC = 0.2634  # arcsec per pixel

# Source masking threshold (number of sigma above median)
DEFAULT_SIGMA_MASK = 5.0

# Aperture width in units of Gaussian sigma
APERTURE_NSIGMA = 3.0

# Conversion factor: FWHM = FWHM_SIGMA_FACTOR * sigma
FWHM_SIGMA_FACTOR = 2.3548

# Magnitude error approximation: delta_m ≈ MAG_ERR_FACTOR / SNR
MAG_ERR_FACTOR = 1.0857

# Signal-to-noise threshold to distinguish background- vs source-dominated regime
SNR_REGIME_THRESHOLD = 5.0
