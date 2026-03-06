"""reca_streaks — Satellite streak photometry for DECam images."""

__version__ = "0.1.0"

from .data import retrieve_hdu_image
from .photometry import (
    gaussian_1d,
    estimate_flux_uncertainty,
    streak_photometry,
    streak_photometry_psf_fitting,
)

__all__ = [
    "retrieve_hdu_image",
    "gaussian_1d",
    "estimate_flux_uncertainty",
    "streak_photometry",
    "streak_photometry_psf_fitting",
]
