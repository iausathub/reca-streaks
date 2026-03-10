"""Aperture and PSF-fitting photometry for satellite streaks in DECam images."""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

from .constants import (
    APERTURE_NSIGMA,
    DECAM_PIXEL_SCALE_ARCSEC,
    DEFAULT_SIGMA_MASK,
    FWHM_SIGMA_FACTOR,
    MAG_ERR_FACTOR,
    SNR_REGIME_THRESHOLD,
)
from .data import retrieve_hdu_image

__all__ = [
    "gaussian_1d",
    "estimate_flux_uncertainty",
    "streak_photometry",
    "streak_photometry_psf_fitting",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def gaussian_1d(y: np.ndarray, A: float, y0: float, sigma: float,
                offset: float) -> np.ndarray:
    """1-D Gaussian plus constant offset.

    Parameters
    ----------
    y : array_like
        Independent variable (pixel coordinate).
    A : float
        Peak amplitude above the offset.
    y0 : float
        Center position.
    sigma : float
        Standard deviation.
    offset : float
        Constant background level.

    Returns
    -------
    ndarray
        Model evaluated at ``y``.
    """
    return A * np.exp(-(y - y0) ** 2 / (2 * sigma ** 2)) + offset


def estimate_flux_uncertainty(
    streak_flux: float,
    on_unmasked: int,
    bkg_std: float,
    gain: float,
    read_noise: float,
    threshold: float = SNR_REGIME_THRESHOLD,
) -> tuple[float, str, float]:
    """Estimate flux uncertainty in ADU for aperture photometry.

    Parameters
    ----------
    streak_flux : float
        Total background-subtracted signal in the aperture [ADU].
    on_unmasked : int
        Number of unmasked pixels in the on-streak aperture.
    bkg_std : float
        Standard deviation of the background pixels [ADU].
    gain : float
        Detector gain [e-/ADU].
    read_noise : float
        Read noise [e-].
    threshold : float, optional
        SNR threshold separating background- and source-dominated regimes.

    Returns
    -------
    flux_err : float
        Uncertainty in flux [ADU].
    regime : str
        ``'background-dominated'`` or ``'source-dominated'``.
    snr : float
        Signal-to-noise ratio.
    """
    flux_e = gain * streak_flux
    bkg_std_e = gain * bkg_std

    signal_per_pixel = flux_e / on_unmasked
    noise_background_var = on_unmasked * (bkg_std_e ** 2 + read_noise ** 2)

    if signal_per_pixel < (threshold * bkg_std_e):
        regime = "background-dominated"
        total_var_e = noise_background_var
    else:
        regime = "source-dominated"
        total_var_e = flux_e + noise_background_var

    flux_err = np.sqrt(total_var_e) / gain
    snr = flux_e / np.sqrt(total_var_e)

    log.info("Regime: %s | flux_err=%.2f ADU | SNR=%.1f", regime, flux_err, snr)
    return flux_err, regime, snr


def _read_header_info(
    expnum: Optional[int],
    detector: Optional[int],
    hdu_list,
) -> tuple:
    """Read gain, read noise, and zeropoint from the HDU list.

    Parameters
    ----------
    expnum, detector : int or None
        If both are provided the HDU list is downloaded.
    hdu_list : HDUList or None
        Pre-loaded FITS HDU list.

    Returns
    -------
    hdu_list, gain, read_noise, zeropoint
    """
    if (expnum is not None) and (detector is not None):
        hdu_list = retrieve_hdu_image(expnum, detector)
    elif hdu_list is None:
        raise ValueError("Provide (expnum, detector) or hdu_list")

    header = hdu_list[1].header
    header_expnum = hdu_list[0].header

    gain = 0.5 * (header["GAINA"] + header["GAINB"])
    read_noise = 0.5 * (header["RDNOISEA"] + header["RDNOISEB"])

    try:
        zeropoint = header_expnum["MAGZERO"]
    except KeyError:
        log.warning("Photometric zeropoint not available; reporting instrumental flux only.")
        zeropoint = None

    return hdu_list, gain, read_noise, zeropoint


# ---------------------------------------------------------------------------
# Aperture photometry
# ---------------------------------------------------------------------------

def streak_photometry(
    image_data: np.ndarray,
    expnum: Optional[int] = None,
    detector: Optional[int] = None,
    hdu_list=None,
    sigma_mask: float = DEFAULT_SIGMA_MASK,
    make_plots: bool = False,
    save_path: Optional[str] = None,
) -> dict:
    """Aperture photometry on a satellite streak.

    Uses a rectangular aperture whose width is set by the fitted Gaussian FWHM
    of the cross-streak profile.

    Parameters
    ----------
    image_data : ndarray
        2-D image cutout containing a horizontal streak.
    expnum : int, optional
        DECam exposure number (used to fetch the header if ``hdu_list`` is None).
    detector : int, optional
        CCD detector number.
    hdu_list : HDUList, optional
        Pre-loaded FITS HDU list.
    sigma_mask : float, optional
        Sigma clipping threshold for bright-source masking.
    make_plots : bool, optional
        If *True*, display diagnostic plots.
    save_path : str, optional
        If given, save the profile plot as a PDF at this path.

    Returns
    -------
    result : dict
        Dictionary containing photometry results and intermediate quantities:
        ``sb_arcsec``, ``sb_arcsec_err``, ``sb_mag``, ``sb_mag_err``,
        ``regime``, ``snr``, ``streak_flux``, ``streak_flux_err``,
        ``zeropoint``, ``fit_params`` (A, y0, sigma, offset),
        ``fwhm_pix``, ``profile_y``, ``y``,
        ``on_ymin``, ``on_ymax``, ``off1_ymin``, ``off1_ymax``,
        ``off2_ymin``, ``off2_ymax``, ``region_mask``.
    """
    hdu_list, gain, read_noise, zeropoint = _read_header_info(
        expnum, detector, hdu_list
    )

    # Step 1: Mask bright sources
    median = np.median(image_data)
    std = np.std(image_data)
    threshold = median + sigma_mask * std
    mask = image_data > threshold
    masked_data = np.ma.array(image_data, mask=mask)

    # Step 2: Collapse to 1-D cross-streak profile
    profile_y = np.ma.mean(masked_data, axis=1)
    y = np.arange(len(profile_y))

    # Step 3: Fit Gaussian
    A0 = float(profile_y.max())
    y0_guess = int(np.argmax(profile_y))
    sigma0 = 3.0
    offset0 = float(np.median(profile_y))
    p0 = [A0, y0_guess, sigma0, offset0]
    popt, _ = curve_fit(gaussian_1d, y, profile_y.filled(offset0), p0=p0)
    A_fit, y0_fit, sigma_fit, offset_fit = popt
    fwhm_pix = FWHM_SIGMA_FACTOR * abs(sigma_fit)

    # Step 4: On/off-streak regions
    on_ymin = int(y0_fit - APERTURE_NSIGMA * abs(sigma_fit))
    on_ymax = int(y0_fit + APERTURE_NSIGMA * abs(sigma_fit))
    height = on_ymax - on_ymin

    off1_ymin = max(0, on_ymin - height)
    off1_ymax = on_ymin
    off2_ymin = on_ymax
    off2_ymax = min(image_data.shape[0], on_ymax + height)

    region_mask = np.zeros_like(image_data, dtype=int)
    region_mask[on_ymin:on_ymax, :] = 1
    region_mask[off1_ymin:off1_ymax, :] = 2
    region_mask[off2_ymin:off2_ymax, :] = 2

    on_region = masked_data[on_ymin:on_ymax, :]
    off1_region = masked_data[off1_ymin:off1_ymax, :]
    off2_region = masked_data[off2_ymin:off2_ymax, :]

    # Step 5: Flux measurement
    on_sum = on_region.sum()
    on_unmasked_pixels = int(np.sum(~on_region.mask))

    off_pixels = int(np.sum(~off1_region.mask) + np.sum(~off2_region.mask))
    off_sum = off1_region.sum() + off2_region.sum()
    bkg_per_pixel = off_sum / off_pixels

    empirical_bkg = bkg_per_pixel * on_unmasked_pixels
    streak_flux = float(on_sum - empirical_bkg)

    pixel_scale = DECAM_PIXEL_SCALE_ARCSEC
    trail_length_pix = image_data.shape[1]
    area_arcsec2 = trail_length_pix * fwhm_pix * pixel_scale ** 2
    sb_arcsec = streak_flux / area_arcsec2

    # Step 6: Error estimation
    off_vals = np.hstack([
        off1_region[~off1_region.mask].ravel(),
        off2_region[~off2_region.mask].ravel(),
    ])
    bkg_std = float(np.std(off_vals))

    streak_flux_err, regime, snr = estimate_flux_uncertainty(
        streak_flux, on_unmasked_pixels, bkg_std, gain, read_noise
    )
    sb_arcsec_err = streak_flux_err / area_arcsec2

    log.info("Streak center y0=%.2f px, sigma=%.2f px, FWHM=%.2f px",
             y0_fit, sigma_fit, fwhm_pix)
    log.info("Streak flux: %.2f +/- %.2f ADU", streak_flux, streak_flux_err)
    log.info("SB: %.2f +/- %.2f counts/arcsec^2 [%s]",
             sb_arcsec, sb_arcsec_err, regime)
    log.info("SNR: %.2f", snr)

    sb_mag = None
    sb_mag_err = None
    if zeropoint is not None:
        sb_mag = -2.5 * np.log10(sb_arcsec) + zeropoint
        sb_mag_err = MAG_ERR_FACTOR * (sb_arcsec_err / sb_arcsec)
        log.info("SB: %.3f +/- %.3f mag/arcsec^2 (zp=%.2f)",
                 sb_mag, sb_mag_err, zeropoint)

    # Plots
    if make_plots or save_path:
        _plot_profile(y, profile_y, popt, on_ymin, on_ymax,
                      off1_ymin, off1_ymax, off2_ymin, off2_ymax,
                      save_path=save_path)

    if make_plots:
        _plot_mask(mask)
        _plot_regions(image_data, region_mask)

    return {
        "sb_arcsec": sb_arcsec,
        "sb_arcsec_err": sb_arcsec_err,
        "sb_mag": sb_mag,
        "sb_mag_err": sb_mag_err,
        "regime": regime,
        "snr": snr,
        "streak_flux": streak_flux,
        "streak_flux_err": streak_flux_err,
        "zeropoint": zeropoint,
        "fit_params": {"A": A_fit, "y0": y0_fit, "sigma": sigma_fit,
                       "offset": offset_fit},
        "fwhm_pix": fwhm_pix,
        "profile_y": profile_y,
        "y": y,
        "on_ymin": on_ymin,
        "on_ymax": on_ymax,
        "off1_ymin": off1_ymin,
        "off1_ymax": off1_ymax,
        "off2_ymin": off2_ymin,
        "off2_ymax": off2_ymax,
        "region_mask": region_mask,
        "image_data": image_data,
        "mask": mask,
    }


# ---------------------------------------------------------------------------
# PSF-fitting photometry (Veres+12 trail model)
# ---------------------------------------------------------------------------

def streak_photometry_psf_fitting(
    image_data: np.ndarray,
    expnum: Optional[int] = None,
    detector: Optional[int] = None,
    hdu_list=None,
    pdf=None,
    sigma_mask: float = DEFAULT_SIGMA_MASK,
    make_plots: bool = False,
) -> dict:
    """PSF-fitting photometry using the Veres+12 trail model.

    Fits Equation 3 of Veres & Jacobson (2012), "Improved Asteroid
    Astrometry and Photometry with Trail Fitting", to a 2-D streak image.

    Parameters
    ----------
    image_data : ndarray
        2-D image cutout containing a streak.
    expnum : int, optional
        DECam exposure number.
    detector : int, optional
        CCD detector number.
    hdu_list : HDUList, optional
        Pre-loaded FITS HDU list.
    pdf : PdfPages, optional
        If given, save the diagnostic figure to this PDF backend.
    sigma_mask : float, optional
        Sigma clipping threshold for bright-source masking.
    make_plots : bool, optional
        If *True*, display diagnostic plots.

    Returns
    -------
    result : dict
        Dictionary with fitted parameters and photometry results.
    """
    pixel_scale = DECAM_PIXEL_SCALE_ARCSEC

    hdu_list, gain, read_noise, zeropoint = _read_header_info(
        expnum, detector, hdu_list
    )
    log.info("Average amp gain (e/ADU): %.2f", gain)
    log.info("Average read noise (e): %.2f", read_noise)

    if zeropoint is not None:
        log.info("Zeropoint [mag]: %.2f", zeropoint)
    else:
        log.warning("No MAGZERO keyword in header")

    # Build bad-pixel mask
    median = np.median(image_data)
    std = np.std(image_data)
    threshold = median + sigma_mask * std
    bp_mask = image_data > threshold

    ny, nx = image_data.shape
    x_grid, y_grid = np.meshgrid(np.arange(nx), np.arange(ny))
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    z_flat = image_data.ravel()
    mask_flat = bp_mask.ravel()

    x_flat = x_flat[~mask_flat]
    y_flat = y_flat[~mask_flat]
    z_flat = z_flat[~mask_flat]

    # Trail model (theta free)
    def trail_model(coords, b, phi, L, sigma, theta, x0, y0):
        x, y = coords
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        x_rot = (x - x0) * sin_t + (y - y0) * cos_t
        Q = (x - x0) * cos_t - (y - y0) * sin_t
        term1 = b
        term2 = (phi / L) * (1 / (2 * sigma * np.sqrt(2 * np.pi)))
        term3 = np.exp(-(x_rot ** 2) / (2 * sigma ** 2))
        erf1 = (Q + L / 2) / (sigma * np.sqrt(2))
        erf2 = (Q - L / 2) / (sigma * np.sqrt(2))
        return term1 + term2 * term3 * (erf(erf1) - erf(erf2))

    # Initial guess and bounds
    b0 = np.median(image_data)
    phi0 = np.sum(image_data) - b0 * image_data.size
    L0 = 2000
    sigma0 = 1.0
    theta0 = 0.0
    x0_0 = nx / 2
    y0_0 = ny / 2
    p0 = [b0, phi0, L0, sigma0, theta0, x0_0, y0_0]

    #bounds = (
    #    [0, 0, 5, 0.3, -np.pi, 0, 0],
    #    [1e9, 1e9, 1e4, 10, np.pi, nx - 1, ny - 1],
    #)

    popt, pcov = curve_fit(
        trail_model, (x_flat, y_flat), z_flat, p0=p0,
        #bounds=bounds, maxfev=5000,
        maxfev=5000,
    )
    b_fit, phi_fit, L_fit, sigma_fit, theta_fit, x0_fit, y0_fit = popt

    # Error propagation via residuals
    model_fit_vals = trail_model((x_flat, y_flat), *popt)
    residuals = z_flat - model_fit_vals
    residual_std = np.std(residuals)
    cov_scaled = pcov * residual_std ** 2
    phi_err = np.sqrt(cov_scaled[1, 1])

    # Reconstruct model image
    model_image = trail_model((x_grid, y_grid), *popt)

    # Define aperture box aligned to trail angle
    length = L_fit + 6 * sigma_fit
    width = 3 * sigma_fit
    Npix = int(length * width)
    log.info("length=%.1f px, width=%.1f px, Npix=%d", length, width, Npix)

    coords_x = x_grid - x0_fit
    coords_y = y_grid - y0_fit
    cos_t = np.cos(theta_fit)
    sin_t = np.sin(theta_fit)
    q = coords_x * cos_t + coords_y * sin_t
    r = coords_x * sin_t - coords_y * cos_t

    ap_mask = ((np.abs(q) < length / 2) & (np.abs(r) < width / 2)) & ~bp_mask
    ap_data = image_data[ap_mask]
    ap_model = model_image[ap_mask]
    ap_N = int(np.sum(ap_mask))

    # Chi-squared
    noise_var = (gain * ap_model + read_noise ** 2) / gain ** 2
    chi2 = float(np.sum((ap_data - ap_model) ** 2 / noise_var))
    chi2_red = chi2 / (ap_N - 7 - 1)

    # S/N (Veres+12 Eq. 8)
    S = phi_fit
    B = b_fit * ap_N
    SNR = S / np.sqrt(S + B)

    # Surface brightness
    psf_fwhm_arcsec = FWHM_SIGMA_FACTOR * sigma_fit * pixel_scale
    area_arcsec2 = L_fit * pixel_scale * psf_fwhm_arcsec
    sb_arcsec2 = S / area_arcsec2
    sb_arcsec2_err = phi_err / area_arcsec2

    log.info("=== Trail Fit Results ===")
    log.info("Background: %.2f counts/pixel", b_fit)
    log.info("Total flux (phi): %.2f +/- %.2f ADU", phi_fit, phi_err)
    log.info("Trail length (L): %.2f px", L_fit)
    log.info("PSF sigma: %.2f px -> FWHM = %.2f px (%.2f arcsec)",
             sigma_fit, FWHM_SIGMA_FACTOR * sigma_fit, psf_fwhm_arcsec)
    log.info("Trail angle: %.2f deg", np.degrees(theta_fit))
    log.info("Center: x0=%.2f, y0=%.2f", x0_fit, y0_fit)
    log.info("SB: %.2f +/- %.2f counts/arcsec^2", sb_arcsec2, sb_arcsec2_err)
    log.info("S/N: %.1f", SNR)
    log.info("Chi2: %.2f, reduced Chi2: %.4f", chi2, chi2_red)

    sb_mag_arcsec2 = None
    sb_mag_arcsec2_err = None
    if zeropoint is not None:
        sb_mag_arcsec2 = zeropoint - 2.5 * np.log10(sb_arcsec2)
        sb_mag_arcsec2_err = MAG_ERR_FACTOR / SNR
        log.info("SB (mag/arcsec^2): %.2f +/- %.4f", sb_mag_arcsec2, sb_mag_arcsec2_err)
    else:
        log.warning("No zeropoint; cannot convert to magnitude")

    final_dict = {
        "SB_counts": sb_arcsec2,
        "SB_counts_err": sb_arcsec2_err,
        "SB_mag": sb_mag_arcsec2,
        "SB_mag_err": sb_mag_arcsec2_err,
        "Zeropoint": zeropoint,
        "Background": b_fit,
        "Flux_total": phi_fit,
        "Flux_err": phi_err,
        "Trail_Length_px": L_fit,
        "PSF_Sigma_px": sigma_fit,
        "PSF_FWHM_arcsec": psf_fwhm_arcsec,
        "Trail_angle_deg": float(np.degrees(theta_fit)),
        "Center_x0": x0_fit,
        "Center_y0": y0_fit,
        "SNR": float(SNR),
        "Chi2": chi2,
        "Chi2_red": chi2_red,
    }

    if make_plots:
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))

        im0 = axes[0].imshow(
            image_data, origin="lower", cmap="viridis",
            vmin=np.percentile(image_data, 5),
            vmax=np.percentile(image_data, 95),
        )
        axes[0].set_title("Observed Image")
        plt.colorbar(im0, ax=axes[0], orientation="vertical", label="Counts (ADU)")

        im1 = axes[1].imshow(model_image, origin="lower", cmap="gray")
        axes[1].set_title("Fitted Model")
        plt.colorbar(im1, ax=axes[1], orientation="vertical", label="Model (ADU)")

        resid = image_data - model_image
        im2 = axes[2].imshow(resid, origin="lower", cmap="seismic", vmin=-10, vmax=10)
        axes[2].set_title("Residual (Data - Model)")
        plt.colorbar(im2, ax=axes[2], orientation="vertical", label="Residual (ADU)")

        im3 = axes[3].imshow(bp_mask, origin="lower", cmap="gray_r")
        axes[3].set_title("Bad Pixel Mask (Thresholded)")
        plt.colorbar(im3, ax=axes[3], orientation="vertical", label="Mask")

        for ax in axes:
            ax.set_xlabel("X (px)")
            ax.set_ylabel("Y (px)")

        plt.tight_layout()
        if pdf is not None:
            pdf.savefig(fig)
        plt.show()

    return final_dict


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_profile(y, profile_y, popt, on_ymin, on_ymax,
                  off1_ymin, off1_ymax, off2_ymin, off2_ymax,
                  save_path=None):
    """Plot 1-D cross-streak profile with Gaussian model overlay."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y, profile_y, label="1D Profile", color="black")

    # Gaussian model overlay
    y_model = np.linspace(y.min(), y.max(), 500)
    ax.plot(y_model, gaussian_1d(y_model, *popt),
            color="orange", linewidth=2, label="Gaussian fit", zorder=5)

    # On-streak boundaries
    ax.axvline(on_ymin, color="red", linestyle="--", linewidth=2,
               label="On-streak", zorder=10)
    ax.axvline(on_ymax, color="red", linestyle="--", linewidth=2, zorder=10)

    # Off-streak boundaries
    ax.axvline(off1_ymin, color="blue", linestyle="--", label="Off-streak")
    ax.axvline(off1_ymax, color="blue", linestyle="--")
    ax.axvline(off2_ymin, color="blue", linestyle="--")
    ax.axvline(off2_ymax, color="blue", linestyle="--")

    ax.legend(fontsize=12)
    ax.set_xlabel("Y (pixels)", fontsize=14)
    ax.set_ylabel("Mean Counts", fontsize=14)
    ax.set_title("Streak Profile and Regions", fontsize=14)
    ax.tick_params(labelsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        log.info("Profile plot saved to %s", save_path)

    plt.show()


def _plot_mask(mask):
    """Plot the bright-source mask."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(mask, origin="lower", cmap="gray")
    ax.set_title("Masked Pixels (bright sources)")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    fig.tight_layout()
    plt.show()


def _plot_regions(image_data, region_mask):
    """Plot 2-D image with on/off-streak region overlay."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(
        image_data, origin="lower", cmap="gray",
        vmin=np.percentile(image_data, 5),
        vmax=np.percentile(image_data, 99),
    )
    ax.imshow(
        np.ma.masked_where(region_mask == 0, region_mask),
        origin="lower", cmap="coolwarm", alpha=0.5,
    )
    ax.set_title("Streak and Background Regions")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    fig.tight_layout()
    plt.show()
