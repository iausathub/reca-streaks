import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from get_decam_data import retrieve_hdu_image


def streak_photometry(expnum, detector, image_data):
    """
    Do aperture photometry on a bright (or faint) streak.

    Parameters
    ----------
    expnum : `int`
        DECam exposure number, used to get the correct header
    detector : `int`
        DECam detector number, used to get the correct header
    image_data : `np.array`
        2D array of a small portion of the given expnum and detector,
        which has already been determined to contain a streak

    The function prints out the results of the photometry to the screen.
    """
    hdu_list = retrieve_hdu_image(expnum, detector)
    header = hdu_list[1].header
    header_expnum = hdu_list[0].header
    try:
        zeropoint = header_expnum['MAGZERO']
    except KeyError:
        print("Photometric zeropoint not available in image header, reporting instrumental flux only.")
        zeropoint = None

    # Estimate average gain and read noise from two amplifiers
    gain = 0.5 * (header["GAINA"] + header["GAINB"])
    read_noise = 0.5 * (header["RDNOISEA"] + header["RDNOISEB"])

    # --- Step 1: Mask bright sources ---
    median = np.median(image_data)
    std = np.std(image_data)
    threshold = median + 3 * std
    mask = image_data > threshold
    masked_data = np.ma.array(image_data, mask=mask)

    # --- Step 2: Collapse image to 1D profile across y ---
    profile_y = np.ma.mean(masked_data, axis=1)
    y = np.arange(len(profile_y))

    # --- Step 3: Fit a 1D Gaussian to the profile ---
    def gaussian(y, A, y0, sigma, offset):
        return A * np.exp(-(y - y0)**2 / (2 * sigma**2)) + offset

    A0 = profile_y.max()
    y0 = y[np.argmax(profile_y)]
    sigma0 = 3
    offset0 = np.median(profile_y)
    p0 = [A0, y0, sigma0, offset0]
    popt, _ = curve_fit(gaussian, y, profile_y.filled(offset0), p0=p0)
    A_fit, y0_fit, sigma_fit, offset_fit = popt
    fwhm = 2.355 * sigma_fit

    # --- Step 4: Define on-streak and off-streak regions ---
    on_ymin = int(y0_fit - 3 * sigma_fit)
    on_ymax = int(y0_fit + 3 * sigma_fit)
    height = on_ymax - on_ymin

    off1_ymin = max(0, on_ymin - height)
    off1_ymax = on_ymin
    off2_ymin = on_ymax
    off2_ymax = min(image_data.shape[0], on_ymax + height)

    region_mask = np.zeros_like(image_data, dtype=int)
    region_mask[on_ymin:on_ymax, :] = 1
    region_mask[off1_ymin:off1_ymax, :] = 2
    region_mask[off2_ymin:off2_ymax, :] = 2

    # --- Step 5: Extract pixel values ---
    on_region = masked_data[on_ymin:on_ymax, :]
    off1_region = masked_data[off1_ymin:off1_ymax, :]
    off2_region = masked_data[off2_ymin:off2_ymax, :]

    on_sum = on_region.sum()
    on_unmasked_pixels = np.sum(~on_region.mask)

    print("on_sum", on_sum)
    print("on_unmasked_pixels", on_unmasked_pixels)

    # --- Step 6: Estimate background ---
    off_pixels = np.sum(~off1_region.mask) + np.sum(~off2_region.mask)
    off_sum = off1_region.sum() + off2_region.sum()
    bkg_per_pixel = off_sum / off_pixels

    print("Number of pixel sin background region off streak: ", off_pixels)

    empirical_bkg = bkg_per_pixel * on_unmasked_pixels

    # --- Step 7: Final flux and surface brightness ---
    streak_flux = on_sum - empirical_bkg

    pixel_scale = 0.27

    sb_pixel = streak_flux / on_unmasked_pixels
    sb_arcsec = sb_pixel / (pixel_scale ** 2)

    # --- Step 8: Error estimation ---
    def estimate_flux_uncertainty(streak_flux, on_unmasked, bkg_std, gain, read_noise):
        """
        Parameters
        ----------
        streak_flux: `float`
            background-subtracted streak total flux after masking (ADU).
        on_unmasked: `int`
            Number of pixels used to calculate streak flux.
        bkd_std: `float`
            Standard deviation of flux in background region (after masking).
        gain: `float`
            Gain in e/ADU
        read_noise: `float`
            Read noise in e.

        Returns
        -------
        flux_err : `float`
            Error in the streak flux (ADU)
        regime: `str`
            Flux or background dominated.
        """
        flux_e = gain * streak_flux
        bkg_std_e = gain * bkg_std

        signal_per_pixel = flux_e / on_unmasked
        noise_background_var = on_unmasked * (bkg_std_e**2 + read_noise**2)

        if signal_per_pixel < (5 * bkg_std_e):
            regime = "background-dominated"
            flux_var = noise_background_var
        else:
            regime = "source-dominated"
            flux_var = flux_e + noise_background_var

        print(f"regime: {regime}")
        print(f"Streak flux in electrons {flux_e}")
        print(f"Noise background var {noise_background_var}")
        print(f"Noise background std dev {np.sqrt(noise_background_var)}")
        print(f"Streak flux error in electrons {np.sqrt(flux_var)}")

        flux_err = np.sqrt(flux_var) / gain  # back to ADU

        return flux_err, regime

    off_vals = np.hstack([
        off1_region[~off1_region.mask].ravel(),
        off2_region[~off2_region.mask].ravel()
    ])

    bkg_std = np.std(off_vals)

    streak_flux_err, regime = estimate_flux_uncertainty(streak_flux,
                                                        on_unmasked_pixels,
                                                        bkg_std,
                                                        gain,
                                                        read_noise)

    print(f"Streak Flux error: {streak_flux_err} (ADU)")

    sb_pixel_err = streak_flux_err / on_unmasked_pixels
    sb_arcsec_err = sb_pixel_err / pixel_scale**2

    # --- Output ---
    print("\n=== Aperture Photometry Result ===")
    print(f"Streak center (y0): {y0_fit:.2f} px")
    print(f"Width: σ = {sigma_fit:.2f} px, FWHM ≈ {fwhm:.2f} px")
    print(f"On-streak region: y = [{on_ymin}, {on_ymax}], {on_unmasked_pixels} unmasked pixels")
    print(f"Streak flux: {streak_flux:.2f}")
    print(f"Surface brightness: {sb_arcsec:.2f} ± {sb_arcsec_err:.2f} counts/arcsec² [{regime}]")

    # --- Final Output Summary ---
    print("\n=== Aperture Photometry Result ===")
    print(f"Streak center (y0): {y0_fit:.2f} px")
    print(f"Width: σ = {sigma_fit:.2f} px, FWHM ≈ {fwhm:.2f} px")
    print(f"On-streak region: y = [{on_ymin}, {on_ymax}], {on_unmasked_pixels} unmasked pixels")
    print(f"Off-streak regions: y = [{off1_ymin}, {off1_ymax}] and [{off2_ymin}, {off2_ymax}]")
    print(f"Signal sum (masked): {on_sum:.2f}")
    print(f"Background level: {bkg_per_pixel:.2f} counts/pixel")
    print(f"Empirical background: {empirical_bkg:.2f}")
    print(f"Streak flux (signal - background): {streak_flux:.2f}")
    print(f"Noise regime: {regime}")

    print(f"Surface brightness: {sb_pixel:.2f} ± {sb_pixel_err:.2f} counts/pixel²")
    print(f"Surface brightness: {sb_arcsec:.2f} ± {sb_arcsec_err:.2f} counts/arcsec²")

    # Convert to magnitudes using flux zeropoint
    if zeropoint is not None:
        sb_mag_arcsec2 = -2.5 * np.log10(sb_arcsec) + zeropoint
        sb_mag_arcsec2_err = 1.0857 * (sb_arcsec_err / sb_arcsec)  # double check this
        print(f"(Using photometric zeropoint {zeropoint:.2f} to convert to magnitude)")
        print(f"Surface brightness: {sb_mag_arcsec2:.3f} ± {sb_mag_arcsec2_err:.3f} mag/arcsec²")

    # --- Plots ---
    plt.figure(figsize=(10, 6))
    plt.imshow(mask, origin='lower', cmap='gray')
    plt.title("Masked Pixels (bright sources)")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.colorbar(label='Mask (True = masked)')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.imshow(image_data, origin='lower', cmap='gray',
               vmin=np.percentile(image_data, 5),
               vmax=np.percentile(image_data, 99))
    plt.imshow(np.ma.masked_where(region_mask == 0, region_mask),
               origin='lower', cmap='coolwarm', alpha=0.5)
    plt.title("Streak and Background Regions")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.colorbar(label='Region: 0=none, 1=on-streak, 2=off-streak')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(y, profile_y, label="1D Profile", color='black')
    plt.axvline(on_ymin, color='red', linestyle='--', label='On-streak')
    plt.axvline(on_ymax, color='red', linestyle='--')
    plt.axvline(off1_ymin, color='blue', linestyle='--', label='Off-streak')
    plt.axvline(off1_ymax, color='blue', linestyle='--')
    plt.axvline(off2_ymin, color='blue', linestyle='--')
    plt.axvline(off2_ymax, color='blue', linestyle='--')
    plt.legend()
    plt.xlabel("Y (pixels)")
    plt.ylabel("Mean Counts")
    plt.title("Streak Profile and Regions")
    plt.tight_layout()
    plt.show()


def streak_photometry_psf_fitting (expnum, detector, image_data, pdf=None):
    """
    Do fitting photometry on a bright (or faint) streak.
    Model: equation 3 ofVeres+12
    "Improved Asteroid Astrometry and Photometry with Trail Fitting"
    https://iopscience.iop.org/article/10.1086/668616/pdf

    Parameters
    ----------
    expnum : `int`
        DECam exposure number, used to get the correct header
    detector : `int`
        DECam detector number, used to get the correct header
    image_data : `np.array`
        2D array of a small portion of the given expnum and detector,
        which has already been determined to contain a streak

    The function prints out the results of the photometry to the screen.

    Returns
    -------
    final_dict: `dict`
        Dictionary with results and photometry.
    """

    hdu_list = retrieve_hdu_image(expnum, detector)
    header = hdu_list[1].header
    header_two = hdu_list[0].header
    
    gain = 0.5 * (header["GAINA"] + header["GAINB"])
    read_noise = 0.5 * (header["RDNOISEA"] + header["RDNOISEB"])
    print ("average amp gain (e/ADU)", gain)
    print ("average read noise (e)", read_noise)

    if 'MAGZERO' in header_two:
        zero_point = header_expnum['MAGZERO']
        print ("zero point [mag]: ", zero_point)
    else:
        print ("No MAGZERO keyword in header")
        zero_point = None
    
    # Build a simple bad pixel mask
    sigma_bad_pixel_mask = 3
    median = np.median(image_data)
    std = np.std(image_data)
    threshold = median + sigma_bad_pixel_mask * std
    bp_mask = image_data > threshold
    
    
    ny, nx = image_data.shape
    x_grid, y_grid = np.meshgrid(np.arange(nx), np.arange(ny))
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    z_flat = image_data.ravel()
    mask_flat = bp_mask.ravel()
    
    # Apply mask: exclude masked pixels from fitting
    x_flat = x_flat[~mask_flat]
    y_flat = y_flat[~mask_flat]
    z_flat = z_flat[~mask_flat]
    
    
    # === Trail model: θ is free ===
    def trail_model(coords, b, phi, L, sigma, theta, x0, y0):
        x, y = coords
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        x_rot = (x - x0) * sin_t + (y - y0) * cos_t
        Q = (x - x0) * cos_t - (y - y0) * sin_t
        term1 = b
        term2 = (phi / L) * (1 / (2 * sigma * np.sqrt(2 * np.pi)))
        term3 = np.exp(- (x_rot ** 2) / (2 * sigma ** 2))
        erf1 = (Q + L/2) / (sigma * np.sqrt(2))
        erf2 = (Q - L/2) / (sigma * np.sqrt(2))
        return term1 + term2 * term3 * (erf(erf1) - erf(erf2))
    
    # === Initial guess and bounds ===
    b0 = np.median(image_data)
    phi0 = np.sum(image_data) - b0 * image_data.size
    L0 = 2000
    sigma0 = 1.0
    theta0 = 0.0
    x0_0 = nx / 2
    y0_0 = ny / 2
    p0 = [b0, phi0, L0, sigma0, theta0, x0_0, y0_0]
    
    bounds = ([0, 0, 5, 0.3, -np.pi, 0, 0], [1e9, 1e9, 1e4, 10, np.pi, nx-1, ny-1])
    
    # === Fit ===
    popt, pcov = curve_fit(trail_model, (x_flat, y_flat), z_flat, p0=p0, bounds=bounds, maxfev=5000)
    b_fit, phi_fit, L_fit, sigma_fit, theta_fit, x0_fit, y0_fit = popt
    phi_err = np.sqrt(np.diag(pcov))[1]
    
    # === Reconstruct model ===
    model_image = trail_model((x_grid, y_grid), *popt)
    
    # Approximate trail aperture mask using rotated bounding box
    coords_x = x_grid - x0_fit
    coords_y = y_grid - y0_fit
    cos_t = np.cos(theta_fit)
    sin_t = np.sin(theta_fit)
    q = coords_x * cos_t + coords_y * sin_t   # along-trail
    r = coords_x * sin_t - coords_y * cos_t   # cross-trail
    
    # Another mask, for the chi2
    mask = ((np.abs(q) < length / 2) & (np.abs(r) < width / 2)) & ~bp_mask
    ap_data = image_data[mask]
    ap_model = model_image[mask]
    ap_N = np.sum(mask)
    
    # === Chi² and reduced Chi² ===
    noise_var = (gain * ap_model + read_noise**2) / gain**2
    chi2 = np.sum((ap_data - ap_model)**2 / noise_var)
    chi2_red = chi2 / (ap_N - 7 - 1)
    
    # === S/N from section 3 near Eq. 8: S / sqrt(S + B)
    S = phi_fit
    B = b_fit * ap_N
    SNR = S / np.sqrt(S + B)
    
    # === Surface brightness ===
    # area_arcsec2 = ap_N * pixel_scale**2
    psf_fwhm_arcsec = 2.355 * sigma_fit * pixel_scale
    area_arcsec2 = L_fit * pixel_scale * psf_fwhm_arcsec
    
    sb_arcsec2 = S / area_arcsec2
    sb_arcsec2_err = phi_err / area_arcsec2
    
    # === Print results ===
    print("\n=== Trail Fit Results (θ free) ===")
    print(f"Background: {b_fit:.2f} counts/pixel")
    print(f"Total flux (phi): {phi_fit:.2f} ± {phi_err:.2f} ADU")
    print(f"Trail length (L): {L_fit:.2f} px")
    print(f"PSF sigma: {sigma_fit:.2f} px → FWHM = {2.355 * sigma_fit:.2f} px --> {2.355 * sigma_fit*pixel_scale:.2f} arcsec ")
    print(f"Trail angle θ: {np.degrees(theta_fit):.2f} deg")
    print(f"Center: x0 = {x0_fit:.2f}, y0 = {y0_fit:.2f}")
    print(f"Surface brightness: {sb_arcsec2:.2f} ± {sb_arcsec2_err:.2f} counts/arcsec²")
    print(f"S/N (section 3): {SNR:.1f}")
    print(f"Chi²: {chi2:.2f}")
    print(f"Reduced Chi²: {chi2_red:.4f}")

    if zero_point: 
        sb_mag_arcsec2 = zero_point - 2.5 * np.log10(sb_arcsec2)
        # Approximate the error as the inverse of SNR
        sb_mag_arcsec2_err = 1.0857 / SNR
    
        # Output
        print(f"\nSurface brightness (mag/arcsec²): {sb_mag_arcsec2:.2f} ± {sb_mag_arcsec2_err:.4f}")
        print(f"Signal-to-noise ratio (SNR): {SNR:.1f}")
    else:
        print ("Image with no zero point, can't convert to magnitude")
        sb_mag_arcsec2, sb_mag_arcsec2_err = None, None


    final_dict = {
    "SB_counts": sb_arcsec2,
    "SB_counts_err": sb_arcsec2_err,
    "SB_mag": sb_mag_arcsec2 if sb_mag_arcsec2 else "No zeropoint",
    "SB_mag_err": sb_mag_arcsec2_err if sb_mag_arcsec2_err else "—",
    "Zeropoint": zero_point if zero_point else "No zeropoint",
    "Background": b_fit,
    "Flux_total": phi_fit,
    "Flux_err": phi_err,
    "Trail_Length_px": L_fit,
    "PSF_Sigma_px": sigma_fit,
    "PSF_FWHM_arcsec": 2.355 * sigma_fit * pixel_scale,
    "Trail_angle_deg": np.degrees(theta_fit),
    "Center_x0": x0_fit,
    "Center_y0": y0_fit,
    "SNR": SNR,
    "Chi2": chi2,
    "Chi2_red": chi2_red
    }
    
    
    # === Plots ===
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))  # 4 rows
    
    # Observed image
    im0 = axes[0].imshow(image_data, origin='lower', cmap='viridis',
                         vmin=np.percentile(image_data, 5),
                         vmax=np.percentile(image_data, 95))
    axes[0].set_title("Observed Image")
    plt.colorbar(im0, ax=axes[0], orientation='vertical', label='Counts (ADU)')
    
    # Fitted model
    im1 = axes[1].imshow(model_image, origin='lower', cmap='gray')
    axes[1].set_title("Fitted Model")
    plt.colorbar(im1, ax=axes[1], orientation='vertical', label='Model (ADU)')
    
    # Residuals
    residuals = image_data - model_image
    im2 = axes[2].imshow(residuals, origin='lower', cmap='seismic', vmin=-10, vmax=10)
    axes[2].set_title("Residual (Data - Model)")
    plt.colorbar(im2, ax=axes[2], orientation='vertical', label='Residual (ADU)')
    
    # Bad pixel mask
    im3 = axes[3].imshow(bp_mask, origin='lower', cmap='gray_r')
    axes[3].set_title("Bad Pixel Mask (Thresholded)")
    plt.colorbar(im3, ax=axes[3], orientation='vertical', label='Mask (1=bad, 0=good)')
    
    # Common axes labels
    for ax in axes:
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")

    if pdf is not None:
        plt.tight_layout()
        pdf.savefig(fig)
        plt.tight_layout()
    else:
        plt.tight_layout()
        plt.tight_layout()

    return final_dict
        
