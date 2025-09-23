import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from get_decam_data import retrieve_hdu_image


def streak_photometry(expnum, detector, image_data):
    """
    Do photometry on a bright (or faint) streak.

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

# Example to run the main function
# streak_array = np.loadtxt("sample_image.txt")  # For expnum 1103448, detector 26!
# streak_photometry (streak_array)
