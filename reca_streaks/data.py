"""Retrieve DECam images from the NOIRLab Astro Data Archive."""

import logging

import pandas as pd
import requests
from astropy.io import fits
from astropy.io.fits import HDUList
from astropy.utils.data import download_file

__all__ = ["retrieve_hdu_image"]

log = logging.getLogger(__name__)

NATROOT = "https://astroarchive.noirlab.edu"


def retrieve_hdu_image(expnum: int, detector: int) -> HDUList:
    """Download a single-HDU DECam image from the NOIRLab archive.

    Parameters
    ----------
    expnum : int
        DECam exposure number.
    detector : int
        CCD detector number (1–62).

    Returns
    -------
    hdu_list : `~astropy.io.fits.HDUList`
        FITS HDU list for the requested exposure and detector.

    Raises
    ------
    ValueError
        If ``expnum`` or ``detector`` are not positive integers.
    RuntimeError
        If the archive query returns no results.
    """
    if not isinstance(expnum, int) or expnum <= 0:
        raise ValueError(f"expnum must be a positive integer, got {expnum!r}")
    if not isinstance(detector, int) or detector <= 0:
        raise ValueError(f"detector must be a positive integer, got {detector!r}")

    adsurl = f"{NATROOT}/api/adv_search"
    query = {
        "outfields": [
            "md5sum",
            "archive_filename",
            "dateobs_center",
            "dateobs_min",
            "dateobs_max",
            "proc_type",
            "prod_type",
            "obs_type",
            "release_date",
            "proposal",
            "caldat",
            "EXPNUM",
        ],
        "search": [
            ["instrument", "decam"],
            ["proc_type", "instcal"],
            ["EXPNUM", expnum, expnum],
            ["prod_type", "image"],
        ],
    }

    apiurl = f"{adsurl}/find/?limit=20"
    log.info("Querying NOIRLab archive: %s", apiurl)
    data = requests.post(apiurl, json=query).json()

    if len(data) < 2:
        raise RuntimeError(
            f"No results returned for expnum={expnum}, detector={detector}"
        )

    query_result = pd.DataFrame(data[1:])
    md5sum = query_result["md5sum"][0]
    access_url = f"{NATROOT}/api/retrieve/{md5sum}/?hdus={detector}"
    log.info("Downloading HDU: %s", access_url)

    filename = download_file(access_url, cache=True)
    hdu_list = fits.open(filename)

    return hdu_list
