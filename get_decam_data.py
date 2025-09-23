import requests
from astropy.io import fits
from astropy.utils.data import download_file
import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np


def retrieve_hdu_image(expnum, detector):
    natroot = 'https://astroarchive.noirlab.edu'
    adsurl = f'{natroot}/api/adv_search'

    jj = {
        "outfields" : [
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
        "search" : [
            ["instrument", "decam"],
            ["proc_type", "instcal"],
            ["EXPNUM", expnum, expnum],  # requires a range
            ["prod_type", "image"],
        ]
    }
    apiurl = f'{adsurl}/find/?limit=20'
    print(f'Using API url: {apiurl}')
    data = requests.post(apiurl, json=jj).json()
    query_result = pd.DataFrame(data[1:])  # there should be just 1 row
    md5sum = query_result['md5sum'][0]
    access_url = f'{natroot}/api/retrieve/{md5sum}/?hdus={detector}'
    print(access_url)
    filename = download_file(access_url, cache=True)
    hdu_list = fits.open(filename)
    # hdu_list.info()
    # header = hdu_list[0].header
    # image = hdu_list[1].data

    # fig = plt.figure(figsize=(10,10))
    # plt.imshow(np.flip(image.T), origin='lower',
    #            vmin = np.percentile(image,5), vmax = np.percentile(image, 95))

    # print(header)

    # There's another header with more info
    # header_image = hdu_list[1].header
    # print(header_image)

    return hdu_list
