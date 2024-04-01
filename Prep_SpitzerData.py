import numpy as np
from astropy.io import fits
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt
import aplpy
from reproject import reproject_interp, reproject_exact
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel

folder = '/Users/akankshabij/Documents/MSc/Research/Data/'
HAWC = fits.open(folder + 'HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')

Spitz_Ch1 = fits.open(folder + 'Spitzer/RCW36-4-selected_Post_BCDs/r15990016/ch1/pbcd/SPITZER_I1_15990016_0000_3_E8591943_maic.fits')[0]
Spitz_Ch2 = fits.open(folder + 'Spitzer/RCW36-4-selected_Post_BCDs/r15990016/ch2/pbcd/SPITZER_I2_15990016_0000_3_E8591952_maic.fits')[0]
Spitz_Ch3 = fits.open(folder + 'Spitzer/RCW36-4-selected_Post_BCDs/r15990016/ch3/pbcd/SPITZER_I3_15990016_0000_3_E8592062_maic.fits')[0]

def projectMap(mapOrigin, ref):
    '''This function projects a given map 'mapOrigin' onto the same WCS coordinates as a reference map and returns the map in the same shape'''
    New = ref.copy()
    proj, footprint = reproject_exact(mapOrigin, ref.header)
    #proj[np.isnan(ref.data)] = np.nan
    proj[0].data = proj
    New.data = proj
    return New

def header_rescale(pre_hdu, rebin):
    """Scale Header according to the rebin size"""
    hdu = pre_hdu.copy()
    naxis1, naxis2 = hdu.header['NAXIS1'], hdu.header['NAXIS2']
    hdu.header['NAXIS1'] = int(rebin * hdu.header['NAXIS1'])
    hdu.header['NAXIS2'] = int(rebin * hdu.header['NAXIS2'])
    hdu.header['CDELT1'] /= rebin
    hdu.header['CDELT2'] /= rebin
    hdu.header['CRPIX1'] = (hdu.header['CRPIX1'] / naxis1) * hdu.header['NAXIS1']
    hdu.header['CRPIX2'] = (hdu.header['CRPIX2'] / naxis2) * hdu.header['NAXIS2']
    hdu.data = np.ones((hdu.header['NAXIS1'], hdu.header['NAXIS2']))
    return hdu

## In COMMENTS of Spitzer (all channels) Header -> CDELT1 = -0.000167 /[deg/pix] , CROTA2 = -81.463580 /[deg]
## CD1_1 = CDELT1 * cos (CROTA2) (source: https://danmoser.github.io/notes/gai_fits-imgs.html)

rebin_val = HAWC[0].header['CDELT1'] / (Spitz_Ch1.header['CD1_1'] / np.cos(np.deg2rad(-81.463580)))
# rescale HAWC header to preserve Spitzer resolution
HAWC_scale = header_rescale(HAWC[0], rebin_val)

def Spitzer_prep(old_spitz, fname, size=1):
    # project Spitzer onto HAWC grid
    new_spitz = projectMap(old_spitz, HAWC_scale)
    # interpolate NAN values in Map
    new_spitz.data = interpolate_replace_nans(new_spitz.data, kernel=Gaussian2DKernel(size))
    # writeout file
    new_spitz.writeto(fname)

outfldr = '/Users/akankshabij/Documents/MSc/Research/Data/Spitzer/'
Spitzer_prep(Spitz_Ch1, fname=outfldr+'Spitz_ch1_rezmatch.fits')
Spitzer_prep(Spitz_Ch2, fname=outfldr+'Spitz_ch2_rezmatch.fits')
Spitzer_prep(Spitz_Ch3, fname=outfldr+'Spitz_ch3_rezmatch.fits')

