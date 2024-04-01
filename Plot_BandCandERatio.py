from configparser import Interpolation
from astropy.io import fits
import glob
import matplotlib.pyplot as plt
import aplpy
from astropy.coordinates import SkyCoord
import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.modeling import models
#from regions import RectangleSkyRegion
from astropy.coordinates import Angle
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from reproject import reproject_interp, reproject_exact

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

plt.rc('font', **font)

HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
HAWE = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits')
Ncol = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits')[0]
vecMask = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandC_polFlux_3.fits')[0]

def projectMap(mapOrigin, ref):
    '''This function projects a given map 'mapOrigin' onto the same WCS coordinates as a reference map and returns the map in the same shape'''
    New = ref.copy()
    proj, footprint = reproject_exact(mapOrigin, ref.header)
    proj[np.isnan(ref.data)] = np.nan
    proj[0].data = proj
    New.data = proj
    return New

HAWE_proj = projectMap(HAWE[13], HAWC[0])
ratio = HAWC[13].copy()
ratio.data = HAWC[13].data/HAWE_proj.data
ratio.data[np.isnan(vecMask.data)]=np.nan

fig = plt.figure(figsize=(5.8, 5.5), dpi=300)
f1 = aplpy.FITSFigure(ratio, figure=fig)
f1.show_vectors(vecMask, HAWC[11], step=7, scale=5, units = 'degrees', color = 'black', linewidth=3) #linewidth=2.5)
f1.show_vectors(vecMask, HAWC[11], step=7, scale=5, units = 'degrees', color = 'white', linewidth=1.5) #linewidth=1.8)
f1.show_colorscale(vmax=4, vmin=0, stretch='linear', cmap='rainbow')
#f2.show_contour(Ncol, levels=[20, 50], colors='white', linewidth=1)
f1.show_contour(Ncol, levels=[15, 50], colors='black', linewidth=1,  alpha=0.6)
#f1.show_contour(CII, levels=[350, 430], colors='black', linewidth=1,  alpha=0.6)
f1.axis_labels.set_ypad(-0.1)
#f1.set_title('Band C / Band E', pad=10, fontsize=14)
#f1.add_colorbar(location='right', box=[0.86, 0.2, 0.015, 0.5], ticks=[0,1,2,3,4], axis_label_pad=9, axis_label_text='Polarized Flux Ratio')
f1.add_colorbar(location='top', width=0.1, pad=-0.07, axis_label_pad=6, ticks=[0,1,2,3,4], axis_label_text='Band C / Band E Polarized Flux Ratio')

plt.savefig('BandC_ERatio.pdf', bbox_inches='tight')