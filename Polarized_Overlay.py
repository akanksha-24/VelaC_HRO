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
Ncol = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits')[0]
vecMask = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandC_polFlux_3.fits')[0]
ALMA_ACA = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/VelaC_CR_ALMA2D.fits')[0]
CII = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CII/RCW36_integratedIntensityCII.fits')[0]

fig = plt.figure(figsize=(9, 4.5), dpi=300)
f1 = aplpy.FITSFigure(HAWC[13], figure=fig, subplot=(1,2,1))
f1.show_vectors(vecMask, HAWC[11], step=7, scale=5, units = 'degrees', color = 'white', linewidth=2) #linewidth=2.5)
f1.show_vectors(vecMask, HAWC[11], step=7, scale=5, units = 'degrees', color = 'black', linewidth=1.2) #linewidth=1.8)
f1.show_colorscale(interpolation='nearest', stretch='linear', cmap='BuPu')
#f2.show_contour(Ncol, levels=[20, 50], colors='white', linewidth=1)
#f1.show_contour(Ncol, levels=[15, 50], colors='black', linewidth=1,  alpha=0.6)
#f1.show_contour(CII, levels=[250, 350, 450, 550], cmap='Greys', linewidth=1)
f1.show_contour(CII, levels=[300, 400, 450, 500], colors='Black', linewidth=1)
f1.axis_labels.set_ypad(-0.03)
f1.set_title('[CII] Comparison', pad=10)
#f1.add_colorbar(location='top', width=0.1, pad=-0.07, axis_label_pad=6, ticks=[0,30,60,90], axis_label_text='Relative Angle $\phi$ ($^{\circ}$)')

f2 = aplpy.FITSFigure(HAWC[13], figure=fig, subplot=(1,2,2))
f2.show_vectors(vecMask, HAWC[11], step=7, scale=5, units = 'degrees', color = 'white', linewidth=3) #linewidth=2.5)
f2.show_vectors(vecMask, HAWC[11], step=7, scale=5, units = 'degrees', color = 'black', linewidth=1.5) #linewidth=1.8)
f2.show_colorscale(interpolation='nearest', stretch='linear', cmap='BuPu')
#f2.show_contour(Ncol, levels=[20, 50], colors='white', linewidth=1)
#f2.show_contour(Ncol, levels=[15, 50], colors='black', linewidth=1,  alpha=0.6)
f2.show_contour(ALMA_ACA, levels=[0.018], colors='black', linewidth=1)
f2.add_colorbar(location='right', box=[0.883, 0.2, 0.015, 0.6], axis_label_pad=9, axis_label_text='Polarized Flux')
f2.axis_labels.hide_y()
f2.tick_labels.hide_y()
f2.set_title('ALMA ACA Comparison', pad=10)
plt.subplots_adjust(wspace=-0.05)
plt.savefig('Polarized_overlay.pdf', bbox_inches='tight')