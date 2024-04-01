from astropy.io import fits
import glob
import matplotlib.pyplot as plt
import aplpy
from astropy.coordinates import SkyCoord
import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.modeling import models
from regions import RectangleSkyRegion
from astropy.coordinates import Angle
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from reproject import reproject_interp, reproject_exact
from regions import Regions

HAWC = '/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits'
HAWE = '/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits'
ALMA_ACA = '/Users/akankshabij/Documents/MSc/Research/Data/ALMA/VelaC_CR_ALMA2D.fits'
ALMA_12m = '/Users/akankshabij/Documents/MSc/Research/Data/ALMA/Cycle8_12m/VelaC_CR1_12m_Cont_flat2D.fits'
Nmap = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits'
vecMask = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandC_polFlux_3.fits')[0]

files = [HAWC, HAWE, ALMA_ACA, ALMA_12m]

nrows=2
ncols=2
panels=len(files)

ra_center = fits.open(HAWC)[0].header['CRVAL1']
dec_center = fits.open(HAWC)[0].header['CRVAL2']

cbar_label = ['HAWC+ 89 $\mu$m [Jy/pixel]', 'HAWC+ 214 $\mu$m [Jy/pixel]', 'ALMA ACA [Jy/beam]', 'ALMA 12m [Jy/beam]']
titles = ['a. SOFIA/HAWC+ 89 $\mu$m', 'b. SOFIA/HAWC+ 214 $\mu$m', 'ALMA ACA 1.1-1.4 mm', 'd. ALMA 12m 1.1-1.4 mm']
beam = [7.8*u.arcsec, 18.2*u.arcsec, 5.4*u.arcsec, 1.4*u.arcsec]
vmax=[7, 13, 0.08, 0.009]
vmin=[-0.1, -0.5, -0.02, -0.002]
ticks=[[0,2,4,6], [0,5,10], [0, 0.05], [0, 0.004, 0.008]]

# font = {'family' : 'normal',
#         'weight' : 'normal',
#         'size'   : 11}

# plt.rc('font', **font)


fig = plt.figure(figsize=(8, 8))

for i in range(panels):
    data = fits.open(files[i])[0]
    f = aplpy.FITSFigure(data, figure=fig, subplot=(nrows, ncols, i+1))
    if i==0:
        f.recenter(ra_center, dec_center, height=0.11, width=0.115)
    #f.show_colorscale(interpolation='nearest', cmap='cividis')
    f.show_colorscale(vmax=vmax[i], vmin=vmin[i], cmap='cividis')
    f.add_colorbar(location='top', width=0.08, pad=0, axis_label_pad=8, axis_label_text=cbar_label[i], ticks=ticks[i])
    f.show_contour(Nmap, levels=[15, 50], colors='cyan', alpha=1, filled=False)
    f.add_beam(beam[i].to(u.deg), beam[i].to(u.deg), beam[i].to(u.deg), color='black', linewidth=1.5, facecolor='none', corner='bottom left')
    f.add_scalebar(length=0.064/4, label='0.25 pc', corner='bottom right', color='black')
    # if i < 2:
    #     f.show_vectors(vecMask, fits.open(HAWC)[11], step=6, scale=4, units = 'degrees', color = 'black', linewidth=1.6) #linewidth=2.5)
    #     f.show_vectors(vecMask, fits.open(HAWC)[11], step=6, scale=4, units = 'degrees', color = 'white', linewidth=1.1) #linewidth=1.8)
    # else:
    #     f.show_vectors(vecMask, fits.open(HAWC)[11], step=6, scale=4, units = 'degrees', color = 'white', linewidth=1.6) #linewidth=2.5)
    #     f.show_vectors(vecMask, fits.open(HAWC)[11], step=6, scale=4, units = 'degrees', color = 'black', linewidth=1.1) #linewidth=1.8)
    f.axis_labels.set_ypad(-0.1)

plt.subplots_adjust(hspace=0.38)

plt.savefig('New_Data.pdf', bbox_inches='tight')

