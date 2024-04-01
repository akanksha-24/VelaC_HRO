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
from spectral_cube import SpectralCube
#from regions import Regions

folder = '/Users/akankshabij/Documents/MSc/Research/Data/'
CO12_cube = fits.open(folder + 'CO_LarsBonne/CO/RCW36_12CO32.fits')[0]
CO13_cube = fits.open(folder + 'CO_LarsBonne/CO/RCW36_13CO32.fits')[0]
CII_cube = fits.open(folder + 'CO_LarsBonne/CII/07_0077_RCW36_CII_L.fits')[0]
OI_cube = fits.open(folder + 'CO_LarsBonne/O/RCW36_OI_30_15.fits')[0]
HNC_cube = fits.open(folder + 'Mopra/HNC_3mm_Vela_C_T_MB.fits')[0]
N2H_cube = fits.open(folder + 'Mopra/M401_3mm_N2H+_hann2_paul.fits')[0]
C18O_cube = fits.open(folder + 'Mopra/C18O_corrHead_cube.fits')[0]
HAWC = '/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits'
Ncol = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits')[0]

Maps = [CO13_cube, CII_cube]#, #HNC_cube]
panel_label = ['$^{12}$CO', '$^{13}$CO', '[CII]', '[OI]', 'HNC', 'N$_{2}$H$^{+}$', 'C$^{18}$O']

hawc = fits.open(HAWC)[0]
ra_center = hawc.header['CRVAL1']
dec_center = hawc.header['CRVAL2']

nrows=3
ncols=3
panels=nrows*ncols

x = 0.95; dx = 0.009 ; dy = 0.1
boxes = [[x, 0.69, dx, dy], [x, 0.432, dx, dy], [x, 0.17, dx, dy]]

fig = plt.figure(figsize=(7, 7))
for i in range(len(Maps)):
    cube = SpectralCube.read(Maps[i]).with_spectral_unit(unit=u.km/u.s, velocity_convention='radio')
    blue = cube.spectral_slab(0*u.km/u.s, 5*u.km/u.s).moment0().hdu
    green = cube.spectral_slab(5*u.km/u.s, 8*u.km/u.s).moment0().hdu
    red = cube.spectral_slab(8*u.km/u.s, 11*u.km/u.s).moment0().hdu

    xcen = ra_center
    ycen = dec_center
    height = 0.15
    width = 0.16

    f1 = aplpy.FITSFigure(blue, figure=fig, subplot=(nrows,ncols,i*ncols+1))
    f1.show_colorscale(interpolation='nearest', cmap='cividis')
    f1.add_colorbar(location='top', width=0.08, pad=0, axis_label_pad=6)
    f1.recenter(xcen, ycen, height=height, width=width)
    f2 = aplpy.FITSFigure(green, figure=fig, subplot=(nrows,ncols,i*ncols+2))
    f2.show_colorscale(interpolation='nearest', cmap='cividis')
    f2.add_colorbar(location='top', width=0.08, pad=0, axis_label_pad=6)
    f2.recenter(xcen, ycen, height=height, width=width)
    f3 = aplpy.FITSFigure(red, figure=fig, subplot=(nrows,ncols,i*ncols+3))
    f3.show_colorscale(interpolation='nearest', cmap='cividis')
    f3.add_colorbar(location='top', width=0.08, pad=0, axis_label_pad=6)
    f3.recenter(xcen, ycen, height=height, width=width)

plt.show()
plt.savefig('Velocity_Grid.png')

