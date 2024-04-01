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

Maps = [CO12_cube, CO13_cube, CII_cube, OI_cube, HNC_cube, N2H_cube, C18O_cube]
panel_label = ['$^{12}$CO', '$^{13}$CO', '[CII]', '[OI]', 'HNC', 'N$_{2}$H$^{+}$', 'C$^{18}$O']

hawc = fits.open(HAWC)[0]
ra_center = hawc.header['CRVAL1']
dec_center = hawc.header['CRVAL2']

def projectMap(mapOrigin, ref):
    '''This function projects a given map 'mapOrigin' onto the same WCS coordinates as a reference map and returns the map in the same shape'''
    New = ref.copy()
    proj, footprint = reproject_exact(mapOrigin, ref.header)
    #proj[np.isnan(ref.data)] = np.nan
    proj[0].data = proj
    New.data = proj
    return New

nrows=3
ncols=3
panels=nrows*ncols

x = 0.95; dx = 0.009 ; dy = 0.1
boxes = [[x, 0.69, dx, dy], [x, 0.432, dx, dy], [x, 0.17, dx, dy]]

fig = plt.figure(figsize=(7, 7))
for i in range(len(Maps)):
    cube = SpectralCube.read(Maps[i]).with_spectral_unit(unit=u.km/u.s, velocity_convention='radio')
    blue = cube.spectral_slab(5*u.km/u.s, 7*u.km/u.s).moment0().hdu#.writeto('blue.fits', overwrite=True)
    green = cube.spectral_slab(7*u.km/u.s, 9*u.km/u.s).moment0().hdu#.writeto('green.fits', overwrite=True)
    red = cube.spectral_slab(9*u.km/u.s, 11*u.km/u.s).moment0().hdu#.writeto('red.fits', overwrite=True)

    projectMap(blue, hawc).writeto('blue.fits', overwrite=True)
    projectMap(green, hawc).writeto('green.fits', overwrite=True)
    projectMap(red, hawc).writeto('red.fits', overwrite=True)

    aplpy.make_rgb_cube(['red.fits', 'green.fits', 'blue.fits'], 'RGB_velocity.fits')
    aplpy.make_rgb_image('RGB_velocity.fits','RGB_velocity.png')

    f = aplpy.FITSFigure('RGB_velocity.png', figure=fig, subplot=(nrows,ncols,i+1))
    f.show_rgb()
    xcen = ra_center
    ycen = dec_center+0.01
    height = 0.15
    width = 0.16
    #f.recenter(xcen, ycen, height=height, width=width)
    f.ticks.set_xspacing(0.06)
    f.show_contour(Ncol, levels=[15, 50], colors='white', linewidth=1, alpha=0.6)

    #if i==0:

    if i<=3:
        f.axis_labels.hide_x()
        f.tick_labels.hide_x()
    if i!=0 and i!=3 and i!=6:
        f.axis_labels.hide_y()
        f.tick_labels.hide_y()
    else:
        f.axis_labels.set_ypad(0.3)

    # if Maps[i]==CII_cube or Maps[i]==OI_cube: 
    #     color='black'
    # else:
    color='white'
    #label_y = (ycen+(0.2)
    #f.add_label((xcen+), label_y, text=panel_label[i], color='black', weight=510, horizontalalignment='left', size=12)
    f.add_label(xcen+0.065, ycen+0.021, text=panel_label[i], color=color, weight=540, horizontalalignment='left', size=12)

    if i==5:
        f.add_label(xcen, ycen-0.11, text='5-7 km/s', color='Blue', weight=570, horizontalalignment='center', size=12)
        f.add_label(xcen, ycen-0.13, text='7-9 km/s', color='Green', weight=570, horizontalalignment='center', size=12)
        f.add_label(xcen, ycen-0.15, text='9-11 km/s', color='Red', weight=570, horizontalalignment='center', size=12)

plt.subplots_adjust(wspace=-0.015, hspace=0.03)
#plt.show()
plt.savefig('RGB_velcityGrid.pdf', bbox_inches='tight')
