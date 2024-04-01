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

folder = '/Users/akankshabij/Documents/MSc/Research/Data/'

HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
Ncol = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits')[0]
CO12_cube = fits.open(folder + 'CO_LarsBonne/CO/RCW36_12CO32.fits')[0]
CO13_cube = fits.open(folder + 'CO_LarsBonne/CO/RCW36_13CO32.fits')[0]
CII_cube = fits.open(folder + 'CO_LarsBonne/CII/07_0077_RCW36_CII_L.fits')[0]
OI_cube = fits.open(folder + 'CO_LarsBonne/O/RCW36_OI_30_15.fits')[0]
HNC_cube = fits.open(folder + 'Mopra/HNC_3mm_Vela_C_T_MB.fits')[0]
N2H_cube = fits.open(folder + 'Mopra/M401_3mm_N2H+_hann2_paul.fits')[0]
C18O_cube = fits.open(folder + 'Mopra/C18O_corrHead_cube.fits')[0]
vecMask = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandC_polFlux_3.fits')[0]


def projectMap(mapOrigin, ref):
    '''This function projects a given map 'mapOrigin' onto the same WCS coordinates as a reference map and returns the map in the same shape'''
    New = ref.copy()
    proj, footprint = reproject_exact(mapOrigin, ref.header)
    proj[np.isnan(ref.data)] = np.nan
    proj[0].data = proj
    New.data = proj
    return New

def velocity_slice_plot(cube, vmax, vmin, title, plotname, vmid=0, log=False, interpolate=False):
    fig = plt.figure(figsize=[6,5], dpi=300)
    speed = np.linspace(4,10,7)
    nrows=2
    ncols=3
    ra_center = HAWC[0].header['CRVAL1']
    dec_center = HAWC[0].header['CRVAL2']
    for i in range(len(speed)-1):
        slab = cube.with_spectral_unit(unit=u.km/u.s, velocity_convention='radio').spectral_slab(speed[i]*u.km/u.s, speed[i+1]*u.km/u.s)
        mom0 = slab.moment0().hdu
        mom0_proj = projectMap(mom0, HAWC[0])

        f = aplpy.FITSFigure(mom0_proj, figure=fig, subplot=(nrows,ncols,i+1))
        if interpolate:
            vmax = np.nanmean(mom0_proj.data) + 2*np.nanstd(mom0_proj.data)
            vmax = np.nanmean(mom0_proj.data) - 2*np.nanstd(mom0_proj.data)
            f.show_colorscale(interpolation='nearest', cmap='BuPu')
        else:
            if log:
                f.show_colorscale(vmax=vmax, vmin=vmin, vmid=vmid, cmap='BuPu', stretch='log')
            else:
                f.show_colorscale(vmax=vmax, vmin=vmin, cmap='BuPu')
        f.show_contour(Ncol, levels=[15, 50], colors='black', linewidth=1,  alpha=0.6)
        f.show_vectors(vecMask, HAWC[11], step=8, scale=6, units = 'degrees', color = 'white', linewidth=1.5) #linewidth=2.5)
        f.show_vectors(vecMask, HAWC[11], step=8, scale=6, units = 'degrees', color = 'black', linewidth=1)
        f.add_label(ra_center+0.042, dec_center+0.045, '{0}-{1} km/s'.format(int(speed[i]), 
                    int(speed[i+1])), color='black', weight='bold', size=8)

        if i==0:
            f.add_label(ra_center+0.04, dec_center+0.086, title, size=12)
        #if interpolate:
        if i==1:
            f.add_colorbar(location='top', width=0.05, pad=0, axis_label_pad=5, axis_label_text='I [K km s$^{-1}$]')
        else:
            f.add_colorbar(location='top', width=0.05, pad=0)
        # else:
        #     if i==1:
        #         f.add_colorbar(location='top', width=0.05, pad=0, axis_label_pad=5, axis_label_text='I [K km s$^{-1}$]')
        f.ticks.set_yspacing(0.035)
        f.ticks.set_xspacing(0.06)

        if i!=0 and i!=3:
            f.axis_labels.hide_y()
            f.tick_labels.hide_y()
            #f.ticks.hide_y()
        else:
            f.axis_labels.set_ypad(0.2)
            
        if i < 3:
            f.axis_labels.hide_x()
            f.tick_labels.hide_x()
            #f.ticks.hide_x()
    #if interpolate:
    plt.subplots_adjust(wspace=0.003, hspace=0.1)
    #else:
    #plt.subplots_adjust(wspace=0.003, hspace=-0.49)
    plt.savefig(plotname, bbox_inches='tight')


cube = SpectralCube.read(CO13_cube)
vmax=25
vmin=-3
title='APEX $^{13}$CO'
plotname='13CO_VelocitySlices_interpolate.pdf'
velocity_slice_plot(cube, vmax, vmin, title, plotname, interpolate=True)

cube = SpectralCube.read(CO12_cube)
vmax=55
vmin=-3
vmid=-4
title='APEX $^{12}$CO'
plotname='12CO_VelocitySlices_interpolate.pdf'
#velocity_slice_plot(cube, vmax, vmin, title, plotname, vmid=vmid, log=True)
velocity_slice_plot(cube, vmax, vmin, title, plotname, interpolate=True)

cube = SpectralCube.read(CII_cube)
vmax=100
vmin=-3
vmid=-4
title='SOFIA [CII]'
plotname='CII_VelocitySlices_interpolate.pdf'
#velocity_slice_plot(cube, vmax, vmin, title, plotname, vmid=vmid, log=True)
#velocity_slice_plot(cube, vmax, vmin, title, plotname, interpolate=True)

cube = SpectralCube.read(OI_cube)
vmax=35
vmin=-10
vmid=35
title='SOFIA [OI]'
plotname='OI_VelocitySlices_interpolate.pdf'
velocity_slice_plot(cube, vmax, vmin, title, plotname, interpolate=False)

cube = SpectralCube.read(HNC_cube)
vmax=100
vmin=-3
vmid=-4
title='Mopra HNC'
plotname='HNC_VelocitySlices_interpolate.pdf'
#velocity_slice_plot(cube, vmax, vmin, title, plotname, interpolate=True)

cube = SpectralCube.read(N2H_cube)
vmax=1
vmin=-0.3
vmid=-4
title='Mopra N$_{2}$H$^{+}$'
plotname='N2H_VelocitySlices.pdf'
#velocity_slice_plot(cube, vmax, vmin, title, plotname, interpolate=False)

cube = SpectralCube.read(C18O_cube)
vmax=1.8
vmin=-0.2
vmid=-4
title='Mopra C$^{18}$O'
plotname='C18O_VelocitySlices.pdf'
#velocity_slice_plot(cube, vmax, vmin, title, plotname, interpolate=False)