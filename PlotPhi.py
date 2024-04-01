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
        'size'   : 13}

plt.rc('font', **font)

HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
Ncol = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits')[0]
vecMask = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandC_polFlux_3.fits')[0]

hawc = HAWC[0]
ra_center = hawc.header['CRVAL1']
dec_center = hawc.header['CRVAL2']

def projectMap(mapOrigin, ref):
    '''This function projects a given map 'mapOrigin' onto the same WCS coordinates as a reference map and returns the map in the same shape'''
    New = ref.copy()
    proj, footprint = reproject_exact(mapOrigin, ref.header)
    proj[np.isnan(ref.data)] = np.nan
    proj[0].data = proj
    New.data = proj
    return New

def AddHeader(data, ref):
    '''This function projects a given map 'mapOrigin' onto the same WCS coordinates as a reference map and returns the map in the same shape'''
    New = ref.copy()
    #proj[np.isnan(ref.data)] = np.nan
    New.data = data
    return New

hro_CBlast = np.load('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/FWHM_3/CompareBfield/BandC_BLAST/HRO_results.npz')
phi_CBlast = AddHeader((hro_CBlast['phi']*u.rad).to(u.deg).value, HAWC[0])
meanPhi_CBlast = np.nanmean((hro_CBlast['phi']*u.rad).to(u.deg).value)
Zx_CBlast = hro_CBlast['Zx_unCorr']

hro_CE = np.load('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/FWHM_3/CompareBfield/BandC_BandE/HRO_results.npz')
phi_CE = AddHeader((hro_CE['phi']*u.rad).to(u.deg).value, HAWC[0])
meanPhi_CE = np.nanmean((hro_CE['phi']*u.rad).to(u.deg).value)
Zx_CE = hro_CE['Zx_unCorr']
print("maximum angle ", np.nanmax((hro_CE['phi']*u.rad).to(u.deg).value))
print("minimum angle ", np.nanmin((hro_CE['phi']*u.rad).to(u.deg).value))


fig = plt.figure(figsize=(8, 4.5), dpi=300)
f1 = aplpy.FITSFigure(phi_CE, figure=fig, subplot=(1,2,1))
f1.show_vectors(vecMask, HAWC[11], step=7, scale=5, units = 'degrees', color = 'white', linewidth=2) #linewidth=2.5)
f1.show_vectors(vecMask, HAWC[11], step=7, scale=5, units = 'degrees', color = 'black', linewidth=1.2) #linewidth=1.8)
f1.show_colorscale(vmax=90, vmin=0, stretch='linear', cmap='Spectral')
#f2.show_contour(Ncol, levels=[20, 50], colors='white', linewidth=1)
f1.show_contour(Ncol, levels=[15, 50], colors='black', linewidth=1,  alpha=0.6)
f1.axis_labels.set_ypad(-0.03)
f1.set_title('Band C vs. Band E', pad=10)
f1.add_label(ra_center-0.049, dec_center-0.042, 'Z\'$_{\mathrm{x}}$'+' = {0}'.format(np.round(Zx_CE, 1)), color='black', size=11)
f1.add_label(ra_center-0.048, dec_center-0.049, '<$\phi$>'+' = {0}$^\circ$'.format(np.round(meanPhi_CE, 1)), color='black', size=11)
#f1.add_colorbar(location='top', width=0.1, pad=-0.07, axis_label_pad=6, ticks=[0,30,60,90], axis_label_text='Relative Angle $\phi$ ($^{\circ}$)')

f2 = aplpy.FITSFigure(phi_CBlast, figure=fig, subplot=(1,2,2))
f2.show_vectors(vecMask, HAWC[11], step=7, scale=5, units = 'degrees', color = 'white', linewidth=2) #linewidth=2.5)
f2.show_vectors(vecMask, HAWC[11], step=7, scale=5, units = 'degrees', color = 'black', linewidth=1.2) #linewidth=1.8)
f2.show_colorscale(vmax=90, vmin=0, stretch='linear', cmap='Spectral')
#f2.show_contour(Ncol, levels=[20, 50], colors='white', linewidth=1)
f2.show_contour(Ncol, levels=[15, 50], colors='black', linewidth=1,  alpha=0.6)
f2.add_colorbar(location='right', ticks=[0,30,60,90], box=[0.9, 0.2, 0.015, 0.6], axis_label_pad=9, axis_label_text='Relative Angle $\phi$ ($^{\circ}$)')
f2.axis_labels.hide_y()
f2.tick_labels.hide_y()
f2.set_title('Band C vs. BLASTPol', pad=10)
f2.add_label(ra_center-0.049, dec_center-0.042, 'Z\'$_{\mathrm{x}}$'+' = {0}'.format(np.round(Zx_CBlast, 1)), color='black', size=11)
f2.add_label(ra_center-0.048, dec_center-0.049, '<$\phi$>'+' = {0}$^\circ$'.format(np.round(meanPhi_CBlast, 1)), color='black', size=11)
plt.subplots_adjust(wspace=0)
plt.savefig('BandC_Comparisons.pdf', bbox_inches='tight')
