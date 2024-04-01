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
#from regions import Regions

prefix = '/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/FWHM_3/'

#location = ['BandC/Hersh_70/', 'BandC/Hersh_160/', 'BandC/Hersh_250/', 'BandC/Hersh_350/', 'BandC/Hersh_500/']
location = ['BandC/Spitz_CH1/',  'BandC/CII/', 'BandC/OI/']

nrows=3
ncols=3
panels=nrows*ncols

HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
Hersh_70 = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_70_flat.fits')[0]
Hersh_160 = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_160_flat.fits')[0]
Hersh_250 = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_hipe11_250.fits')[0]
#Hersh_350 = fits.oepn('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_hipe11_350_10as.fits')[0]
Ncol = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits')[0]
Hersh_500 = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_hipe11_500.fits')[0]
CO12 = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_integratedIntensity12CO.fits')[0]
CO13 = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_integratedIntensity13CO.fits')[0]
CII = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CII/RCW36_integratedIntensityCII.fits')[0]
OI = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/O/RCW36_OI_Integrated.fits')[0]
HNC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Mopra/HNC_Integrated_mom0_2to10kmpers.fits')[0]
C18O = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Mopra/C18O_Integrated_mom0_2to10kmpers.fits')[0]
N2H = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Mopra/N2H_Integrated_mom0_5to15kmpers.fits')[0]
spitzer_CH1 = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Spitzer/Spitz_ch1_rezmatch.fits')[0]

Maps = [spitzer_CH1, CII, OI]
vecMask = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandC_polFlux_3.fits')[0]

def projectMap(mapOrigin, ref, Nan=True):
    '''This function projects a given map 'mapOrigin' onto the same WCS coordinates as a reference map and returns the map in the same shape'''
    New = ref.copy()
    proj, footprint = reproject_exact(mapOrigin, ref.header)
    if Nan:
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

x = 0.047; dx = 0.009 ; dy = 0.1
boxes = [[x, 0.69, dx, dy], [x, 0.432, dx, dy], [x, 0.17, dx, dy]]
ticks = [[0, 60], [0,400], [0, 1e5]]
log_format = [False, False, True]
Vmax = [80, 500, 100000]
Vmin = [-4, -50, -20000]
Vmid = [10, -3978, -2.6]
Map_labels = ['Spizter 3.6 $\mu$m', 'SOFIA [CII]', 'SOFIA [OI]']
Ncol_HAWC = projectMap(Ncol, HAWC[0], Nan=False)

fig = plt.figure(figsize=(7, 7))
for i in range(len(location)):
    hro = np.load(prefix+location[i]+'HRO_results.npz', allow_pickle=True)
    mapProj = projectMap(Maps[i], HAWC[0])
    
    f1 = aplpy.FITSFigure(mapProj, figure=fig, subplot=(nrows,ncols,i*nrows+1))
    
    f1.show_colorscale(vmax=Vmax[i], vmin=Vmin[i], cmap='BuPu')
    f1.show_contour(Ncol, levels=[20, 50], colors='black', linewidth=1, alpha=0.6)
    f1.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'white', linewidth=1.6) #linewidth=2.5)
    f1.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'black', linewidth=1.1) #linewidth=1.8)
    #f1.show_contour(Ncol, levels=[20, 50], colors='white', linewidth=0.5)
    #f1.ticks.set_tick_direction('in')#, log_format=True)
    # if i==0:
    #     f1.add_colorbar(box=boxes[i], axis_label_pad=3, axis_label_text='Jy/pixel', ticks=ticks[i], log)
    # else:
    f1.add_colorbar(box=boxes[i], ticks=ticks[i], log_format=log_format[i])#, axis_label_text='Jy/pixel')
    f1.ticks.set_yspacing(0.04)
    if i==0:
        f1.add_label(2, 2, 'Jy/pixel', horizontalalignment='center', size=11)
    #colorbar.set_box(box=, box_orientation='vertical')
    
    if i==0:
        phi = AddHeader((hro['phi']*u.rad).to(u.deg).value, spitzer_CH1)
    else:
        phi = AddHeader((hro['phi']*u.rad).to(u.deg).value, HAWC[0])
    
    f2 = aplpy.FITSFigure(phi, figure=fig, subplot=(nrows,ncols,i*nrows+2))
    f2.show_colorscale(vmax=90, vmin=0, stretch='linear', cmap='Spectral')
    #f2.show_contour(Ncol, levels=[20, 50], colors='white', linewidth=1)
    f2.show_contour(Ncol, levels=[20, 50], colors='black', linewidth=1,  alpha=0.6)
    f2.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'white', linewidth=1.6) #linewidth=2.5)
    f2.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'black', linewidth=1.1) #linewidth=1.8)
    f2.axis_labels.hide_y()
    f2.tick_labels.hide_y()

    if (i!=1):
        f1.axis_labels.hide_y()

    if (i!=2):
        f1.axis_labels.hide_x()
        f1.tick_labels.hide_x()
        f2.axis_labels.hide_x()
        f2.tick_labels.hide_x()

    ax = plt.subplot(nrows,ncols,i*nrows+3)
    angle = np.linspace(0, 90, hro['histbins'])
    plt.plot(angle, hro['hist'], linewidth=1.4, color='black')
    plt.xlim(0,90)
    plt.ylim(0, 1.7)
    asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
    print("aspect is ", asp)
    ax.set_aspect(asp)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_yticks([0.5, 1.0, 1.5])
    ax.text(10, 1.4, Map_labels[i])
    ax.text(10, 1.2, 'Z$_{\mathrm{x}}$'+' = {0}'.format(np.round(hro['Zx_corr'], 1)))
    ax.text(10, 1.0, 'Z$_{\mathrm{x}}\'$'+' = {0}'.format(np.round(hro['Zx_unCorr'], 1)))
    if (i!=2):
        ax.set_xticks([0, 0.5, 1, 1.5])
    if (i==2):
        ax.set_xticks([0, 30, 60, 90])
        plt.xlabel('Relative Angle $\phi$ ($^{\circ}$)')
    if (i==1):
        plt.ylabel('Histogram density', labelpad=15)
        # f2.ticks.hide_y()
        # f2.ticks.hide_x()
    if i==0:
        #f1.add_colorbar(location='top', width=0.1, pad=-0.07, axis_label_pad=6, log_format=True, axis_label_text='Intensity (Jy/pixel)')
        f2.add_colorbar(location='top', width=0.1, pad=-0.07, axis_label_pad=6, ticks=[0,30,60,90], axis_label_text='Relative Angle $\phi$ ($^{\circ}$)')

#plt.tight_layout()
plt.subplots_adjust(wspace=-0.07, hspace=-0.01)
plt.savefig('Results_PDR.pdf')
#plt.show()