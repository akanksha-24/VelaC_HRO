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
import matplotlib.gridspec as gridspec

HAWC = '/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits'
HAWE = '/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits'
Hersh_160 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_160_flat.fits'
Hersh_70 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_70_flat.fits'
Hersh_250 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_hipe11_250.fits'
Hersh_350 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_hipe11_350_10as.fits'
Hersh_500 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_hipe11_500.fits'
spitzer_CH1 = '/Users/akankshabij/Documents/MSc/Research/Data/Spitzer/RCW36-4-selected_Post_BCDs/r15990016/ch1/pbcd/SPITZER_I1_15990016_0000_3_E8591943_maic.fits'
Nmap = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits'
Tmap = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresTmap_velac_temperature_cf_r500_medsmo3.fits'
HNC = '/Users/akankshabij/Documents/MSc/Research/Data/Cutouts/Mopra_C18O.fits'
C18O = '/Users/akankshabij/Documents/MSc/Research/Data/Cutouts/Mopra_HNC.fits'
N2H = '/Users/akankshabij/Documents/MSc/Research/Data/Cutouts/Mopra_N2H.fits'
CO12 = '/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_integratedIntensity12CO.fits'
CO13 = '/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_integratedIntensity13CO.fits'
CII = '/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CII/RCW36_integratedIntensityCII.fits'
OI = '/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/O/RCW36_OI_Integrated.fits'
Halpha = '/Users/akankshabij/Documents/MSc/Research/Data/Halpha/Halpha_corrHead.fits'
ALMA_ACA = '/Users/akankshabij/Documents/MSc/Research/Data/ALMA/VelaC_CR_ALMA2D.fits'
ALMA_12m = '/Users/akankshabij/Documents/MSc/Research/Data/ALMA/Cycle8_12m/VelaC_CR1_12m_Cont_flat2D.fits'
BLASTPol = '/Users/akankshabij/Documents/MSc/Research/Data/BlastPOL/BLASTPol_500_intermediate_I.fits'

def projectMap(mapOrigin, ref):
    '''This function projects a given map 'mapOrigin' onto the same WCS coordinates as a reference map and returns the map in the same shape'''
    New = ref.copy()
    proj, footprint = reproject_exact(mapOrigin, ref.header)
    #proj[np.isnan(ref.data)] = np.nan
    proj[0].data = proj
    New.data = proj
    return New

H = fits.open(Hersh_70)[0]
spitz_proj = projectMap(fits.open(spitzer_CH1)[0], H)

hawc = fits.open(HAWC)[0]
ra_center = hawc.header['CRVAL1']
dec_center = hawc.header['CRVAL2']

#data = [Hersh_500, Hersh_350, Hersh_250, Hersh_160, Hersh_70, Nmap, Tmap, HAWC, HAWE, spitzer_CH1, CO12, CO13, CII, OI]
#data = [HAWC, HAWE, spitzer_CH1, CO12, CO13, CII, OI]
data = [Hersh_250, Hersh_70, Tmap, BLASTPol, CO12, CO13, HNC, N2H, C18O, spitz_proj, CII, OI] #HAWC, HAWE, ALMA_ACA, ALMA_12m]

panels = len(data)
ncols = 3
nrows = 4

fig = plt.figure(figsize=(7, 10), constrained_layout=True)

for i in range(panels):
    f = aplpy.FITSFigure(data[i], figure=fig, subplot=(nrows,ncols,i+1))
    if data[i] == spitz_proj:
        dat = data[i].data
    else:
        dat = fits.open(data[i])[0].data
    if i < 2:
        print(i)
        vmax = np.nanmean(dat) + 80*np.nanstd(dat)
        print(vmax)
        vmin = np.nanmean(dat) - 2*np.nanstd(dat)
        print(vmin)
        vmid = np.nanmean(dat) - 20*np.nanstd(dat)
        print(vmid)
        f.show_colorscale(vmax=vmax, vmin=vmin, vmid=vmid, stretch='log', cmap='cividis')
    elif i==2:
        f.show_colorscale(vmax=30, vmin=18, cmap='cividis')
    else:
        #vmax = np.nanmean(dat) + np.nanstd(dat)
        #vmin = np.nanmean(dat) - np.nanstd(dat)
        #f.show_colorscale(vmax=vmax, vmin=vmin, cmap='viridis');
        f.show_colorscale(interpolation='nearest', cmap='cividis')
    cbar = f.add_colorbar(location='top', width=0.08, pad=0, axis_label_pad=0, log_format='True')

    
    #f.show_contour(Nmap, levels=[15,30,100], colors='white')
    f.recenter(ra_center,dec_center+0.01, height=0.15, width=0.17)

    #if (i not in range(panels-ncols, panels)):
    if i < (panels - ncols):
        f.tick_labels.hide_x()
        f.axis_labels.hide_x()
    if ((i+1) not in range(1, nrows*ncols, ncols)):
        f.tick_labels.hide_y()
        f.axis_labels.hide_y()

#plt.tight_layout()
#plt.subplots_adjust(wspace=0, hspace=0.3)
plt.savefig()