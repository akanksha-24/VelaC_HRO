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
Hersh_160 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_160_flat.fits'
Hersh_70 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_70_flat.fits'
Hersh_250 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_hipe11_250.fits'
Hersh_350 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_hipe11_350_10as.fits'
Hersh_500 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_hipe11_500.fits'
spitzer_CH1 = '/Users/akankshabij/Documents/MSc/Research/Data/Spitzer/Spitz_ch2_rezmatch.fits'
Nmap = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits'
Tmap = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresTmap_velac_temperature_cf_r500_medsmo3.fits'
HNC = '/Users/akankshabij/Documents/MSc/Research/Data/Mopra/HNC_Integrated_mom0_2to10kmpers.fits'
C18O = '/Users/akankshabij/Documents/MSc/Research/Data/Mopra/C18O_Integrated_mom0_2to10kmpers.fits'
N2H = '/Users/akankshabij/Documents/MSc/Research/Data/Mopra/N2H_Integrated_mom0_5to15kmpers.fits'
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
#spitz_proj = projectMap(fits.open(spitzer_CH1)[0], H)
HNC_proj = projectMap(fits.open(HNC)[0], fits.open(Hersh_500)[0])
N2H_proj = projectMap(fits.open(N2H)[0], fits.open(Hersh_500)[0])
C18O_proj = projectMap(fits.open(C18O)[0], fits.open(Hersh_500)[0])

hawc = fits.open(HAWC)[0]
ra_center = hawc.header['CRVAL1']
dec_center = hawc.header['CRVAL2']

#data = [Hersh_500, Hersh_350, Hersh_250, Hersh_160, Hersh_70, Nmap, Tmap, HAWC, HAWE, spitzer_CH1, CO12, CO13, CII, OI]
#data = [HAWC, HAWE, spitzer_CH1, CO12, CO13, CII, OI]
data = [BLASTPol, Hersh_250, Hersh_70, Tmap, CO12, CO13, HNC_proj, N2H_proj, C18O_proj, spitzer_CH1, CII, OI] #HAWC, HAWE, ALMA_ACA, ALMA_12m]
#data = [BLASTPol, Hersh_250, Hersh_70, Tmap, HNC, N2H, C18O, spitz_proj, CII, OI]
log_format = [True, True, False, False, False, False, False, False, False, False, False, True]
vmax = [1.5e-03, 16765, 10, 32, 180, 100, 7000, 3000, 5000, 100, 500, 100000]
vmin = [-1.362e-04, -244, -0.255, 20, -5, -5, -100, -100, -20, -4, -20, -2000]
vmid = [0, -3978, -2.6]
ticks = [[0, 1e-3], [0, 1e4], [0,5,10], [20, 30], [0,100], [0, 50, 100], [0, 5000], [0, 3000],
        [0,4000], [0,50], [0, 300], [0, 1e5]]
cbar_label = ['Intensity [Jy/pixel]', 'Intensity [Jy/pixel]', 'Intensity [Jy/pixel]', 
              'Temperature [K]', 'Int. intensity [K km/s]', 'Int. intensity [K km/s]',
              'Int. intensity [K km/s]', 'Int. intensity [K km/s]', 'Int. intensity [K km/s]',
              'Intensity [MJy/sr]', 'Int. intensity [K km/s]', 'Int. intensity [K km/s]']
panel_label = ['a. BLASTPol 500 $\mu$m', 'b. Herschel 250 $\mu$m', 'c. Herschel 70 $\mu$m', 
               'd. Temperature', 'e. APEX $^{12}$CO', 'f. APEX $^{13}$CO', 
               'g. Mopra HNC', 'h. Mopra N$_{2}$H$^{+}$', 'i. Mopra C$^{18}$O', 
               'j. Spitzer 3.8$\mu$m', 'k. SOFIA [CII]', 'l. SOFIA [OI]']
beam = [2.5*u.arcmin, 18*u.arcsec, 5*u.arcsec,
        18*u.arcsec, 18.2*u.arcsec, 18.2*u.arcsec,
        36*u.arcsec, 36*u.arcsec, 33*u.arcsec,
        1.66*u.arcsec, 20*u.arcsec, 30*u.arcsec]

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 10}

plt.rc('font', **font)

# cbar_label = ['', 'Intensity (arb)', '', 
#               'Temperature (K)', 'Int. intesnity (K km/s)', 'Int. intesnity (K km/s)',
#               '', 'Int. intesnity (K km/s)', '',
#               'Intensity (arb)', 'Int. intesnity (K km/s)', 'Int. intesnity (K km/s)']


panels = len(data)
ncols = 3
nrows = 4

dx = 0.24
dy = 0.16
loc = [[0.16, 0.76], [0.37, 0.76], [0.58, 0.76], 
       [0.16, 0.55], [0.37, 0.55], [0.58, 0.55], 
       [0.16, 0.34], [0.37, 0.34], [0.58, 0.34],
       [0.16, 0.10], [0.37, 0.10], [0.58, 0.10]]

fig = plt.figure(figsize=(8, 10))#, constrained_layout=True)
# plt.xlabel('RA (J2000)')      
# plt.ylabel('Dec (J2000)') 
# ax = plt.gca()
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
      
for i in range(panels):
    f = aplpy.FITSFigure(data[i], figure=fig, subplot=[loc[i][0],loc[i][1],dx,dy])

    if i==1 or i==2:
        f.show_colorscale(vmax=vmax[i], vmin=vmin[i], vmid=vmid[i], stretch='log', cmap='cividis')
    else:
        f.show_colorscale(vmax=vmax[i], vmin=vmin[i], stretch='linear', cmap='cividis')

    if i < 9:
        xcen = ra_center
        ycen = dec_center+0.01
        height = 0.15
        width = 0.16
    else:
        xcen = ra_center
        ycen = dec_center-0.02
        ycen = dec_center
        height = 0.15/1.5
        width = 0.16/1.5

    if data[i]==spitzer_CH1 or data[i]==OI:
        label_y = (ycen+(height/3.7))
    
    else:
        label_y = (ycen+(height/2.5))

    print(xcen, ycen, height, width)    
    f.recenter(xcen, ycen, height=height, width=width)
    if data[i]==CII: 
        color='black'
    else:
        color='white'
    f.add_label((xcen+(width/1.7)), label_y, text=panel_label[i], color='black', weight=500, horizontalalignment='left')
    f.add_label((xcen+(width/1.7)), label_y, text=panel_label[i], color=color, weight=500, horizontalalignment='left')

    # if data[i] == spitzer_CH1:
    #     ticks = [0, 50]
    # elif data[i] == Tmap:
    #     ticks=[20, 25]
    # else:
    #     ticks = None
    
    f.add_colorbar(location='top', width=0.08, pad=0, axis_label_pad=6, log_format=log_format[i], ticks=ticks[i], axis_label_text=cbar_label[i])
    
    f.ticks.set_xspacing(0.08)
    if i!=10:
        f.axis_labels.hide_x()
    if i!=3:
        f.axis_labels.hide_y()
    else:
        f.axis_labels.set_ypad(0)

    if i < (panels - ncols*2):
        f.tick_labels.hide_x()
    if ((i+1) not in range(1, nrows*ncols, ncols)):
        f.tick_labels.hide_y()
    else:
        f.axis_labels.set_ypad(0)

    f.show_contour(Nmap, levels=[15, 50], colors='cyan', linewidth=0.001, linestyle=':', alpha=0.4, filled=False)
    if data[i] == spitzer_CH1:
        f.add_beam(beam[i].to(u.deg), beam[i].to(u.deg), beam[i].to(u.deg), color='white', linewidth=2.5, facecolor='none', corner='bottom left')
    else:
        f.add_beam(beam[i].to(u.deg), beam[i].to(u.deg), beam[i].to(u.deg), color='black', linewidth=2.5, facecolor='none', corner='bottom left')
        f.add_beam(beam[i].to(u.deg), beam[i].to(u.deg), beam[i].to(u.deg), color='white', linewidth=1.5, facecolor='none', corner='bottom left')

    if i==8 or i==11:
        f.add_scalebar(length=0.064, label='1 pc', corner='bottom right', color='white')

#plt.savefig('AllMaps_test.png')
plt.subplots_adjust(wspace=0)
plt.savefig('AuxData_Maps.pdf', bbox_inches='tight')