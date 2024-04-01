import numpy as np
import aplpy
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from reproject import reproject_interp, reproject_exact
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as c
import PIL
from spectral_cube import SpectralCube

PIL.Image.MAX_IMAGE_PIXELS = 933120000

# vectors
HAWC = '/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits'
HAWE = '/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits'
vecMask_C = '/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandC_polFlux_3.fits' 
vecMask_E = '/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandE_polFlux_3.fits'
BlastPol = '/Users/akankshabij/Documents/MSc/Research/Data/BlastPOL/BLASTPol_500_intermediate_BPOSang.fits'

# maps
CO12_cube = '/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_12CO32.fits'
CO13_cube = '/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_13CO32.fits'
CII_cube = '/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CII/07_0077_RCW36_CII_L.fits'
ALMA_CO = '/Users/akankshabij/Documents/MSc/Research/Data/ALMA/CO_Cube.fits'
CO12 = '/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_integratedIntensity12CO.fits' 
CO13 = '/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_integratedIntensity13CO.fits' 
CII = '/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CII/RCW36_integratedIntensityCII.fits'
ALMA = '/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Data/Observations/ALMA_Band6_continuum.fits'
spitzer_CH1 = '/Users/akankshabij/Documents/MSc/Research/Data/Spitzer/RCW36-4-selected_Post_BCDs/r15990016/ch1/pbcd/SPITZER_I1_15990016_0000_3_E8591943_maic.fits'
spitzer_CH2 = '/Users/akankshabij/Documents/MSc/Research/Data/Spitzer/RCW36-4-selected_Post_BCDs/r15990016/ch2/pbcd/SPITZER_I2_15990016_0000_3_E8591952_maic.fits'
spitzer_CH3 = '/Users/akankshabij/Documents/MSc/Research/Data/Spitzer/RCW36-4-selected_Post_BCDs/r15990016/ch3/pbcd/SPITZER_I3_15990016_0000_3_E8592062_maic.fits'
spitzer_CH4 = '/Users/akankshabij/Documents/MSc/Research/Data/Spitzer/RCW36-4-selected_Post_BCDs/r15990016/ch4/pbcd/SPITZER_I4_15990016_0000_3_E8592077_maic.fits'
colDen = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits'
Mopra = '/Users/akankshabij/Documents/MSc/Research/Data/Mopra/Integrated_mom0_0to12kmpers.fits'
Tmap = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresTmap_velac_temperature_cf_r500_medsmo3.fits'
Hersh_160 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_160_flat.fits'
Hersh_70 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_70_flat.fits'
Hersh_250 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_hipe11_250.fits'
Hersh_350 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_hipe11_350_10as.fits'
Hersh_500 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_hipe11_500.fits'
# Halph = '/Users/akankshabij/Documents/MSc/Research/Data/ha83701_full.fits'
# Hal = fits.open(Halph)[0]
# header = Hal.header
# del header[8:172]
# del header[20:30]
# Hal.writeto('/Users/akankshabij/Documents/MSc/Research/Data/Halpha_corrHead.fits')
Halph = '/Users/akankshabij/Documents/MSc/Research/Data/Halpha_corrHead.fits'

BlastMask = fits.open(BlastPol)[0].copy()
BlastMask.data = np.ones(fits.open(BlastPol)[0].data.shape)

#stars 
OB9 = SkyCoord('8 59 27.34 -43 45 25.84', unit=(u.hourangle, u.deg))
OB9_7 = SkyCoord('8 59 27.55 -43 45 28.41', unit=(u.hourangle, u.deg))

matplotlib.rcParams.update({'font.size': 23, 'axes.ymargin': 0.5})

# **** Large-scale
#aplpy.make_rgb_cube([Hersh_160, Hersh_70, Halph], 'cube_L.fits')
# aplpy.make_rgb_image('cube_L.fits','cube_L.png', 
#                      stretch_r='linear',stretch_g='log',stretch_b='linear',
#                      vmax_r=11, vmin_r=-0.5, vmin_g=0, vmid_g=-3, vmax_g=35, vmin_b=2500, vmax_b=14000)

# fig = plt.figure(figsize=[10,8], dpi=300)
# f = aplpy.FITSFigure('cube_L.png', figure=fig)
# f.show_rgb()
# f.recenter(134.87,-43.74, height=0.19, width=0.28)
# # f.show_vectors(BlastMask, fits.open(BlastPol)[0], step=10, scale =4, units='radians', color = 'black', linewidth=4)
# # f.show_vectors(BlastMask, fits.open(BlastPol)[0], step=10, scale =4, units='radians', color = 'white', linewidth=2)
# #f.show_vectors(BlastMask, fits.open(BlastPol)[0], step=8, scale =5, units='radians', color = 'black', linewidth=1)
# #f.show_vectors(fits.open(vecMask_E)[0], fits.open(HAWE)[11], step=3, scale=2, units='degrees', color='white', linewidth=2)
# #f.show_vectors(fits.open(vecMask_E)[0], fits.open(HAWE)[11], step=3, scale=2, units='degrees', color='black', linewidth=1)
# f.show_contour(colDen, levels=[15, 50], colors='cyan', linewidth=1.5)
# #f.show_markers([OB9.ra.value, OB9_7.ra.value], [OB9.dec.value, OB9_7.dec.value], coords_frame='world', marker="*", s=2**8.7, c='white')
# f.show_markers([OB9.ra.value], [OB9.dec.value], coords_frame='world', marker="*", s=2**8.6, c='white')
# f.axis_labels.set_ypad(-1)
# f.add_scalebar(3.8197186/60.)
# f.scalebar.set_label('1 parcsec')
# f.scalebar.set_color('white')
# f.scalebar.set_font_size(15)
# f.scalebar.set_linewidth(3)
# plt.savefig('RGBcube_L.pdf', bbox_inches='tight')

matplotlib.rcParams.update({'font.size': 20, 'axes.ymargin': 0.5})
# ***** Mid-scale
# aplpy.make_rgb_cube([CO12, CO13, CII], 'cube_M.fits')
# aplpy.make_rgb_image('cube_M.fits','cube_M.png', 
#                      stretch_r='linear',stretch_g='linear',stretch_b='linear')

# fig = plt.figure(figsize=[5,6], dpi=300)
# f = aplpy.FITSFigure('cube_M.png', figure=fig)
# f.show_rgb()
# f.recenter(134.874,-43.75, height=0.15, width=0.13)
# # f.show_vectors(fits.open(vecMask_E)[0], fits.open(HAWE)[11], step=5, scale=4, units='degrees', color = 'black', linewidth=3)
# # f.show_vectors(fits.open(vecMask_E)[0], fits.open(HAWE)[11], step=5, scale=4, units='degrees', color = 'white', linewidth=1.5)
# #f.show_vectors(fits.open(vecMask_C)[0], fits.open(HAWC)[11], step=5, scale=4, units='degrees', color='white', linewidth=1.5)
# f.show_contour(colDen, levels=[15, 50], colors='cyan', linewidth=1.5)
# f.show_contour(ALMA, levels=[0.018], colors='black')
# f.show_markers([OB9.ra.value], [OB9.dec.value], coords_frame='world', marker="*", s=2**8.6, c='white')
# #f.axis_labels.set_ypad(-1)
# f.axis_labels.hide_y()
# #f.show_markers([OB9.ra.value, OB9_7.ra.value], [OB9.dec.value, OB9_7.dec.value], coords_frame='world', marker="*", s=2**8, c='magenta')
# f.add_scalebar(1.9098593/60.)
# f.scalebar.set_label('0.5 parcsec')
# f.scalebar.set_color('white')
# f.scalebar.set_font_size(15) 
# f.scalebar.set_linewidth(3)
# plt.savefig('RGBcube_M.pdf', bbox_inches='tight')

# **** Small Scale
#fits.open(HAWE)[0].writeto('HAWE_fl.fits')
# aplpy.make_rgb_cube([ALMA, spitzer_CH2, 'HAWE_fl.fits'], 'cube_S.fits')
# aplpy.make_rgb_image('cube_S.fits','cube_S.png', 
#                      stretch_r='linear',stretch_g='linear',stretch_b='linear',
#                      vmin_r=-0.005,vmin_g=-1,
#                      vmax_r=0.04,vmax_g=90)

# fig = plt.figure(figsize=[10,8], dpi=300)
# f = aplpy.FITSFigure('cube_S.png', figure=fig)
# f.show_rgb()
# f.recenter(134.87,-43.755, height=0.09, width=0.08)
# f.show_vectors(fits.open(vecMask_C)[0], fits.open(HAWC)[11], step=5, scale =4, units='degrees', color = 'White', linewidth=2.5)
# f.show_vectors(fits.open(vecMask_C)[0], fits.open(HAWC)[11], step=5, scale=4, units='degrees', color='Black', linewidth=1)
# f.show_contour(colDen, levels=[20, 50, 100], colors='cyan', linewidth=1.5)
# #f.show_contour(HAWC[13], levels=[np.nanmax(HAWC[13].data)*0.3, np.nanmax(HAWC[13].data)*0.4, np.nanmax(HAWC[13].data)*0.5, np.nanmax(HAWC[13].data)*0.6], colors='cyan')
# f.show_markers([OB9.ra.value, OB9_7.ra.value], [OB9.dec.value, OB9_7.dec.value], coords_frame='world', marker="*", s=2**8, c='yellow')
# f.add_scalebar(0.38197186/60.)
# f.scalebar.set_label('0.1 parcsec')
# f.scalebar.set_color('white')
# f.scalebar.set_font_size(15)
# f.scalebar.set_linewidth(3)
# plt.savefig('RGBcube_S.png')

# aplpy.make_rgb_cube([ALMA, spitzer_CH1, CO13], 'cube_XS_13CO.fits')
# aplpy.make_rgb_image('cube_XS_13CO.fits','cube_XS_13CO.png', 
#                      stretch_r='linear',stretch_g='linear',stretch_b='linear',
#                      vmin_r=-0.005,vmin_g=-5, vmin_b=-5,
#                      vmax_r=0.04,vmax_g=130, vmax_b=150)

# fig = plt.figure(figsize=[10,8], dpi=300)
# f = aplpy.FITSFigure('cube_XS_13CO.png', figure=fig)
# f.show_rgb()
# f.recenter(134.87,-43.761, height=0.068, width=0.075)
# f.show_vectors(fits.open(vecMask_C)[0], fits.open(HAWC)[11], step=5, scale =4, units='degrees', color = 'White', linewidth=1.5)
# f.show_vectors(fits.open(vecMask_C)[0], fits.open(HAWC)[11], step=5, scale=4, units='degrees', color='Black', linewidth=0.9)
# f.show_contour(colDen, levels=[20, 50, 100], colors='lightgray')
# #f.show_contour(HAWC[13], levels=[np.nanmax(HAWC[13].data)*0.3, np.nanmax(HAWC[13].data)*0.4, np.nanmax(HAWC[13].data)*0.5, np.nanmax(HAWC[13].data)*0.6], colors='cyan')
# f.show_markers([OB9.ra.value, OB9_7.ra.value], [OB9.dec.value, OB9_7.dec.value], coords_frame='world', marker="*", s=2**8, c='yellow')
# plt.savefig('RGBcube_XS_13CO.png')

# aplpy.make_rgb_cube([ALMA, spitzer_CH1, CO12], 'cube_XS_12CO.fits')
# aplpy.make_rgb_image('cube_XS_12CO.fits','cube_XS_12CO.png', 
#                      stretch_r='linear',stretch_g='linear',stretch_b='linear',
#                      vmin_r=-0.005,vmin_g=-5, vmin_b=10,
#                      vmax_r=0.04,vmax_g=130, vmax_b=220)

fig = plt.figure(figsize=[5,6], dpi=300)
f = aplpy.FITSFigure('cube_XS_12CO.png', figure=fig)
f.show_rgb()
f.recenter(134.862,-43.761, height=0.068, width=0.06)
#f.recenter(134.87,-43.74, height=0.15, width=0.15)
# f.show_vectors(fits.open(vecMask_C)[0], fits.open(HAWC)[11], step=6, scale=4, units='degrees', color='Black', linewidth=2)
# f.show_vectors(fits.open(vecMask_C)[0], fits.open(HAWC)[11], step=6, scale =4, units='degrees', color = 'White', linewidth=1.5)
f.show_contour(colDen, levels=[15, 50], colors='cyan', linewidth=1.5)
#f.show_contour(HAWC[13], levels=[np.nanmax(HAWC[13].data)*0.3, np.nanmax(HAWC[13].data)*0.4, np.nanmax(HAWC[13].data)*0.5, np.nanmax(HAWC[13].data)*0.6], colors='cyan')
#f.show_markers([OB9.ra.value, OB9_7.ra.value], [OB9.dec.value, OB9_7.dec.value], coords_frame='world', marker="*", s=2**8, c='yellow')
f.show_markers([OB9.ra.value], [OB9.dec.value], coords_frame='world', marker="*", s=2**8.6, c='white')
f.add_scalebar(0.38197186/60.)
f.scalebar.set_label('0.1 parcsec')
f.scalebar.set_color('white')
f.scalebar.set_font_size(15)
f.scalebar.set_linewidth(3)
# f.axis_labels.set_ypad(-1)
f.axis_labels.hide_y()
plt.savefig('RGBcube_XS_12CO.pdf', bbox_inches='tight')


# aplpy.make_rgb_cube([ALMA, CO13, CO12], 'cube_M_1213CO.fits')
# aplpy.make_rgb_image('cube_M_1213CO.fits','cube_M_1213CO.png', 
#                      stretch_r='linear',stretch_g='linear',stretch_b='linear',
#                      vmin_r=-0.005,vmin_b=10,
#                      vmax_r=0.04,vmax_b=220)

# fig = plt.figure(figsize=[10,8], dpi=300)
# f = aplpy.FITSFigure('cube_M_1213CO.png', figure=fig)
# f.show_rgb()
# #f.recenter(134.862,-43.761, height=0.068, width=0.06)
# f.recenter(134.87,-43.74, height=0.15, width=0.15)
# f.show_vectors(fits.open(vecMask_C)[0], fits.open(HAWC)[11], step=5, scale =4, units='degrees', color = 'White', linewidth=2)
# f.show_vectors(fits.open(vecMask_C)[0], fits.open(HAWC)[11], step=5, scale=4, units='degrees', color='Black', linewidth=1)
# f.show_contour(colDen, levels=[15, 50], colors='cyan', linewidth=1.5)
# #f.show_contour(HAWC[13], levels=[np.nanmax(HAWC[13].data)*0.3, np.nanmax(HAWC[13].data)*0.4, np.nanmax(HAWC[13].data)*0.5, np.nanmax(HAWC[13].data)*0.6], colors='cyan')
# #f.show_markers([OB9.ra.value, OB9_7.ra.value], [OB9.dec.value, OB9_7.dec.value], coords_frame='world', marker="*", s=2**8, c='yellow')
# f.show_markers([OB9.ra.value], [OB9.dec.value], coords_frame='world', marker="*", s=2**8.6, c='white')
# f.add_scalebar(0.38197186/60.)
# f.scalebar.set_label('0.1 parcsec')
# f.scalebar.set_color('white')
# f.scalebar.set_font_size(15)
# f.scalebar.set_linewidth(3)
# f.axis_labels.set_ypad(-1)
# plt.savefig('RGBcube_XS_12CO.png')
# plt.savefig('RGBcube_M_1213CO.png')


# ******** Cube 

# data = fits.open(ALMA_CO)[0]
# cube_s = SpectralCube.read(data)
# cube = cube_s.with_spectral_unit(unit=u.km/u.s, velocity_convention='optical')
# slab_B = cube.spectral_slab(-3*u.km/u.s, 6*u.km/u.s)
# slab_G = cube.spectral_slab(6*u.km/u.s, 9*u.km/u.s)
# slab_R = cube.spectral_slab(9*u.km/u.s, 17*u.km/u.s)

# mom0_B = slab_B.moment0()
# mom0_G = slab_G.moment0()
# mom0_R = slab_R.moment0()

# mom0_B.hdu.writeto('ALMACO_B.fits')
# mom0_G.hdu.writeto('ALMACO_G.fits')
# mom0_R.hdu.writeto('ALMACO_R.fits')

# aplpy.make_rgb_cube(['ALMACO_R.fits', 'ALMACO_G.fits', 'ALMACO_B.fits'], '12CO_speeds.fits')
# aplpy.make_rgb_image('12CO_speeds.fits','ALMACO_speeds.png', 
#                      stretch_r='linear',stretch_g='linear',stretch_b='linear')

# fig = plt.figure(figsize=[10,8], dpi=300)
# f = aplpy.FITSFigure('ALMACO_speeds.png', figure=fig)
# f.show_rgb()
# f.recenter(134.87,-43.74, height=0.15, width=0.15)
# #f.recenter(134.87,-43.74, height=0.19, width=0.28)
# #f.show_vectors(BlastMask, fits.open(BlastPol)[0], step=7, scale =4, units='radians', color = 'gray', linewidth=0.8)
# f.show_vectors(fits.open(vecMask_E)[0], fits.open(HAWE)[11], step=5, scale=3, units='degrees', color='cyan', linewidth=1.5)
# f.show_contour(colDen, levels=[20, 50, 100], colors='lightgray')
# f.show_markers([OB9.ra.value, OB9_7.ra.value], [OB9.dec.value, OB9_7.dec.value], coords_frame='world', marker="*", s=2**8, c='yellow')
# plt.savefig('ALMACO_speeds.png')