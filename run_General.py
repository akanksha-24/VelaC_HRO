import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp, reproject_exact
import HRO as h
import plots as p
import matplotlib.pyplot as plt
import aplpy
from spectral_cube import SpectralCube
import astropy.units as u
import astropy.constants as c
import os
from spectral_cube import SpectralCube
import imageio
import matplotlib.backends.backend_pdf as pdf

# HAWC = '/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits'
# HAWE = '/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits'
ALMA = '/Users/akankshabij/Documents/MSc/Research/Data/ALMA/VelaC_CR_ALMA2D.fits'
#ALMA = 'Data/Observations/ALMA_Band6_continuum.fits'
# ALMA_13CS = '/Users/akankshabij/Documents/MSc/Research/Data/ALMA/13CS_Cube.fits'
Hersch = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits')[0]
HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
vecMask = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandC_polFlux_3.fits')[0]

def runAll():
    Hersch = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits')[0]
    HAWC = fits.open(HAWC)
    HAWE = fits.open(HAWE)
    Hersch_C = projectMap(Hersch, HAWC[0])

    Imap_CC, Imap_CE = runBoth('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits', Mask=None, outName='HAWC_BandC/', project=True, label='HAWC+ Band C')#, BinMap=Hersch_C)  
    Imap_EC, Imap_EE = runBoth('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits', Mask=None, outName='HAWC_BandE/', project=True, label='HAWC+ Band E')#, BinMap=Hersch_C)  

    Mopra_C, Mopra_E = runBoth('/Users/akankshabij/Documents/MSc/Research/Data/Mopra/Integrated_mom0_0to12kmpers.fits', Mask=None, outName='Mopra/', kstep=5, index=0, label='Mopra HNC')#, BinMap=Hersch_C)            
    Nmap_C, Nmap_E = runBoth('Data/Observations/Herschel_ColumnDensity.fits', Mask=None, outName='ColumnDensity/', kstep=5, index=1, label='Herschel Column Density')#, BinMap=Hersch_C)        
    Tmap_C, Tmap_E = runBoth('Data/Observations/Hershel_Temperature.fits',    Mask=None, outName='Temperature/',   kstep=5, label='Herschel Temperature')#, BinMap=Hersch_C)   

    #mask = fits.open('/Users/akankshabij/Documents/MSc/Research/Plots/Data_masks/HerschelColDensity_Highres/mask.fits')[0]
    highres_Nmap_C, highres_Nmap_E = runBoth('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits', Mask=None, outName='Highres_ColumnDensity/', kstep=5, label='Highres Herschel Column Density')#, BinMap=Hersch_C)        
    highres_Tmap_C, highres_Tmap_E = runBoth('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresTmap_velac_temperature_cf_r500_medsmo3.fits', Mask=None, outName='Highres_Temperature/', kstep=5, label='Highres Herschel Temperature')#, BinMap=Hersch_C)                 

    Hersh_160 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_160_flat.fits'
    Hersh_70 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_70_flat.fits'
    Hersh_250 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_hipe11_250.fits'
    Hersh_350 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_hipe11_350_10as.fits'
    Hersh_500 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_hipe11_500.fits'

    runBoth(Hersh_160, Mask=None, outName='Hersh_160/', kstep=3, label='Herschel 160 micron')#, BinMap=Hersch_C) 
    runBoth(Hersh_70, Mask=None, outName='Hersh_70/', kstep=3, label='Herschel 70 micron')
    runBoth(Hersh_250, Mask=None, outName='Hersh_250/', kstep=3, label='Herschel 250 micron')
    runBoth(Hersh_350, Mask=None, outName='Hersh_350/', kstep=3, label='Herschel 350 micron')
    runBoth(Hersh_500, Mask=None, outName='Hersh_500/', kstep=3, label='Herschel 500 micron')

    #mask = fits.open('/Users/akankshabij/Documents/MSc/Research/Plots/Data_masks/CO12/mask.fits')[0]
    CO12_C, CO12_E = runBoth('Data/Observations/12CO_Integrated.fits', Mask=None, outName='12CO/', kstep=2, label='12CO(3-2)')#, BinMap=Hersch_C)                 
    #mask = fits.open('/Users/akankshabij/Documents/MSc/Research/Plots/Data_masks/CO13/mask.fits')[0]
    CO13_C, CO13_E = runBoth('Data/Observations/13CO_Integrated.fits', Mask=None, outName='13CO/', kstep=2, label='13CO(3-2)')#, BinMap=Hersch_C)                 
    #mask = fits.open('/Users/akankshabij/Documents/MSc/Research/Plots/Data_masks/CII/mask.fits')[0]
    CII_C, CII_E  = runBoth('Data/Observations/CII_Integrated.fits', Mask=None, outName='CII/',  kstep=1, label='[CII]')#, BinMap=Hersch_C) 

    mask = fits.open('/Users/akankshabij/Documents/MSc/Research/Plots/Data_masks/ALMA/mask.fits')[0]
    alma = fits.open('Data/Observations/ALMA_Band6_continuum.fits')[0]
    #Hersch_C = projectMap(Hersch, alma)
    ALMA_C, ALMA_E = runBoth('Data/Observations/ALMA_Band6_continuum.fits',   Mask=mask, outName='ALMA/', label='ALMA', project=False, highRes=True)#, BinMap=Hersch_C) 

    spitzer_CH1 = '/Users/akankshabij/Documents/MSc/Research/Data/Spitzer/RCW36-4-selected_Post_BCDs/r15990016/ch1/pbcd/SPITZER_I1_15990016_0000_3_E8591943_maic.fits'
    spitzer_CH2 = '/Users/akankshabij/Documents/MSc/Research/Data/Spitzer/RCW36-4-selected_Post_BCDs/r15990016/ch2/pbcd/SPITZER_I2_15990016_0000_3_E8591952_maic.fits'
    spitzer_CH3 = '/Users/akankshabij/Documents/MSc/Research/Data/Spitzer/RCW36-4-selected_Post_BCDs/r15990016/ch3/pbcd/SPITZER_I3_15990016_0000_3_E8592062_maic.fits'
    spitzer_CH4 = '/Users/akankshabij/Documents/MSc/Research/Data/Spitzer/RCW36-4-selected_Post_BCDs/r15990016/ch4/pbcd/SPITZER_I4_15990016_0000_3_E8592077_maic.fits'

    runBoth(spitzer_CH1, Mask=None, outName='Spitzer_CH1/', kstep=1, label='Spitzer CH1')#, BinMap=Hersch_C) 
    runBoth(spitzer_CH2, Mask=None, outName='Spitzer_CH2/', kstep=1, label='Spitzer CH2')
    runBoth(spitzer_CH3, Mask=None, outName='Spitzer_CH3/', kstep=1, label='Spitzer CH3')
    runBoth(spitzer_CH4, Mask=None, outName='Spitzer_CH4/', kstep=1, label='Spitzer CH4')

    # print('\n\n ******* Projected Rayleigh Statistic Results *********')
    # print('*** Note: Zx >> 0 gives statiscally parallel alignment, Zx << 0 gives statistically perpendicular alignment ***')
    # print('Band C')
    # print('For HAWC+ 79 micron Intensity, Zx=', Imap_CC.Zx)
    # print('For HAWC+ 214 micron Intensity, Zx=', Imap_CE.Zx)
    # print('For Mopra, HNC, Zx=', Mopra_C.Zx)
    # print('For Herscehl Column Density, Zx=', Nmap_C.Zx)
    # print('For Herschel Temperature, Zx=', Tmap_C.Zx)
    # print('For highres Herscehl Column Density, Zx=', highres_Nmap_C.Zx)
    # print('For highres Herschel Temperature, Zx=', highres_Tmap_C.Zx)
    # print('For 12CO(3-2) integrated line intensity, Zx=', CO12_C.Zx)
    # print('For 13CO(3-2) integrated line intensity, Zx=', CO13_C.Zx)
    # print('For CII integrated line intensity, Zx=', CII_C.Zx)
    # print('For ALMA 1mm continuum, Zx=', ALMA_C.Zx)
    # print('\n\n')
    # print('Band E')
    # print('For HAWC+ 79 micron Intensity, Zx=', Imap_EC.Zx)
    # print('For HAWC+ 214 micron Intensity, Zx=', Imap_EE.Zx)
    # print('For Mopra, HNC, Zx=', Mopra_E.Zx)
    # print('For Herscehl Column Density, Zx=', Nmap_E.Zx)
    # print('For Herschel Temperature, Zx=', Tmap_E.Zx)
    # print('For highres Herscehl Column Density, Zx=', highres_Nmap_E.Zx)
    # print('For highres Herschel Temperature, Zx=', highres_Tmap_E.Zx)
    # print('For 12CO(3-2) integrated line intensity, Zx=', CO12_E.Zx)
    # print('For 13CO(3-2) integrated line intensity, Zx=', CO13_E.Zx)
    # print('For CII integrated line intensity, Zx=', CII_E.Zx)
    # print('For ALMA 1mm continuum, Zx=', ALMA_E.Zx)
    # print('\n\n')

def runBinHersch():
    Hersch = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits')[0]
    HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
    HAWE = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits')
    Hersch_C = projectMap(Hersch, HAWC[0])
    Hersch_E = projectMap(Hersch, HAWE[0])

    Imap_CC, Imap_CE = runBoth('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits', Mask=None, outName='HAWC_BandC/', project=True, label='HAWC+ Band C', BinMap=Hersch_C, BinMapE=Hersch_E)  
    Imap_EC, Imap_EE = runBoth('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits', Mask=None, outName='HAWC_BandE/', project=True, label='HAWC+ Band E', BinMap=Hersch_C, BinMapE=Hersch_E)  

    Mopra_C, Mopra_E = runBoth('/Users/akankshabij/Documents/MSc/Research/Data/Mopra/Integrated_mom0_0to12kmpers.fits', Mask=None, outName='Mopra/', kstep=5, index=0, label='Mopra HNC', BinMap=Hersch_C, BinMapE=Hersch_E)            
    Nmap_C, Nmap_E = runBoth('Data/Observations/Herschel_ColumnDensity.fits', Mask=None, outName='ColumnDensity/', kstep=5, index=1, label='Herschel Column Density', BinMap=Hersch_C, BinMapE=Hersch_E)        
    Tmap_C, Tmap_E = runBoth('Data/Observations/Hershel_Temperature.fits',    Mask=None, outName='Temperature/',   kstep=5, label='Herschel Temperature', BinMap=Hersch_C, BinMapE=Hersch_E)   

    #mask = fits.open('/Users/akankshabij/Documents/MSc/Research/Plots/Data_masks/HerschelColDensity_Highres/mask.fits')[0]
    highres_Nmap_C, highres_Nmap_E = runBoth('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits', Mask=None, outName='Highres_ColumnDensity/', kstep=5, label='Highres Herschel Column Density', BinMap=Hersch_C, BinMapE=Hersch_E)        
    highres_Tmap_C, highres_Tmap_E = runBoth('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresTmap_velac_temperature_cf_r500_medsmo3.fits', Mask=None, outName='Highres_Temperature/', kstep=5, label='Highres Herschel Temperature', BinMap=Hersch_C, BinMapE=Hersch_E)                 

    Hersh_160 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_160_flat.fits'
    Hersh_70 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_70_flat.fits'
    Hersh_250 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_hipe11_250.fits'
    Hersh_350 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_hipe11_350_10as.fits'
    Hersh_500 = '/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_hipe11_500.fits'

    runBoth(Hersh_160, Mask=None, outName='Hersh_160/', kstep=3, label='Herschel 160 micron', BinMap=Hersch_C, BinMapE=Hersch_E) 
    runBoth(Hersh_70, Mask=None, outName='Hersh_70/', kstep=3, label='Herschel 70 micron', BinMap=Hersch_C, BinMapE=Hersch_E)
    runBoth(Hersh_250, Mask=None, outName='Hersh_250/', kstep=3, label='Herschel 250 micron', BinMap=Hersch_C, BinMapE=Hersch_E)
    runBoth(Hersh_350, Mask=None, outName='Hersh_350/', kstep=3, label='Herschel 350 micron', BinMap=Hersch_C, BinMapE=Hersch_E)
    runBoth(Hersh_500, Mask=None, outName='Hersh_500/', kstep=3, label='Herschel 500 micron', BinMap=Hersch_C, BinMapE=Hersch_E)

    #mask = fits.open('/Users/akankshabij/Documents/MSc/Research/Plots/Data_masks/CO12/mask.fits')[0]
    CO12_C, CO12_E = runBoth('Data/Observations/12CO_Integrated.fits', Mask=None, outName='12CO/', kstep=2, label='12CO(3-2)', BinMap=Hersch_C, BinMapE=Hersch_E)                 
    #mask = fits.open('/Users/akankshabij/Documents/MSc/Research/Plots/Data_masks/CO13/mask.fits')[0]
    CO13_C, CO13_E = runBoth('Data/Observations/13CO_Integrated.fits', Mask=None, outName='13CO/', kstep=2, label='13CO(3-2)', BinMap=Hersch_C, BinMapE=Hersch_E)                 
    #mask = fits.open('/Users/akankshabij/Documents/MSc/Research/Plots/Data_masks/CII/mask.fits')[0]
    CII_C, CII_E  = runBoth('Data/Observations/CII_Integrated.fits', Mask=None, outName='CII/',  kstep=2, label='[CII]', BinMap=Hersch_C, BinMapE=Hersch_E) 

    mask = fits.open('/Users/akankshabij/Documents/MSc/Research/Plots/Data_masks/ALMA/mask.fits')[0]
    alma = fits.open('Data/Observations/ALMA_Band6_continuum.fits')[0]
    #Hersch_C = projectMap(Hersch, alma)
    ALMA_C, ALMA_E = runBoth('Data/Observations/ALMA_Band6_continuum.fits',   Mask=mask, outName='ALMA/', label='ALMA', project=False, highRes=True, BinMap=Hersch_C, BinMapE=Hersch_E) 

    spitzer_CH1 = '/Users/akankshabij/Documents/MSc/Research/Data/Spitzer/RCW36-4-selected_Post_BCDs/r15990016/ch1/pbcd/SPITZER_I1_15990016_0000_3_E8591943_maic.fits'
    spitzer_CH2 = '/Users/akankshabij/Documents/MSc/Research/Data/Spitzer/RCW36-4-selected_Post_BCDs/r15990016/ch2/pbcd/SPITZER_I2_15990016_0000_3_E8591952_maic.fits'
    spitzer_CH3 = '/Users/akankshabij/Documents/MSc/Research/Data/Spitzer/RCW36-4-selected_Post_BCDs/r15990016/ch3/pbcd/SPITZER_I3_15990016_0000_3_E8592062_maic.fits'
    spitzer_CH4 = '/Users/akankshabij/Documents/MSc/Research/Data/Spitzer/RCW36-4-selected_Post_BCDs/r15990016/ch4/pbcd/SPITZER_I4_15990016_0000_3_E8592077_maic.fits'

    runBoth(spitzer_CH1, Mask=None, outName='Spitzer_CH1/', kstep=1, label='Spitzer CH1', BinMap=Hersch_C, BinMapE=Hersch_E) 
    runBoth(spitzer_CH2, Mask=None, outName='Spitzer_CH2/', kstep=1, label='Spitzer CH2', BinMap=Hersch_C, BinMapE=Hersch_E)
    runBoth(spitzer_CH3, Mask=None, outName='Spitzer_CH3/', kstep=1, label='Spitzer CH3', BinMap=Hersch_C, BinMapE=Hersch_E)
    runBoth(spitzer_CH4, Mask=None, outName='Spitzer_CH4/', kstep=1, label='Spitzer CH4', BinMap=Hersch_C, BinMapE=Hersch_E)

def CompareFields():
    fldr = '/BandC/'
    HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
    vecMask = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandC_polFlux_3.fits')[0]
    Qmap = HAWC['STOKES Q'].data
    Umap = HAWC['STOKES U'].data
    print(Qmap.shape)

    HAWE = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits')
    Qmap2 = projectMap(HAWE['STOKES Q'], HAWC[0])
    Umap2 = projectMap(HAWE['STOKES U'], HAWC[0])
    vecMask2 = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandE_polFlux_3.fits')[0]
    mask = vecMask.copy()
    mask.data = projectMap(vecMask2, HAWC[0])
    print(Qmap2.shape)

    hro = h.HRO(HAWC[0].data, Qmap, Umap, hdu=mask, vecMask=vecMask, msk=True, compare=True, Qmap2=Qmap2, Umap2=Umap2)   
    prefix = 'Output_Plots'+ fldr +'FieldComparison/Smoothed/' 
    label = 'Field Comparison'  
    makePlots(hro, prefix, isSim=False, label=label, BinMap=None)
    print('Field Comaprison, Zx=', hro.Zx)

def runChannelMaps(fl, fldr):
    fitsfl = fits.open(fl)
    cube = SpectralCube.read(fitsfl)
    speed = np.round(np.linspace(3,10,71), 1)
    folder = '/ChannelMaps/' + fldr  
    HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')

    for i in speed:
        slab = cube.spectral_slab(i*u.km/u.s, (i+0.5)*u.km/u.s)
        mom0 = slab.moment0()
        mom0_HAWC = projectMap(mom0.hdu, HAWC[0])
        outName = folder + str(i) + '/'
        if os.path.isdir(outName)==False:
            os.mkdir(outName)
        runAnalysis(fits_fl=None, outName=outName, Mask=None, label=outName, project=False, band=0, file=False, Map=mom0_HAWC)

def binHerschel():
    HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
    Qmap = HAWC['STOKES Q']
    Umap = HAWC['STOKES U']
    vecMask = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandC_polFlux_3.fits')[0]
    Mask = vecMask.copy()
    Mask.data = np.ones(vecMask.data.shape)
    CII = fits.open('Data/Observations/CII_Integrated.fits')[0]
    Hersch = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits')[0]

    Hersch_C = projectMap(Hersch, HAWC[0])
    CII_C = projectMap(CII, HAWC[0])

    hro = h.HRO(CII_C, Qmap.data, Umap.data, hdu=Mask, vecMask=vecMask, msk=True, kstep=3, BinMap=Hersch_C)         
    makePlots(hro, prefix='Output_Plots/'+ 'BandC/' + 'CII_binHersch/', isSim=False, label='CII binned to Herschel', BinMap=None)

def binHerschel2():
    HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
    Qmap = HAWC['STOKES Q']
    Umap = HAWC['STOKES U']
    vecMask = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandC_polFlux_3.fits')[0]
    CII = fits.open('Data/Observations/CII_Integrated.fits')[0]

    Hersch = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits')[0]
    Qmap_Her = Hersch.copy()
    Umap_Her = Hersch.copy()
    vecMask_Her = Hersch.copy()
    CII_Her = Hersch.copy()

    Qmap_Her.data = projectMap(Qmap, Hersch)
    Umap_Her.data = projectMap(Umap, Hersch)
    vecMask_Her.data = projectMap(vecMask, Hersch)
    CII_Her.data = projectMap(CII, Hersch)

    Qmap_H = HAWC[0].copy()
    Umap_H = HAWC[0].copy()
    vecMask_H = HAWC[0].copy()
    CII_H = HAWC[0].copy()

    Qmap_H.data = projectMap(Qmap_Her, HAWC[0])
    Umap_H.data = projectMap(Umap_Her, HAWC[0])
    vecMask_H.data = projectMap(vecMask_Her, HAWC[0])
    CII_H.data = projectMap(CII_Her, HAWC[0])
    Mask = vecMask_H.copy()
    Mask.data = np.ones(vecMask.data.shape)

    hro = h.HRO(CII_H.data, Qmap_H.data, Umap_H.data, hdu=Mask, vecMask=vecMask_H, msk=True, kstep=3)         
    makePlots(hro, prefix='Output_Plots/'+ 'BandC/' + 'CII_binHersch/', isSim=False, label='CII binned to Herschel')
    
def runBoth(fits_fl, outName, Mask=None, label='', index=0, project=True, kernel='Gaussian', kstep=1, isSim=False, highRes=False, file=True, Map=None, BinMap=None, BinMapE=None):
    bandC = runAnalysis(fits_fl, outName, Mask, label, index, project, kernel, kstep, isSim, band=0, highRes=highRes, file=file, Map=Map, BinMap=BinMap)
    bandE = runAnalysis(fits_fl, outName, Mask, label, index, project, kernel, kstep, isSim, band=1, highRes=highRes, file=file, Map=Map, BinMap=BinMapE)
    #bandE = None
    return bandC, bandE

def CubeFlow(fits_fl, folder, out, start=-3, end=17, startNoise=-3, endNoise=-1, index=0, project=False, highRes=True, band=0):
    plotCube('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_12CO32.fits', folder='Output_Plots/BandE/12CO/Cube/', project=True, start=-3, end=17, pdfName='mom0_cube.pdf', gifName='mom0_cube.gif', band=1)
    # calcNoise(fits_fl='/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_12CO32.fits', fldr='Output_Plots/BandE/12CO/Cube/', start=-3, end=-1, project=True, band=1)
    # runCube(fits_fl='/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_12CO32.fits', folder='Output_Plots/BandE/12CO/Cube/', out='12CO/Cube/', start=-2, end=17, project=True, highRes=False, band=1)
    # gifCube('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandE/12CO/Cube/')
    # plotPRS_speed('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandE/12CO/Cube/')
    # plotHIST_speed('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandE/12CO/Cube/')

def runCube(cube_fl, bmap_fl, out_fldr, regions=None,  mask_fl=None, start=3, end=10, incr=0.1, index=0, project=False, highRes=False, band=2, pdfName='/mom0.pdf', gifName='/mom0.gif', projMap=None, kstep=2, colDen=None):
    data = fits.open(cube_fl)[index]
    cube = SpectralCube.read(data)
    splits = int((end - start)/incr) + 1
    #splits2 = int((end - start)/0.5) + 1
    speed = np.round(np.linspace(start,end,splits), 1)
    bmap = fits.open(bmap_fl)[11]

    PRS = np.zeros((3,len(speed)))
    PRS[0] = speed
    hists = np.zeros((len(speed),20))
    Phi = []
    j = 0
    _pdf = pdf.PdfPages(out_fldr + pdfName)
    moms = []
    for i in speed:
        cube_speed = cube.with_spectral_unit(unit=u.km/u.s, velocity_convention='optical')
        slab = cube_speed.spectral_slab(i*u.km/u.s, (i+incr)*u.km/u.s)
        mom0 = slab.moment0()
        if project:
            Bmap = bmap
            mom0_HAWC = bmap.copy()
            mom0_HAWC.data = projectMap(mom0.hdu, Bmap)
        else:
            mom0_HAWC = mom0.hdu
            Bmap = mom0_HAWC.copy()
            #Bmap.data = projectMap(bmap, mom0.hdu)
            Bmap.data = ((projectMap(bmap, mom0.hdu) * u.deg).to(u.rad)).value
        if (projMap is None)==False:
            mom0_HAWC = projMap.copy()
            mom0_HAWC.data = projectMap(mom0.hdu, projMap)
            Bmap = projMap.copy()
            Bmap.data = projectMap(bmap, projMap)
        print("*** mom0 SHAPE IS *** ", mom0_HAWC.shape)
        fldr = out_fldr + '/{0}_kms/'.format(i)
        label =  '{0}-{1} km/s'.format(i, i+incr)
        if os.path.exists(fldr)==False:
            os.mkdir(fldr)
        #Mask = HAWC[0].copy()
        #Mask.data = np.ones(HAWC[0].data.shape)

        if mask_fl is None:
            Mask = Bmap.copy()
            Mask.data = np.ones(mom0_HAWC.shape)

            if (regions is None)==False:  
                Std = []; Mean = []
                for region in regions:
                    s = np.nanstd(mom0_HAWC.data[region[0]-10:region[0]+10, region[1]-10:region[1]+10])
                    print(s)
                    m = np.nanmean(mom0_HAWC.data[region[0]-10:region[0]+10, region[1]-10:region[1]+10])
                    Std.append(s)
                    Mean.append(m)
                std = np.nanmean(np.asarray(Std))
                mean = np.nanmean(np.asarray(Mean))
                
                Mask.data[((mom0_HAWC.data - mean) / std) < 4]=0  
                Mask.data[np.isnan(mom0_HAWC.data)]=np.nan
        
        else:
            Mask = fits.open(mask_fl)[0]

        plt.close('all')
        fig = plt.figure(figsize=[12,8])
        fxx = aplpy.FITSFigure(Mask, figure=fig)
        fxx.show_colorscale(vmax=1, vmin=0)
        fxx.add_colorbar()
        fxx.add_label(0.85, 0.85, text='{0} km/s'.format(i), relative=True, color='white', size='xx-large', weight='demibold')
        plt.savefig(fldr + '/DataMask.png')

        fig = plt.figure(figsize=[12,8])
        fxx = aplpy.FITSFigure(mom0_HAWC, figure=fig)
        fxx.show_colorscale(interpolation='nearest')
        #fxx.show_colorscale(vmax=18, vmin=-7)
        fxx.add_colorbar()
        fxx.add_label(0.85, 0.85, text='{0} km/s'.format(i), relative=True, color='black', size='xx-large', weight='demibold')
        fxx.show_vectors(Mask, Bmap, step=20, scale =20, units='radians', color = 'White', linewidth=3)
        fxx.show_vectors(Mask, Bmap, step=20, scale =20, units='radians', color = 'Black', linewidth=1)
        #fxx.show_contour(Hersch, levels=[15,20,50,80], colors='black')
        if (colDen is None)==False:
            fxx.show_contour(fits.open(colDen)[0], levels=[0.7e23,1e23,1.5e23,2.5e23], colors='black')
        figName =  fldr + '/mom0.png'
        plt.savefig(figName)
        moms.append(imageio.imread(figName))
        _pdf.savefig(fig)

        hro = runAnalysis(Bmap=Bmap.data, outName=fldr+'/', Mask=Mask, vecMask=Mask, label=label, index=index, project=False, kernel='Gaussian', kstep=kstep, isSim=False, band=band, highRes=highRes, file=False, Map=mom0_HAWC)
        PRS[1,j] = hro.Zx
        PRS[2,j] = hro.meanPhi
        hists[j] = hro.hist
        Phi.append(hro.phi)
        j = j+1
    print(PRS.shape)
    print(hists.shape)
    np.save(out_fldr + '/PRS_speed.npy', PRS)
    np.save(out_fldr + '/hist_speed.npy', hists)
    np.save(out_fldr + '/total_phi.npy', Phi)
    _pdf.close()
    imageio.mimsave(out_fldr + gifName, moms, fps=4)
    gifCube(out_fldr, speed)
    plotPRS_speed(out_fldr)
    plotHIST_speed(out_fldr)

def runMaskCube(fits_fl, folder, out, regions, start=3, end=10, index=0, project=False, highRes=True, band=0):
    data = fits.open(fits_fl)[index]
    incr = (int(end) - int(start))*10 + 1
    speed = np.round(np.linspace(start,end,incr), 1)
    cube = SpectralCube.read(data)
    vecMask = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandC_polFlux_3.fits')[0]
    Hersch = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits')
    if band==0:
        HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
    elif band==1:
        HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits')
    else:
        HAWC = fits.open('Data/Observations/12CO_Integrated.fits')
    PRS = np.zeros((3,len(speed)))
    PRS[0] = speed
    hists = np.zeros((len(speed),20))
    Phi = []
    j = 0
    for i in speed:
        cube_speed = cube.with_spectral_unit(unit=u.km/u.s, velocity_convention='optical')
        slab = cube_speed.spectral_slab(i*u.km/u.s, (i+0.1)*u.km/u.s)
        mom0 = slab.moment0()
        if project:
            mom0_HAWC = HAWC[0].copy()
            mom0_HAWC.data = projectMap(mom0.hdu, HAWC[0])
        else:
            mom0_HAWC = mom0.hdu
        fldr = '{0}_kmpers/'.format(i)
        outName = out + fldr
        label =  outName + '{0}-{1} km/s'.format(i, i+0.1)
        if os.path.exists(folder + fldr)==False:
            os.mkdir(folder + fldr)
        Mask = mom0_HAWC.copy()
        Mask.data = np.ones(mom0_HAWC.shape)
        
        Std = []; Mean = []
        for region in regions:
            s = np.nanstd(mom0_HAWC.data[region[0]-10:region[0]+10, region[1]-10:region[1]+10])
            print(s)
            m = np.nanmean(mom0_HAWC.data[region[0]-10:region[0]+10, region[1]-10:region[1]+10])
            Std.append(s)
            Mean.append(m)
        std = np.nanmean(np.asarray(Std))
        mean = np.nanmean(np.asarray(Mean))
        
        Mask.data[((mom0_HAWC.data - mean) / std) < 4]=0  
        Mask.data[np.isnan(mom0_HAWC.data)]=np.nan

        # plt.close('all')
        # fig = plt.figure(figsize=[12,8])
        # fxx = aplpy.FITSFigure(mom0_HAWC, figure=fig)
        # fxx.show_colorscale(interpolation='nearest')
        # fxx.add_colorbar()
        # fxx.add_label(0.85, 0.85, text='{0} km/s'.format(i), relative=True, color='black', size='xx-large', weight='demibold')
        # fxx.show_vectors(vecMask, HAWC[11], step=6, scale =4, units='degrees', color = 'White', linewidth=3)
        # fxx.show_vectors(vecMask, HAWC[11], step=6, scale =4, units='degrees', color = 'Black', linewidth=1)
        # fxx.show_contour(Hersch, levels=[15,20,50,80], colors='black')
        # for region in regions:
        #     fxx.show_rectangles(region[0], region[1], height=20, width=20, coords_frame='pixel', edgecolor='cyan', linewidth=1.8)
        # plt.savefig(folder + fldr + 'mom0_regions.png') 

        plt.close('all')
        fig = plt.figure(figsize=[12,8])
        fxx = aplpy.FITSFigure(Mask, figure=fig)
        fxx.show_colorscale(vmax=1, vmin=0)
        fxx.add_colorbar()
        fxx.add_label(0.85, 0.85, text='{0} km/s'.format(i), relative=True, color='white', size='xx-large', weight='demibold')
        plt.savefig(folder + fldr + 'DataMask.png')

        plt.close('all')
        fig = plt.figure(figsize=[12,8])
        fxx = aplpy.FITSFigure(mom0_HAWC, figure=fig)
        fxx.show_colorscale(interpolation='nearest')
        fxx.add_colorbar()
        fxx.add_label(0.85, 0.85, text='{0} km/s'.format(i), relative=True, color='black', size='xx-large', weight='demibold')
        fxx.show_vectors(vecMask, HAWC[11], step=6, scale =4, units='degrees', color = 'White', linewidth=3)
        fxx.show_vectors(vecMask, HAWC[11], step=6, scale =4, units='degrees', color = 'Black', linewidth=1)
        fxx.show_contour(Hersch, levels=[15,20,50,80], colors='black')
        fxx.show_contour(Mask, levels=[0.9], colors='red')
        for region in regions:
            fxx.show_rectangles(region[0], region[1], height=20, width=20, coords_frame='pixel', edgecolor='cyan', linewidth=1.8)
        plt.savefig(folder + fldr + 'mom0_regions.png') 

        hro = runAnalysis('', outName=outName, Mask=Mask, label=label, index=index, project=False, kernel='Gaussian', kstep=3, isSim=False, band=band, highRes=highRes, file=False, Map=mom0_HAWC)
        PRS[1,j] = hro.Zx
        PRS[2,j] = hro.meanPhi
        hists[j] = hro.hist
        Phi.append(hro.phi)
        j = j+1
    print(PRS.shape)
    print(hists.shape)
    np.save(folder + 'PRS_speed.npy', PRS)
    np.save(folder + 'hist_speed.npy', hists)
    np.save(folder + 'total_phi.npy', Phi)

def gifCube(fldr, speed):
    #speed = np.round(np.linspace(-3,3,71), 1)
    #cube = SpectralCube.read(fitsfl)
    Hist = [] ; Phi = []; Mask = [] ; Mom = []
    #_pdf = pdf.PdfPages('Output_Plots/BandC/12CO/Cube/' + pdfName)
    for i in speed:
        folder = fldr + '/{0}_kms/'.format(i)

        hist = folder + '/_phi_secthistogram.png'
        phi = folder + '/phi.png'
        mask = folder + '/DataMask.png'
        mom = folder + '/mom0.png'

        Hist.append(imageio.imread(hist))
        Phi.append(imageio.imread(phi))
        Mask.append(imageio.imread(mask))
        Mom.append(imageio.imread(mom))
        #_pdf.savefig(hist)
    #_pdf.close()
    imageio.mimsave(fldr + '/DataMask.gif', Mask, fps=6)
    imageio.mimsave(fldr + '/mom0.gif', Mom, fps=6)
    imageio.mimsave(fldr + '/phi.gif', Phi, fps=6)
    imageio.mimsave(fldr + '/Hist.gif', Hist, fps=6)

def plotCube(fits_fl, bmap_fl, out_fldr, start=-1, end=13, incr=0.1, index=0, pdfName='', gifName='', project=True, band=0):
    data = fits.open(fits_fl)[index]
    splits = int((end - start)/incr) + 1
    speed = np.round(np.linspace(start,end,splits), 1)
    cube = SpectralCube.read(data)
    Bmap = fits.open(bmap_fl)[0]
    _pdf = pdf.PdfPages(folder + pdfName)
    moms = []; maps = []

    for i in speed:
        cube_speed = cube.with_spectral_unit(unit=u.km/u.s, velocity_convention='radio')
        slab = cube_speed.spectral_slab(i*u.km/u.s, (i+incr)*u.km/u.s)
        mom0 = slab.moment0()
        # if project and band!=2:
        #     mom0_HAWC = HAWC[0].copy()
        #     mom0_HAWC.data = projectMap(mom0.hdu, HAWC[0])
        # elif project and band==2:
        #     mom0_HAWC = CO12[0].copy()
        #     mom0_HAWC.data = projectMap(mom0.hdu, CO12[0])
        # elif project==False: 
        #     mom0_HAWC = mom0.hdu
        if project:
            mom0_HAWC = Bmap.copy()
            mom0_HAWC.data = projectMap(mom0.hdu, Bmap)
        else:
            mom0_HAWC = mom0.hdu
        maps.append(mom0_HAWC.data)
        fldr = '/{0}_kms/'.format(i)
        outName = out_fldr + fldr
        if os.path.exists(outName)==False:
            os.mkdir(outName)
        plt.close('all')
        fig = plt.figure(figsize=[12,8])
        fxx = aplpy.FITSFigure(mom0_HAWC, figure=fig)
        #fxx.show_colorscale(interpolation='nearest')
        fxx.show_colorscale(vmax=18, vmin=-7)
        fxx.add_colorbar()
        fxx.add_label(0.85, 0.85, text='{0} km/s'.format(i), relative=True, color='black', size='xx-large', weight='demibold')
        if band!=2:
            fxx.show_vectors(vecMask, HAWC[11], step=6, scale =4, units='degrees', color = 'White', linewidth=3)
            fxx.show_vectors(vecMask, HAWC[11], step=6, scale =4, units='degrees', color = 'Black', linewidth=1)
        else:
            fxx.show_vectors(vecMask, BLASTpol[0], step=6, scale =4, units='radians', color = 'White', linewidth=3)
            fxx.show_vectors(vecMask, BLASTpol[0], step=6, scale =4, units='radians', color = 'Black', linewidth=1)
        fxx.show_contour(Hersch, levels=[15,20,50,80], colors='black')
        #fxx.show_contour(fits.open(ALMA)[0], levels=[0.01, 0.02], colors='black')
        figName = outName + 'mom0_ALMABand6.png'
        plt.savefig(figName)
        moms.append(imageio.imread(figName))
        _pdf.savefig(fig)
    _pdf.close()
    imageio.mimsave(folder + gifName, moms, fps=6)

def ALMACont(fits_fl, folder, start=-1, end=13, index=0, pdfName='', gifName='', project=True, band=0):
    data = fits.open(fits_fl)[index]
    incr = (int(end) - int(start))*10 + 1
    speed = np.round(np.linspace(start,end,incr), 1)
    cube = SpectralCube.read(data)
    #ALMA_cube = SpectralCube.read(fits.open(ALMA_13CS)[0])
    _pdf = pdf.PdfPages(folder + pdfName)
    moms = []; maps = []
    CO12 = fits.open('Data/Observations/12CO_Integrated.fits')
    if band==0:
        HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
        vecMask = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandC_polFlux_3.fits')[0]
    elif band==1:
        HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits')
        vecMask = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandE_polFlux_3.fits')[0]
    else:
        BLASTpol = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/BlastPOL/BLASTPol_500_intermediate_BPOSang.fits')
        vecMask = BLASTpol[0].copy()
        vecMask.data = np.ones(BLASTpol[0].data.shape)
    Hersch = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits')
    for i in speed:
        cube_speed = cube.with_spectral_unit(unit=u.km/u.s, velocity_convention='optical')
        #ALMA_speed = ALMA_cube.with_spectral_unit(unit=u.km/u.s, velocity_convention='optical')
        
        slab = cube_speed.spectral_slab(i*u.km/u.s, (i+0.1)*u.km/u.s)
        #ALMA_slab = ALMA_speed.spectral_slab(i*u.km/u.s, (i+0.1)*u.km/u.s)
        
        mom0 = slab.moment0()
        #ALMA_mom0 = ALMA_slab.moment0()

        if project and band!=2:
            mom0_HAWC = HAWC[0].copy()
            mom0_HAWC.data = projectMap(mom0.hdu, HAWC[0])
        elif project and band==2:
            mom0_HAWC = CO12[0].copy()
            mom0_HAWC.data = projectMap(mom0.hdu, CO12[0])
        elif project==False: 
            mom0_HAWC = mom0.hdu
        maps.append(mom0_HAWC.data)
        fldr = '{0}_kmpers/'.format(i)
        outName = folder + fldr
        if os.path.exists(outName)==False:
            os.mkdir(outName)
        plt.close('all')
        fig = plt.figure(figsize=[12,8])
        fxx = aplpy.FITSFigure(mom0_HAWC, figure=fig)
        fxx.show_colorscale(interpolation='nearest')
        fxx.add_colorbar()
        fxx.add_label(0.85, 0.85, text='{0} km/s'.format(i), relative=True, color='black', size='xx-large', weight='demibold')
        if band!=2:
            fxx.show_vectors(vecMask, HAWC[11], step=6, scale =4, units='degrees', color = 'White', linewidth=3)
            fxx.show_vectors(vecMask, HAWC[11], step=6, scale =4, units='degrees', color = 'Black', linewidth=1)
        else:
            fxx.show_vectors(vecMask, BLASTpol[0], step=6, scale =4, units='radians', color = 'White', linewidth=3)
            fxx.show_vectors(vecMask, BLASTpol[0], step=6, scale =4, units='radians', color = 'Black', linewidth=1)
        #fxx.show_contour(Hersch, levels=[15,20,50,80], colors='black')
        #fxx.show_contour(ALMA_mom0.hdu, levels=[0.1, 0.15, 0.2], colors='black')
        fxx.show_contour(fits.open(ALMA)[0], levels=[0.01, 0.015, 0.02], colors='black')
        figName = outName + 'mom0_ALMABand6.png'
        plt.savefig(figName)
        moms.append(imageio.imread(figName))
        _pdf.savefig(fig)
    _pdf.close()
    imageio.mimsave(folder + gifName, moms, fps=6)

def mom1_1km(cube_fl, bmap_fl, folder, start=-1, end=13, index=0, incr=0.1, pdfName='', gifName='', project=True, band=0, projMap=None, colDen=None):
    data = fits.open(cube_fl)[index]
    cube = SpectralCube.read(data)
    splits = int((end - start)/incr) + 1
    speed = np.round(np.linspace(start,end,splits), 1)
    bmap = fits.open(bmap_fl)[0]
    _pdf = pdf.PdfPages(folder + '/mom1.pdf')
    moms = []; maps = []

    for i in speed:
        cube_speed = cube.with_spectral_unit(unit=u.km/u.s, velocity_convention='optical')
        slab = cube_speed.spectral_slab(i*u.km/u.s, (i+1)*u.km/u.s)
        mom0 = slab.moment1()
        if project:
            Bmap = bmap
            mom0_HAWC = bmap.copy()
            mom0_HAWC.data = projectMap(mom0.hdu, Bmap)
        else:
            mom0_HAWC = mom0.hdu
            Bmap = mom0_HAWC.copy()
            Bmap.data = projectMap(bmap, mom0.hdu)
        if (projMap is None)==False:
            mom0_HAWC = projMap.copy()
            mom0_HAWC.data = projectMap(mom0.hdu, projMap)
            Bmap = projMap.copy()
            Bmap.data = projectMap(bmap, projMap)
        maps.append(mom0_HAWC.data)
        outName = folder + '/{0}_kms/'.format(i)
        label =  '{0}-{1} km/s'.format(i, i+incr)
        if os.path.exists(outName)==False:
            os.mkdir(outName)
        plt.close('all')
        fig = plt.figure(figsize=[12,8])
        fxx = aplpy.FITSFigure(mom0_HAWC, figure=fig)
        fxx.recenter(134.87,-43.74, height=0.22, width=0.22)
        #fxx.show_colorscale(interpolation='nearest')
        fxx.show_colorscale(vmax=i+1, vmin=i, cmap='bwr')
        fxx.add_colorbar()
        fxx.add_label(0.85, 0.85, text='{0} km/s'.format(i), relative=True, color='black', size='xx-large', weight='demibold')
        if (colDen is None)==False:
            #fxx.show_contour(fits.open(colDen)[0], levels=[0.7e23,1e23,1.5e23,2.5e23], colors='black')
            fxx.show_contour(fits.open(colDen)[0], levels=[20, 50, 100], colors='black')
        # fxx.show_vectors(vecMask, Bmap, step=10, scale =8, units='degrees', color = 'White', linewidth=3)
        # fxx.show_vectors(vecMask, Bmap, step=10, scale =8, units='degrees', color = 'Black', linewidth=1)
        # fxx.show_contour(Hersch, levels=[15,20,50,80], colors='black')
        fxx.show_contour(fits.open(ALMA)[0], levels=[0.01, 0.02], colors='black')
        figName = outName + '/mom1.png'
        plt.savefig(figName)
        moms.append(imageio.imread(figName))
        _pdf.savefig(fig)
    _pdf.close()
    imageio.mimsave(folder + '/mom1.gif', moms, fps=5)

def mom0_1km(fits_fl, folder, start=-1, end=13, index=0, pdfName='', gifName='', project=True, band=0):
    data = fits.open(fits_fl)[index]
    incr = (int(end) - int(start))*5 + 1
    speed = np.round(np.linspace(start,end,incr), 1)
    cube = SpectralCube.read(data)
    _pdf = pdf.PdfPages(folder + pdfName)
    moms = []; maps = []
    CO12 = fits.open('Data/Observations/12CO_Integrated.fits')
    if band==0:
        HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
        vecMask = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandC_polFlux_3.fits')[0]
    elif band==1:
        HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits')
        vecMask = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandE_polFlux_3.fits')[0]
    else:
        BLASTpol = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/BlastPOL/BLASTPol_500_intermediate_BPOSang.fits')
        vecMask = BLASTpol[0].copy()
        vecMask.data = np.ones(BLASTpol[0].data.shape)
    Hersch = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits')
    for i in speed:
        cube_speed = cube.with_spectral_unit(unit=u.km/u.s, velocity_convention='optical')
        slab = cube_speed.spectral_slab(i*u.km/u.s, (i+1)*u.km/u.s)
        mom0 = slab.moment0()
        if project and band!=2:
            mom0_HAWC = HAWC[0].copy()
            mom0_HAWC.data = projectMap(mom0.hdu, HAWC[0])
        elif project and band==2:
            mom0_HAWC = CO12[0].copy()
            mom0_HAWC.data = projectMap(mom0.hdu, CO12[0])
        elif project==False: 
            mom0_HAWC = mom0.hdu
        maps.append(mom0_HAWC.data)
        fldr = '{0}_kmpers/'.format(i)
        outName = folder + fldr
        if os.path.exists(outName)==False:
            os.mkdir(outName)
        plt.close('all')
        fig = plt.figure(figsize=[12,8])
        fxx = aplpy.FITSFigure(mom0_HAWC, figure=fig)
        fxx.show_colorscale(interpolation='nearest')
        #fxx.show_colorscale(vmax=i+1, vmin=i)
        fxx.add_colorbar()
        fxx.add_label(0.85, 0.85, text='{0} km/s'.format(i), relative=True, color='black', size='xx-large', weight='demibold')
        if band!=2:
            fxx.show_vectors(vecMask, HAWC[11], step=6, scale =4, units='degrees', color = 'White', linewidth=3)
            fxx.show_vectors(vecMask, HAWC[11], step=6, scale =4, units='degrees', color = 'Black', linewidth=1)
        else:
            fxx.show_vectors(vecMask, BLASTpol[0], step=6, scale =4, units='radians', color = 'White', linewidth=3)
            fxx.show_vectors(vecMask, BLASTpol[0], step=6, scale =4, units='radians', color = 'Black', linewidth=1)
        fxx.show_contour(Hersch, levels=[15,20,50,80], colors='black')
        #fxx.show_contour(fits.open(ALMA)[0], levels=[0.01, 0.02], colors='black')
        figName = outName + 'mom0.png'
        plt.savefig(figName)
        moms.append(imageio.imread(figName))
        _pdf.savefig(fig)
    _pdf.close()
    imageio.mimsave(folder + gifName, moms, fps=5)

def RGB_Cube(cube_fl, out_fldr,  velocity=[-3,6,9,17], projMap=None, colDen=None):
    data = fits.open(cube_fl)[0]
    cube_s = SpectralCube.read(data)
    cube = cube_s.with_spectral_unit(unit=u.km/u.s, velocity_convention='optical')
    slab_B = cube.spectral_slab(velocity[0]*u.km/u.s, velocity[1]*u.km/u.s)
    slab_G = cube.spectral_slab(velocity[1]*u.km/u.s, velocity[2]*u.km/u.s)
    slab_R = cube.spectral_slab(velocity[2]*u.km/u.s, velocity[3]*u.km/u.s)

    if (projMap is None)==False:
        mom0_B = projMap.copy()
        mom0_B.data = projectMap(slab_B.moment0().hdu, projMap)
        mom0_G = projMap.copy()
        mom0_G.data = projectMap(slab_G.moment0().hdu, projMap)
        mom0_R = projMap.copy()
        mom0_R.data = projectMap(slab_R.moment0().hdu, projMap)

    else:
        mom0_B = slab_B.moment0().hdu
        mom0_G = slab_G.moment0().hdu
        mom0_R = slab_R.moment0().hdu

    mom0_B.writeto(out_fldr + '/Cube_B.fits', overwrite=True)
    mom0_G.writeto(out_fldr + '/Cube_G.fits', overwrite=True)
    mom0_R.writeto(out_fldr + '/Cube_R.fits', overwrite=True)

    aplpy.make_rgb_cube([out_fldr + '/Cube_R.fits', out_fldr + '/Cube_G.fits', out_fldr + '/Cube_B.fits'], out_fldr + '/RGB_speeds.fits')
    aplpy.make_rgb_image(out_fldr + '/RGB_speeds.fits', out_fldr + '/RGB_speeds.png', 
                        stretch_r='linear',stretch_g='linear',stretch_b='linear')

    fig = plt.figure(figsize=[10,8], dpi=300)
    f = aplpy.FITSFigure(out_fldr + '/RGB_speeds.png', figure=fig)
    f.show_rgb()
    #if (colDen is None)==False:
    #        f.show_contour(fits.open(colDen)[0], levels=[0.7e23,1e23,1.5e23,2.5e23], colors='black')
    #f.recenter(134.87,-43.78, height=0.2, width=0.2)
    #f.recenter(134.87,-43.74, height=0.15, width=0.15)
    f.show_contour(Hersch, levels=[15,20,50,80], colors='black')
    f.recenter(134.87,-43.74, height=0.3, width=0.3)
    #f.show_vectors(BlastMask, fits.open(BlastPol)[0], step=7, scale =4, units='radians', color = 'gray', linewidth=0.8)
    f.show_vectors(vecMask, HAWC[11], step=5, scale=3, units='degrees', color='cyan', linewidth=1.5)
    #f.show_contour(colDen, levels=[20, 50, 100], colors='lightgray')
    #f.show_markers([OB9.ra.value, OB9_7.ra.value], [OB9.dec.value, OB9_7.dec.value], coords_frame='world', marker="*", s=2**8, c='yellow')
    plt.savefig(out_fldr + '/RGB_speeds.png')

def calcNoise(fits_fl, fldr, index=0, start=-1, end=1, project=False, band=0):
    data = fits.open(fits_fl)[index]
    print(data.shape)
    incr = (int(end) - int(start))*10 + 1
    speed = np.round(np.linspace(start,end,21), incr)
    cube = SpectralCube.read(data)
    if band==0:
        HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
    elif band==1:
        HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits')
    else:
        HAWC = fits.open('Data/Observations/12CO_Integrated.fits')
    if project==False:
        maps = np.zeros((len(speed), data.data.shape[2], data.data.shape[3]))
    else:
        maps = np.zeros((len(speed), HAWC[0].shape[0], HAWC[0].shape[1]))
    j=0
    for i in speed:
        print(i)
        cube_speed = cube.with_spectral_unit(unit=u.km/u.s, velocity_convention='optical')
        slab = cube_speed.spectral_slab(i*u.km/u.s, (i+0.1)*u.km/u.s)
        mom0 = slab.moment0()
        if project:
            mom0_HAWC = projectMap(mom0.hdu, HAWC[0])
        else:
            mom0_HAWC = mom0.hdu.data
            print(mom0.hdu.data.shape)
        maps[j] = mom0_HAWC
        j = j + 1
        #maps.append(mom0_HAWC.data)
        # fldr = '{0}_kmpers/'.format(i)
        # outName = 'Output_Plots/BandC/12CO/Cube/' + fldr
        # if os.path.exists(outName)==False:
        #     os.mkdir(outName)
    #Maps = np.asarray(maps)
    print(maps.shape)
    std = np.std(maps,0)
    mean = np.mean(maps,0)
    print(std.shape)
    std_Map = HAWC[0].copy()
    std_Map.data = std
    mean_Map = HAWC[0].copy()
    mean_Map.data = mean
    std_Map.writeto(fldr + 'NoiseMap_std.fits')
    mean_Map.writeto(fldr + 'NoiseMap_mean.fits')
    fig = plt.figure(figsize=[12,8])
    fxx = aplpy.FITSFigure(std_Map, figure=fig)
    fxx.show_colorscale(interpolation='nearest')
    fxx.add_colorbar()
    plt.savefig(fldr + 'NoiseMap_std.png')
    fig = plt.figure(figsize=[12,8])
    fxx = aplpy.FITSFigure(mean_Map, figure=fig)
    fxx.show_colorscale(interpolation='nearest')
    fxx.add_colorbar()
    plt.savefig(fldr + 'NoiseMap_mean.png')

def loopNoise(fits_fl, outName, counter, Mask=None, label='', index=0, project=True, kernel='Gaussian', kstep=1, isSim=False, band=0, highRes=False, BinMap=None):
    PRS = []
    folder = 'Output_Plots/WhiteNoise/BlastPol/' + outName
    for i in range(counter):
        Zx = runNoise(fits_fl, outName, Mask=Mask, label=label, index=index, project=project, kernel=kernel, kstep=kstep, isSim=isSim, band=band, highRes=highRes, BinMap=BinMap)
        PRS.append(Zx)
    np.save(folder + 'PRS_{0}.npy'.format(counter), PRS)

def runNoise(fits_fl, outName, Mask=None, label='', index=0, project=True, kernel='Gaussian', kstep=1, isSim=False, band=0, highRes=False, BinMap=None):
    prefix = 'Output_Plots/WhiteNoise/BlastPol/' + outName
    #HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
    #vecMask = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandC_polFlux_3.fits')[0]
    #Qmap = HAWC['STOKES Q'].data
    #Umap = HAWC['STOKES U'].data
    Bvec = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/BlastPOL/BLASTPol_500_intermediate_BPOSang.fits')[0]
    Mask = vecMask.copy()
    Mask.data = np.ones(vecMask.data.shape)

    noiseMap = WhiteNoiseMap(fits_fl, index, Bvec, prefix)
    Bmap = projectMap(Bvec, noiseMap) 
    hro = h.HRO(noiseMap.data, Bmap=Bmap, hdu=Mask, vecMask=vecMask, msk=True, kernel=kernel, kstep=kstep) 
    #makePlots(hro, prefix, isSim, label, BinMap)
    return hro.Zx

def WhiteNoiseMap(fits_fl, index, ref, prefix=''):
    Map = fits.open(fits_fl)[index]
    mean = np.nanmean(Map.data)
    std = np.nanstd(Map.data)
    noise = np.random.normal(mean, std, size=Map.data.shape)
    noise_full = Map.copy()
    noise_full.data = noise
    print(noise_full.data.shape)
    noise_Map = noise_full
    #noise_Map = ref.copy()
    #noise_Map = projectMap(noise_full, ref)

    fig = plt.figure(figsize=[12,8])
    fxx = aplpy.FITSFigure(noise_Map, figure=fig, slices=[0])
    fxx.show_colorscale(interpolation='nearest', cmap='binary')
    plt.savefig(prefix + 'NoiseMap.png')
    return noise_Map

def WhiteNoise_PRS(outName, counter):
    folder = 'Output_Plots/WhiteNoise/BlastPol' + outName
    PRS = np.load(folder + 'PRS_{0}.npy'.format(counter))
    print(outName + ' : PRS mean = ', np.mean(PRS))
    print(outName + ' : PRS std = ', np.std(PRS))

def runAnalysis(Bmap, outName, vecMask=None, Mask=None, label='', index=0, project=True, kernel='Gaussian', kstep=1, isSim=False, band=0, highRes=False, file=True, Map=None, BinMap=None):
    
    #Bvec = fits.open(fits_fl)
    #Bmap = projectMap(bmap, Map) 
    #Mask = Map.copy()
    #Mask.data = np.ones(Bmap.shape)
    Qmap = None
    Umap = None

    if isSim==False:
        hro = h.HRO(Map.data, Qmap=Qmap, Umap=Umap, Bmap=Bmap, hdu=Mask, vecMask=vecMask, msk=True, kernel=kernel, kstep=kstep, BinMap=BinMap)   
        prefix=outName    
    makePlots(hro, prefix, isSim, label, BinMap)
    return hro

def makePlots(hro, prefix, isSim, label, BinMap=None):
    if isSim: scale=15; step=20
    else: scale=5; step=6
    if BinMap is None:
        label=label
    else:
        label=label+' Binned to Column Density'

    p.plot_Map(hro, prefix, norm=False)
    p.plot_Fields(hro, prefix, Bfield=True, Efield=True, step=step, scale=scale)
    p.plot_Gradient(hro, prefix, norm=False)
    p.plot_GradientAmp(hro, prefix, norm=True)
    p.plot_vectors(hro, prefix, step=step, scale=scale)
    p.plot_regions(hro, prefix, step=step, scale=scale)
    p.plot_phi(hro, prefix, step=step, scale=scale, label=label)
    p.plot_histShaded(hro, label, prefix)
    #p.plot_FEWsect(hro, label, prefix)
    p.plot_secthist(hro, label, prefix)

def makePlots_Compare(hro, prefix, isSim, label, BinMap=None):
    if isSim: scale=15; step=20
    else: scale=5; step=6
    if BinMap is None:
        label=label
    else:
        label=label+' Binned to Column Density'

    #p.plot_Map(hro, prefix, norm=False)
    p.plot_Fields(hro, prefix, Bfield=True, Efield=True, step=step, scale=scale)
    #p.plot_Gradient(hro, prefix, norm=False)
    #p.plot_GradientAmp(hro, prefix, norm=True)
    #p.plot_vectors(hro, prefix, step=step, scale=scale)
    #p.plot_regions(hro, prefix, step=step, scale=scale)
    p.plot_phi(hro, prefix, step=step, scale=scale, label=label)
    p.plot_histShaded(hro, label, prefix)
    #p.plot_FEWsect(hro, label, prefix)
    p.plot_secthist(hro, label, prefix)

def plotPRS_speed(folder):
    PRS = np.load(folder + '/PRS_speed.npy')
    plt.figure(figsize=[12,8], dpi=200)
    speed = PRS[0]
    stat =  PRS[1]
    angle = PRS[2] * 180/np.pi
    # for i in range(stat.shape[0]):
    #     if speed[i] < 8.2 and speed[i] > 7:
    #         stat[i] = np.nan
    print(stat)
    print(speed)
    plt.figure()
    plt.plot(speed, stat)
    plt.ylabel('Projected Rayleigh Statistic')
    plt.xlabel('Speed')
    plt.title('Statistic vs Speed')
    plt.savefig(folder + '/PRS_speed.png')
    
    plt.figure()
    plt.plot(speed, angle)
    plt.ylabel('Mean Relative Angle')
    plt.xlabel('Speed')
    plt.title('Mean Phi vs Speed')
    plt.savefig(folder + '/Phi_speed.png')

    fig, ax = plt.subplots()
    ax.plot(speed, stat)
    ax.set_ylabel('Projected Rayleigh Statistic')
    ax.set_xlabel('Speed')
    ax2=ax.twinx()
    ax2.plot(speed, angle, color='black')
    #ax2.set_ylim(0,90)
    ax2.set_ylabel('Mean Relative Angle')
    plt.savefig(folder + '/PhiPRS_speed.png')
    #np.save(folder + 'PRS_speed.npy', [speed, stat])

def plotPRS_Bands(cube):
    folder = '/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/'
    PRS_C = np.load(folder + 'BandC/' + cube + 'Cube/PRS_speed.npy')
    PRS_E = np.load(folder + 'BandE/' + cube + 'Cube/PRS_speed.npy')
    PRS_Blast = np.load(folder + 'BLASTpol/' + cube + 'Cube/PRS_speed.npy')
    plt.figure(figsize=[12,8], dpi=200)
    speed_C = PRS_C[0]
    stat_C =  PRS_C[1]
    speed_E = PRS_E[0]
    stat_E =  PRS_E[1]
    speed_Bl = PRS_Blast[0]
    stat_Bl =  PRS_Blast[1]
    plt.plot(speed_C, stat_C, label='HAWC+ Band C')
    plt.plot(speed_E, stat_E, label='HAWC+ Band E', linestyle='--')
    plt.plot(speed_Bl, stat_Bl, label='BLASTPol', linestyle=':')
    plt.legend()
    plt.ylabel('Projected Rayleigh Statistic')
    plt.xlabel('Speed')
    plt.title('Statistic vs Speed')
    plt.savefig(folder + 'BandC/' + cube + 'Cube/PRS_speed_bands.png')

def plotHIST_speed(folder):
    hists = np.load(folder + '/hist_speed.npy')
    speed = np.load(folder + '/PRS_speed.npy')[0]
    print(hists.shape)
    #print(speed[0])
    # hist_array = np.zeros((20, 131))
    # for i in range(hist_array.shape[0]):
    #     for j in range(hist_array.shape[1]):
    #         hist_array[i,j] = hist[j][i]

    # for i in range(hist_array.shape[1]):
    #     if speed[0][i] < 8.2 and speed[0][i] > 7:
    #         hist_array[:,i] = np.nan
    fig = plt.figure(figsize=[12,8])
    cax = plt.imshow(hists, aspect='auto', extent=[0,90,speed[0],speed[-1]], vmax=1, vmin=0, origin='lower')
    fig.colorbar(cax)
    plt.xlabel('relative angle')
    plt.ylabel('speed (km/s)')
    plt.title('colormap amplitude is histogram density')
    plt.savefig(folder + '/hist_summary.png')

def totalHIST(folder):
    Phi = np.load(folder + 'total_phi.npy')
    hist, bin_edges = np.histogram(Phi, bins=20, range=(0, np.pi/2), density=True)
    Zx = RayleighStatistic(Phi)
    meanPhi = MeanPhi(Phi)
    plt.close('all')
    plt.figure(figsize=[5.5,4], dpi=250)
    angle = np.linspace(0, 90, 20)
    plt.plot(angle, hist, linewidth=2, label='PRS: Zx={0}\n Mean $\phi$={1}'.format(np.round(Zx, 2), np.round(meanPhi*180/np.pi, 2)))
    plt.xlabel('Relative angle, $\phi$')
    plt.ylabel('Histogram Density')
    plt.grid(True, linewidth=0.5)
    plt.title('All channels - Histogram of Relative Orientation', fontsize=10)
    plt.legend(fontsize=12)
    plt.savefig(folder + 'totalHist.png')

def RayleighStatistic(angles, weights=None):
    # angles needs to be on domain (-pi/2, pi/2) (input parameter theta is 2*phi 
    theta = np.arctan(np.tan(angles))*2

    # Set default weights to be 1
    if weights==None:
        weights = np.ones(theta.shape)
        weights[np.isnan(theta)] = np.nan

    # Calculate weighted Projected Rayleigh Statistic, as defined in report
    Zx = np.nansum(weights*np.cos(theta)) / np.sqrt(np.nansum(weights**2)/2)

    return Zx

def MeanPhi(phi, weights=None):
    # Set default weights to be 1
    if weights==None:
        weights = np.ones(phi.shape)
        weights[np.isnan(phi)] = np.nan

    x = np.nansum(weights*np.cos(phi)) / np.nansum(weights)
    y = np.nansum(weights*np.sin(phi)) / np.nansum(weights)
    meanPhi = np.arctan2(y, x)
    return meanPhi

def projectMap(mapOrigin, ref):
    '''This function projects a given map 'mapOrigin' onto the same WCS coordinates as a reference map and returns the map in the same shape'''
    proj, footprint = reproject_exact(mapOrigin, ref.header)
    proj[np.isnan(ref.data)] = np.nan
    proj[0].data = proj
    return proj







