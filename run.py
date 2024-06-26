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

HAWC = '/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits'
HAWE = '/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits'
#ALMA = '/Users/akankshabij/Documents/MSc/Research/Data/ALMA/VelaC-CR-A_cont.fits'
ALMA = 'Data/Observations/ALMA_Band6_continuum.fits'
ALMA_13CS = '/Users/akankshabij/Documents/MSc/Research/Data/ALMA/13CS_Cube.fits'

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

def runCube(fits_fl, folder, out, start=3, end=10, index=0, project=False, highRes=True, band=0):
    data = fits.open(fits_fl)[index]
    incr = (int(end) - int(start))*10 + 1
    speed = np.round(np.linspace(start,end,incr), 1)
    cube = SpectralCube.read(data)
    #noise = fits.open(folder + 'NoiseMap_std.fits')[0]
    #mean = fits.open(folder + 'NoiseMap_mean.fits')[0]
    #print("*** NOISE SHAPE IS *** ", noise.shape)
    if band==0:
        HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
    elif band==1:
        HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits')
    else:
        HAWC = fits.open('Data/Observations/12CO_Integrated.fits')
    print("*** HAWC SHAPE IS *** ", HAWC[0].shape)
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
        print("*** mom0 SHAPE IS *** ", mom0_HAWC.shape)
        fldr = '{0}_kmpers/'.format(i)
        outName = out + fldr
        label =  outName + '{0}-{1} km/s'.format(i, i+0.1)
        if os.path.exists(folder + fldr)==False:
            os.mkdir(folder + fldr)
        #Mask = HAWC[0].copy()
        #Mask.data = np.ones(HAWC[0].data.shape)
        Mask = HAWC[0].copy()
        Mask.data = np.ones(mom0_HAWC.shape)
        #print("*** MASK SHAPE IS *** ", Mask.data.shape)
        #print("*** mean std is ***", np.nanmean(noise.data))
        #Mask.data[(mom0_HAWC.data / 0.014) < 3]=0 #C18O = 0.026, SiO=0.015; 13CS=0.02; 13CN = 0.014
        #Mask.data[(mom0_HAWC.data / noise.data) < 3]=0
        #Mask.data[np.isnan(mom0_HAWC.data)]=np.nan
        fig = plt.figure(figsize=[12,8])
        fxx = aplpy.FITSFigure(Mask, figure=fig)
        fxx.show_colorscale(vmax=1, vmin=0)
        fxx.add_colorbar()
        fxx.add_label(0.85, 0.85, text='{0} km/s'.format(i), relative=True, color='white', size='xx-large', weight='demibold')

        plt.savefig(folder + fldr + 'DataMask.png')
        fig = plt.figure(figsize=[12,8])
        fxx = aplpy.FITSFigure(Mask, figure=fig)
        fxx.show_colorscale(vmax=1, vmin=0)
        fxx.add_colorbar()
        fxx.add_label(0.85, 0.85, text='{0} km/s'.format(i), relative=True, color='white', size='xx-large', weight='demibold')
        plt.savefig(folder + fldr + 'DataMask.png')
        # for i in range(mom0_HAWC.shape[0]):
        #     for j in range(mom0_HAWC.shape[1]):
        #         if (mom0_HAWC[i,j] / noise.data[i,j]) < 3:
        #             Mask.data[i,j]=0
        print(outName)
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

def gifCube(fldr):
    speed = np.round(np.linspace(-3,3,71), 1)
    #cube = SpectralCube.read(fitsfl)
    Hist = [] ; Phi = []; Mask = [] ; Mom = []
    #_pdf = pdf.PdfPages('Output_Plots/BandC/12CO/Cube/' + pdfName)
    for i in speed:
        folder = fldr + '{0}_kmpers/'.format(i)

        hist = folder + 'MASKED__phi_secthistogram.png'
        phi = folder + 'MASKED_phi.png'
        mask = folder + 'DataMask.png'
        mom = folder + 'mom0_regions.png'

        Hist.append(imageio.imread(hist))
        Phi.append(imageio.imread(phi))
        Mask.append(imageio.imread(mask))
        Mom.append(imageio.imread(mom))
        #_pdf.savefig(hist)
    #_pdf.close()
    imageio.mimsave(fldr + 'DataMask.gif', Mask, fps=6)
    imageio.mimsave(fldr + 'mom0_regions.gif', Mom, fps=6)
    imageio.mimsave(fldr + 'phi.gif', Phi, fps=6)
    imageio.mimsave(fldr + 'Hist.gif', Hist, fps=6)

def plotCube(fits_fl, folder, start=-1, end=13, index=0, pdfName='', gifName='', project=True, band=0):
    data = fits.open(fits_fl)[index]
    incr = (int(end) - int(start))*10 + 1
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
        cube_speed = cube.with_spectral_unit(unit=u.km/u.s, velocity_convention='radio')
        slab = cube_speed.spectral_slab(i*u.km/u.s, (i+0.5)*u.km/u.s)
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

def mom1_1km(fits_fl, folder, start=-1, end=13, index=0, pdfName='', gifName='', project=True, band=0):
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
        mom0 = slab.moment1()
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
        #fxx.show_colorscale(interpolation='nearest')
        fxx.show_colorscale(vmax=i+1, vmin=i, cmap='bwr')
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
        figName = outName + 'mom1.png'
        plt.savefig(figName)
        moms.append(imageio.imread(figName))
        _pdf.savefig(fig)
    _pdf.close()
    imageio.mimsave(folder + gifName, moms, fps=5)

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

def runNoise(fits_fl, outName, Mask=None, label='', index=0, project=True, kernel='Gaussian', kstep=1, isSim=False, band=0, highRes=False, BinMap=None):
    prefix = 'Output_Plots/WhiteNoise/' + outName
    HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
    vecMask = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandC_polFlux_3.fits')[0]
    Qmap = HAWC['STOKES Q'].data
    Umap = HAWC['STOKES U'].data
    Mask = vecMask.copy()
    Mask.data = np.ones(vecMask.data.shape)

    noiseMap = WhiteNoiseMap(fits_fl, index, HAWC[0], prefix)
    hro = h.HRO(noiseMap.data, Qmap, Umap, Mask, vecMask, True, kernel, kstep) 
    #makePlots(hro, prefix, isSim, label, BinMap)
    return hro.Zx

def loopNoise(fits_fl, outName, counter, Mask=None, label='', index=0, project=True, kernel='Gaussian', kstep=1, isSim=False, band=0, highRes=False, BinMap=None):
    PRS = []
    folder = 'Output_Plots/WhiteNoise/' + outName
    for i in range(counter):
        Zx = runNoise(fits_fl, outName, Mask=Mask, label=label, index=index, project=project, kernel=kernel, kstep=kstep, isSim=isSim, band=band, highRes=highRes, BinMap=BinMap)
        PRS.append(Zx)
    np.save(folder + 'PRS_{0}.npy'.format(counter), PRS)

def WhiteNoise_PRS(outName, counter):
    folder = 'Output_Plots/WhiteNoise/' + outName
    PRS = np.load(folder + 'PRS_{0}.npy'.format(counter))
    print(outName + ' : PRS mean = ', np.mean(PRS))
    print(outName + ' : PRS std = ', np.std(PRS))

def runAnalysis(fits_fl, outName, Mask=None, label='', index=0, project=True, kernel='Gaussian', kstep=1, isSim=False, band=0, highRes=False, file=True, Map=None, BinMap=None):
    if file:
        Map = fits.open(fits_fl)[index]
    else:
        Map = Map   

    if band==0:
        fldr = '/BandC/'
        HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
        vecMask = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandC_polFlux_3.fits')[0] 
        Qmap = HAWC['STOKES Q'].data
        Umap = HAWC['STOKES U'].data
        Bmap=None

    elif band==1:
        fldr = '/BandE/'
        HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits')
        vecMask = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandE_polFlux_3.fits')[0]
        Qmap = HAWC['STOKES Q'].data
        Umap = HAWC['STOKES U'].data
        Bmap=None

    else:
        fldr = '/BLASTpol/'
        Bvec = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/BlastPOL/BLASTPol_500_intermediate_BPOSang.fits')
        Bmap = projectMap(Bvec, Map) 
        vecMask = Map.copy()
        vecMask.data = np.ones(Bmap.shape)
        Qmap = None
        Umap = None

    # elif band==2:
    #     fldr = '/BLASTpol/'
    #     HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits')
    #     vecMask = 
    #     Qmap = HAWC['STOKES Q'].data
    #     Umap = HAWC['STOKES U'].data

    if project: 
        Map = projectMap(Map, HAWC[0])                        # project map onto HAWC+ WCS map

    elif highRes:
        Qmap = projectMap(HAWC['STOKES Q'], Map).data
        Umap = projectMap(HAWC['STOKES U'], Map).data 
        HAWC_mask = vecMask.copy()
        vecMask = Map.copy()
        vecMask.data = projectMap(HAWC_mask, Map) 

    if BinMap is None:
        prefix='Output_Plots'+ fldr + outName
    else:
        prefix='Output_Plots/Bin_Hersch'+ fldr + outName

    if Mask==None:
        Mask = vecMask.copy()
        Mask.data = np.ones(vecMask.data.shape)

    # Run HRO analysis without masking data
    # hro = h.HRO(Map.data, Qmap, Umap, Mask, vecMask, False, kernel, kstep)         
    # makePlots(hro, prefix, isSim, label)

    # Repeat for masked data 
    if isSim==False:
        hro = h.HRO(Map.data, Qmap=Qmap, Umap=Umap, Bmap=Bmap, hdu=Mask, vecMask=vecMask, msk=True, kernel=kernel, kstep=kstep, BinMap=BinMap)   
        prefix=prefix + 'MASKED_'    
    makePlots(hro, prefix, isSim, label, BinMap)
    return hro

def makePlots(hro, prefix, isSim, label, BinMap=None):
    if isSim: scale=15; step=20
    else: scale=8; step=15
    if BinMap is None:
        label=label
    else:
        label=label+' Binned to Column Density'

    p.plot_Fields(hro, prefix, Bfield=True, Efield=True, step=step, scale=scale)
    p.plot_Map(hro, prefix='', norm=False)
    p.plot_Gradient(hro, prefix, norm=False)
    p.plot_GradientAmp(hro, prefix, norm=True)
    p.plot_vectors(hro, prefix, step=step, scale=scale)
    p.plot_regions(hro, prefix, step=step, scale=scale)
    p.plot_phi(hro, prefix, step=step, scale=scale, label=label)
    p.plot_histShaded(hro, label, prefix)
    #p.plot_FEWsect(hro, label, prefix)
    p.plot_secthist(hro, label, prefix)

def plotPRS_speed(folder):
    PRS = np.load(folder + 'PRS_speed.npy')
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
    plt.savefig(folder + 'PRS_speed.png')
    
    plt.figure()
    plt.plot(speed, angle)
    plt.ylabel('Mean Relative Angle')
    plt.xlabel('Speed')
    plt.title('Mean Phi vs Speed')
    plt.savefig(folder + 'Phi_speed.png')

    fig, ax = plt.subplots()
    ax.plot(speed, stat)
    ax.set_ylabel('Projected Rayleigh Statistic')
    ax.set_xlabel('Speed')
    ax2=ax.twinx()
    ax2.plot(speed, angle, color='black')
    #ax2.set_ylim(0,90)
    ax2.set_ylabel('Mean Relative Angle')
    plt.savefig(folder + 'PhiPRS_speed.png')
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
    hists = np.load(folder + 'hist_speed.npy')
    speed = np.load(folder + 'PRS_speed.npy')[0]
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
    plt.savefig(folder + 'hist_summary.png')

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

def WhiteNoiseMap(fits_fl, index, ref, prefix=''):
    Map = fits.open(fits_fl)[index]
    mean = np.nanmean(Map.data)
    std = np.nanstd(Map.data)
    noise = np.random.normal(mean, std, size=Map.data.shape)
    noise_full = Map.copy()
    noise_full.data = noise
    print(noise_full.data.shape)
    noise_Map = ref.copy()
    noise_Map = projectMap(noise_full, ref)

    fig = plt.figure(figsize=[12,8])
    fxx = aplpy.FITSFigure(noise_Map, figure=fig, slices=[0])
    fxx.show_colorscale(interpolation='nearest', cmap='binary')
    plt.savefig(prefix + 'NoiseMap.png')
    return noise_Map

#ALMACont('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/13CS_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/13CS/', project=False, start=0, end=13, pdfName='13CS_mom0_cube_ALMAband6.pdf', gifName='N2D_mom0_cube_ALMAband6.gif', band=0)
# ALMACont('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/CO_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/CO/', project=False, start=0, end=13, pdfName='CO_mom0_cube_ALMAband6.pdf', gifName='CO_mom0_cube_ALMAband6.gif', band=0)
# ALMACont('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/SiO_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/SiO/', project=False, start=0, end=13, pdfName='SiO_mom0_cube_ALMAband6.pdf', gifName='SiO_mom0_cube_ALMAband6.gif', band=0)
# ALMACont('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/C18O_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/C18O/', project=False, start=0, end=13, pdfName='C18O_mom0_cube_ALMAband6.pdf', gifName='C18O_mom0_cube_ALMAband6.gif', band=0)
# ALMACont('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/H2CO_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/H2CO/', project=False, start=0, end=13, pdfName='H2CO_mom0_cube_ALMAband6.pdf', gifName='H2CO_mom0_cube_ALMAband6.gif', band=0)
# ALMACont('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/13CN_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/13CN/', project=False, start=0, end=13, pdfName='13CN_mom0_cube_ALMAband6.pdf', gifName='13CN_mom0_cube_ALMAband6.gif', band=0)
# ALMACont('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/N2D_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/N2D/', project=False, start=-3, end=3, pdfName='N2D_mom0_cube_ALMAband6.pdf', gifName='N2D_mom0_cube_ALMAband6.gif', band=0)

#plotCube('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/SiO_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/SiO/', project=False, start=0, end=13, pdfName='mom0_cube_ALMASiO_contoursACA.pdf', gifName='mom0_cube_ALMASiO_contoursACA.gif')
#runAnalysis(HAWC, outName='HAWC_polI/', Mask=None, label='HAWC+ Band C Polarized I', index=13, project=False, band=0, highRes=False, kstep=2)
#runAnalysis(HAWE, outName='HAWE_polI/', Mask=None, label='HAWC+ Band E Polarized I', index=13, project=False, band=1, highRes=False, kstep=2)

#plotPRS_speed('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/ALMA/cubes/C18O/')
#plotPRS_speed('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/ALMA/cubes/13CN/')
#plotHIST_speed('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/ALMA/cubes/13CN/')
#plotHIST_speed('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/ALMA/cubes/13CS/')
#binHerschel()
#runCube('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_13CO32.fits', 0)
#gifCube('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/ALMA/cubes/SiO/', 'DataMask.pdf', 'DataMask.gif')
#gifCube('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/ALMA/cubes/C18O/', 'hist.pdf', 'hist.gif')
#runAll()
#plotCube('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_13CO32.fits', 0, '13CO_Map.pdf', '13CO_Map.gif', folder='Output_Plots/BandC/13CO/Cube/')

#plotCube('/Users/akankshabij/Document2s/MSc/Research/Data/CO_LarsBonne/CII/07_0077_RCW36_CII_L.fits', folder='Output_Plots/BandE/CII/Cube/', project=True, start=-5, end=17, pdfName='mom0_cube.pdf', gifName='mom0_cube.gif', band=1)
#calcNoise(fits_fl='/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CII/07_0077_RCW36_CII_L.fits', fldr='Output_Plots/BLASTpol/CII/Cube/', start=17, end=22, project=False)
#runCube(fits_fl='/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CII/07_0077_RCW36_CII_L.fits', folder='Output_Plots/BandC/CII/Cube/', out='CII/Cube/', start=-1, end=16, project=True, highRes=False, band=0)
#gifCube('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BLASTpol/CII/Cube/')
#plotPRS_speed('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/CII/Cube/')
#plotHIST_speed('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BLASTpol/CII/Cube/')
# plotPRS_Bands('CII/')
# plotPRS_Bands('12CO/')
# plotPRS_Bands('13CO/')
# plotPRS_Bands('Mopra/')
#totalHIST('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/CII/Cube/')

#plotCube('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_12CO32.fits', folder='Output_Plots/BandE/12CO/Cube/', project=True, start=-3, end=17, pdfName='mom0_cube.pdf', gifName='mom0_cube.gif', band=1)
# plotCube('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/O/RCW36_OI_30_15.fits', folder='Output_Plots/BandC/OI/Cube/', project=True, start=-3, end=17, pdfName='mom0_cube.pdf', gifName='mom0_cube.gif', band=0)
# plotCube('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/O/RCW36_OI_30_15.fits', folder='Output_Plots/BLASTpol/OI/Cube/', project=False, start=-3, end=17, pdfName='mom0_cube.pdf', gifName='mom0_cube.gif', band=2)
# plotCube('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/O/RCW36_OI_30_15.fits', folder='Output_Plots/BandE/OI/Cube/', project=True, start=-3, end=17, pdfName='mom0_cube.pdf', gifName='mom0_cube.gif', band=1)
# mom0_1km('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_12CO32.fits', folder='Output_Plots/BandC/12CO/Cube/', project=True, start=-3, end=17, pdfName='12CO_mom0_cube_1kms.pdf', gifName='12CO_mom0_cube_1kms.gif', band=0)
# mom0_1km('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_13CO32.fits', folder='Output_Plots/BandC/13CO/Cube/', project=True, start=-3, end=17, pdfName='13CO_mom0_cube_1kms.pdf', gifName='13CO_mom0_cube_1kms.gif', band=0)
# mom0_1km('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CII/07_0077_RCW36_CII_L.fits', folder='Output_Plots/BandC/CII/Cube/', project=True, start=-3, end=17, pdfName='CII_mom0_cube_1kms.pdf', gifName='CII_mom0_cube_kms.gif', band=0)
# mom0_1km('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/13CS_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/13CS/', project=False, start=0, end=13, pdfName='mom0_cube_ALMA13CS.pdf', gifName='mom0_cube_ALMA13CS.gif')
# mom0_1km('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/CO_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/CO/', project=False, start=0, end=13, pdfName='mom0_cube.pdf', gifName='mom0_cube.gif')
# mom0_1km('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/SiO_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/SiO/', project=False, start=0, end=13, pdfName='mom0_cube_ALMASiO.pdf', gifName='mom0_cube_ALMASiO.gif')
# mom0_1km('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/C18O_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/C18O/', project=False, start=0, end=13, pdfName='mom0_cube.pdf', gifName='mom0_cube.gif')
# mom0_1km('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/H2CO_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/H2CO/', project=False, start=0, end=13, pdfName='mom0_cube.pdf', gifName='mom0_cube.gif')
# mom0_1km('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/13CN_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/13CN/', project=False, start=0, end=13, pdfName='mom0_cube.pdf', gifName='mom0_cube.gif')
# mom0_1km('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/N2D_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/N2D/', project=False, start=-3, end=3, pdfName='mom0_cube_ALMA.pdf', gifName='mom0_cube_ALMA.gif')
# mom0_1km('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_13CO32.fits', folder='Output_Plots/BandC/13CO/Cube/', project=True, start=-3, end=17, pdfName='13CO_mom0_cube_1kms.pdf', gifName='13CO_mom0_cube_kms.gif', band=0)
# mom0_1km('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CII/07_0077_RCW36_CII_L.fits', folder='Output_Plots/BandC/CII/Cube/', project=True, start=-3, end=17, pdfName='CII_mom0_cube_1kms.pdf', gifName='CII_mom0_cube_kms.gif', band=0)

#ALMACont('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_13CO32.fits', folder='Output_Plots/BandC/13CO/Cube/', project=True, start=-3, end=17, pdfName='13CO_mom0_ALMA13CS.pdf', gifName='13CO_mom0_ALMA13CS.gif', band=0)
# calcNoise(fits_fl='/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_12CO32.fits', fldr='Output_Plots/BandE/12CO/Cube/', start=-3, end=-1, project=True, band=1)
#runCube(fits_fl='/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_12CO32.fits', folder='Output_Plots/BandC/12CO/Cube/', out='12CO/Cube/', start=-2, end=17, project=True, highRes=False, band=0)
#gifCube('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/12CO/Cube/')
#plotPRS_speed('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/12CO/Cube/')
#plotHIST_speed('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/12CO/Cube/')
#totalHIST('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/12CO/Cube/')

#plotCube('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_13CO32.fits', folder='Output_Plots/BandE/13CO/Cube/', project=True, start=-3, end=17, pdfName='mom0_cube.pdf', gifName='mom0_cube.gif', band=1)
# calcNoise(fits_fl='/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_13CO32.fits', fldr='Output_Plots/BLASTpol/13CO/Cube/', start=-3, end=0, project=False)
# runCube(fits_fl='/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_13CO32.fits', folder='Output_Plots/BLASTpol/13CO/Cube/', out='13CO/Cube/', start=-2, end=17, project=False, highRes=False, band=2)
# gifCube('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BLASTpol/13CO/Cube/')
# plotPRS_speed('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BLASTpol/13CO/Cube/')
# plotHIST_speed('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BLASTpol/13CO/Cube/')

# plotCube('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_13CO32.fits', folder='Output_Plots/BandC/13CO/Cube/', project=True, start=-3, end=17, pdfName='mom0_cube.pdf', gifName='mom0_cube.gif', band=0)
# calcNoise(fits_fl='/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_13CO32.fits', fldr='Output_Plots/BandC/13CO/Cube/', start=-3, end=0, project=True, band=0)
# runCube(fits_fl='/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_13CO32.fits', folder='Output_Plots/BandC/13CO/Cube/', out='13CO/Cube/', start=-2, end=17, project=True, highRes=False, band=0)
# gifCube('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/13CO/Cube/')
# plotPRS_speed('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/13CO/Cube/')
# plotHIST_speed('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/13CO/Cube/')
# totalHIST('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/13CO/Cube/')

# plotCube('/Users/akankshabij/Documents/MSc/Research/Data/Mopra/HNC_3mm_Vela_C_T_MB.fits', folder='Output_Plots/BandC/Mopra/Cube/', project=True, start=-3, end=17, pdfName='mom0_cube.pdf', gifName='mom0_cube.gif', band=0)
# calcNoise(fits_fl='/Users/akankshabij/Documents/MSc/Research/Data/Mopra/HNC_3mm_Vela_C_T_MB.fits', fldr='Output_Plots/BandC/Mopra/Cube/', start=-3, end=-1, project=True, band=0)
# runCube(fits_fl='/Users/akankshabij/Documents/MSc/Research/Data/Mopra/HNC_3mm_Vela_C_T_MB.fits', folder='Output_Plots/BandC/Mopra/Cube/', out='Mopra/Cube/', start=-2, end=17, project=True, highRes=False, band=0)
# gifCube('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/Mopra/Cube/')
# plotPRS_speed('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/Mopra/Cube/')
# plotHIST_speed('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/Mopra/Cube/')
# totalHIST('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/Mopra/Cube/')

# runMaskCube(fits_fl='/Users/akankshabij/Documents/MSc/Research/Data/ALMA/13CN_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/13CN/', out='ALMA/cubes/13CN/', regions=[[150, 70], [40, 100], [180, 380], [230, 200]], start=0, end=13, project=False, highRes=True, band=0)
# gifCube(fldr='Output_Plots/BandC/ALMA/cubes/13CN/')
# plotPRS_speed('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/ALMA/cubes/13CN/')
# plotHIST_speed('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/ALMA/cubes/13CN/')
# totalHIST('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/ALMA/cubes/13CN/')
#runAnalysis(fits_fl='Data/Observations/CII_Integrated.fits', outName='CII/', index=0, project=False, kernel='Gaussian', kstep=3, isSim=False, band=2, highRes=False, file=True, Map=None, BinMap=None)
#plotCube('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/CO_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/CO/', project=False, start=0, end=13, pdfName='mom0_cube.pdf', gifName='mom0_cube.gif')
#plotCube('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/H2CO_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/H2CO/', project=False, start=0, end=13, pdfName='mom0_cube.pdf', gifName='mom0_cube.gif')
#plotCube('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/13CN_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/13CN/', project=False, start=0, end=13, pdfName='mom0_cube.pdf', gifName='mom0_cube.gif')
#plotCube('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/N2D_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/N2D/', project=False, start=-3, end=3, pdfName='mom0_cube_ALMA.pdf', gifName='mom0_cube_ALMA.gif')
#plotCube('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/13CS_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/13CS/', project=False, start=0, end=13, pdfName='mom0_cube_ALMA.pdf', gifName='mom0_cube_ALMA.gif')
#runMaskCube(fits_fl='/Users/akankshabij/Documents/MSc/Research/Data/ALMA/N2D_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/N2D/', out='ALMA/cubes/N2D/', regions=[[150, 70], [40, 100], [180, 380], [230, 200]], start=-3, end=3, project=False, highRes=True, band=0)
# gifCube(fldr='Output_Plots/BandC/ALMA/cubes/N2D/')
# plotPRS_speed('Output_Plots/BandC/ALMA/cubes/N2D/')
# plotHIST_speed('Output_Plots/BandC/ALMA/cubes/N2D/')
# totalHIST('Output_Plots/BandC/ALMA/cubes/N2D/')
#calcNoise(fits_fl='/Users/akankshabij/Documents/MSc/Research/Data/ALMA/CO_Cube.fits', fldr='Output_Plots/BandC/ALMA/cubes/CO/', start=-9, end=-7)
#calcNoise(fits_fl='/Users/akankshabij/Documents/MSc/Research/Data/ALMA/C18O_Cube.fits', fldr='Output_Plots/BandC/ALMA/cubes/C18O/', start=0, end=2)
#runCube(fits_fl='/Users/akankshabij/Documents/MSc/Research/Data/ALMA/SiO_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/SiO/', out='ALMA/cubes/SiO/', start=0, end=13)
#gifCube('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/ALMA/cubes/SiO/')
#runCube(fits_fl='/Users/akankshabij/Documents/MSc/Research/Data/ALMA/C18O_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/C18O/', out='ALMA/cubes/C18O/', start=0, end=13)
#runCube(fits_fl='/Users/akankshabij/Documents/MSc/Research/Data/ALMA/13CN_Cube.fits', folder='Output_Plots/BandC/ALMA/cubes/13CN/', out='ALMA/cubes/13CN/', start=0, end=13)
#gifCube('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/ALMA/cubes/13CN/')
#runBinHersch()
#runCube(fits_fl='/Users/akankshabij/Documents/MSc/Research/Data/ALMA/Cycle8/VelaC_CR1_12m_C43_Band6_12CO_cube.fits', folder='Output_Plots/BandC/ALMA/cubes/Cycle8/12CO/', out='ALMA/cubes/Cycle8/12CO/', project=False, start=0, end=13, band=0)
#calcNoise('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_13CO32.fits', 0, fldr='Output_Plots/BandC/13CO/Cube/', start=-1, end=2)
#runChannelMaps(fl='/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_12CO32.fits', fldr='12CO/')
#runNoise('Data/Observations/Herschel_ColumnDensity.fits', outName='ColumnDensity/', label='Herschel Column Density', index=1, project=True, kernel='Gaussian', kstep=5, isSim=False, band=0, highRes=False)

# loopNoise('Data/Observations/Herschel_ColumnDensity.fits', outName='ColumnDensity/', counter=10, label='Herschel Column Density', index=1, kstep=5)
# loopNoise('Data/Observations/Hershel_Temperature.fits', outName='Temperature/', counter=10, label='Herschel Temperature', kstep=5)
# loopNoise('Data/Observations/12CO_Integrated.fits', outName='12CO/', counter=10, label='12CO', kstep=2)
# loopNoise('Data/Observations/13CO_Integrated.fits', outName='13CO/', counter=10, label='13CO', kstep=2)
# loopNoise('Data/Observations/CII_Integrated.fits',  outName='CII/',  counter=10, label='CII',  kstep=2)
# loopNoise('Data/Observations/ALMA_Band6_continuum.fits', outName='ALMA/', counter=10, label='ALMA Band 6 Continuum', kstep=1)
# loopNoise('/Users/akankshabij/Documents/MSc/Research/Data/Mopra/Integrated_mom0_0to12kmpers.fits', outName='Mopra/', counter=10, label='Mopra HNC', kstep=5)
# loopNoise('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits', outName='Highres_ColumnDensity/', counter=10, label='Highres Herschel Column Density', kstep=5)
# loopNoise('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresTmap_velac_temperature_cf_r500_medsmo3.fits', outName='Highres_Temperature/', counter=10, label='Highres Herschel Temperature', kstep=5)
# loopNoise('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits', outName='HAWC_BandC/', counter=10, label='HAWC Band C Intensity', kstep=1)
# loopNoise('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits', outName='HAWC_BandE/', counter=10, label='HAWC Band E Intensity', kstep=1)

# loopNoise('Data/Observations/Herschel_ColumnDensity.fits', outName='ColumnDensity/', counter=100, label='Herschel Column Density', index=1, kstep=5)
# loopNoise('Data/Observations/Hershel_Temperature.fits', outName='Temperature/', counter=100, label='Herschel Temperature', kstep=5)
# loopNoise('Data/Observations/12CO_Integrated.fits', outName='12CO/', counter=100, label='12CO', kstep=2)
# loopNoise('Data/Observations/13CO_Integrated.fits', outName='13CO/', counter=100, label='13CO', kstep=2)
# loopNoise('Data/Observations/CII_Integrated.fits',  outName='CII/',  counter=100, label='CII',  kstep=2)
# loopNoise('Data/Observations/ALMA_Band6_continuum.fits', outName='ALMA/', counter=100, label='ALMA Band 6 Continuum', kstep=1)
# loopNoise('/Users/akankshabij/Documents/MSc/Research/Data/Mopra/Integrated_mom0_0to12kmpers.fits', outName='Mopra/', counter=100, label='Mopra HNC', kstep=5)
# loopNoise('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits', outName='Highres_ColumnDensity/', counter=100, label='Highres Herschel Column Density', kstep=5)
# loopNoise('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresTmap_velac_temperature_cf_r500_medsmo3.fits', outName='Highres_Temperature/', counter=100, label='Highres Herschel Temperature', kstep=5)
# loopNoise('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits', outName='HAWC_BandC/', counter=100, label='HAWC Band C Intensity', kstep=1)
# loopNoise('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits', outName='HAWC_BandE/', counter=100, label='HAWC Band E Intensity', kstep=1)

# loopNoise('Data/Observations/Herschel_ColumnDensity.fits', outName='ColumnDensity/', counter=1000, label='Herschel Column Density', index=1, kstep=5)
# loopNoise('Data/Observations/Hershel_Temperature.fits', outName='Temperature/', counter=1000, label='Herschel Temperature', kstep=5)
# loopNoise('Data/Observations/12CO_Integrated.fits', outName='12CO/', counter=1000, label='12CO', kstep=2)
# loopNoise('Data/Observations/13CO_Integrated.fits', outName='13CO/', counter=1000, label='13CO', kstep=2)
# loopNoise('Data/Observations/CII_Integrated.fits',  outName='CII/',  counter=1000, label='CII',  kstep=2)
# loopNoise('Data/Observations/ALMA_Band6_continuum.fits', outName='ALMA/', counter=1000, label='ALMA Band 6 Continuum', kstep=1)
# loopNoise('/Users/akankshabij/Documents/MSc/Research/Data/Mopra/Integrated_mom0_0to12kmpers.fits', outName='Mopra/', counter=1000, label='Mopra HNC', kstep=5)
# loopNoise('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits', outName='Highres_ColumnDensity/', counter=1000, label='Highres Herschel Column Density', kstep=5)
# loopNoise('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresTmap_velac_temperature_cf_r500_medsmo3.fits', outName='Highres_Temperature/', counter=1000, label='Highres Herschel Temperature', kstep=5)
# loopNoise('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits', outName='HAWC_BandC/', counter=1000, label='HAWC Band C Intensity', kstep=1)
# loopNoise('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits', outName='HAWC_BandE/', counter=1000, label='HAWC Band E Intensity', kstep=1)

# WhiteNoise_PRS(outName='ColumnDensity/', counter=1000)
# WhiteNoise_PRS(outName='Temperature/', counter=1000)
# WhiteNoise_PRS(outName='12CO/', counter=1000)
# WhiteNoise_PRS(outName='13CO/', counter=1000)
# WhiteNoise_PRS(outName='CII/', counter=1000)
# WhiteNoise_PRS(outName='ALMA/', counter=1000)
# WhiteNoise_PRS(outName='Mopra/', counter=1000)
# WhiteNoise_PRS(outName='Highres_ColumnDensity/', counter=1000)
# WhiteNoise_PRS(outName='Highres_Temperature/', counter=1000)
# WhiteNoise_PRS(outName='HAWC_BandC/', counter=1000)
# WhiteNoise_PRS(outName='HAWC_BandE/', counter=1000)

# HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
# Nmap = 'Data/Observations/Herschel_ColumnDensity.fits'
# WhiteNoiseMap(Nmap, 1, HAWC[0], 'Output_Plots/WhiteNoise/ColumnDensity/')
#Load in simulation data
# Athena = fits.open('Data/Simulations/StrongBField/L1M10_Q.fits')
# Athena_mask = Athena[0].copy()
# Athena_mask.data = np.ones(Athena_mask.shape)
# Athena_col = fits.open('Data/Simulations/StrongBField/L1M10_colN.fits')[0]
# Qmap = (Athena[0]).data
# Umap = (fits.open('Data/Simulations/StrongBField/L1M10_U.fits')[0]).data
# hro = h.HRO(Athena_col.data, Qmap, Umap, hdu=Athena_mask, vecMask=Athena_mask, msk=True)   
# prefix = 'Output_Plots/BandC/Athena/StrongBField/'
# makePlots(hro, prefix, isSim=True, label='')
#Athena_StrongB = runAnalysis('Data/Simulations/StrongBField/L1M10_colN.fits',  outName='Athena/StrongBField/', project=False, Mask=Athena_mask, isSim=True, label='Athena - Strong Field')  

# Qmap = (fits.open('Data/Simulations/SuperAlfvenic/L10_Q.fits')[0]).data
# Umap = (fits.open('Data/Simulations/SuperAlfvenic/L10_U.fits')[0]).data
# Athena_SuperAlf = runAnalysis('Data/Simulations/SuperAlfvenic/L10_colN.fits',  prefix='Output_Plots/Athena/SuperAlfvenic/', project=False, Mask=Athena_mask, isSim=True, label='Athena - Super Alfvenic')  



