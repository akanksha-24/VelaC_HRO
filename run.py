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

def runAll():
    Hersch = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits')[0]
    HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
    HAWE = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits')
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

def runCube(fits_fl, index):
    data = fits.open(fits_fl)[index]
    speed = np.round(np.linspace(3,10,71), 1)
    cube = SpectralCube.read(data)
    noise = fits.open('Output_Plots/BandC/13CO/Cube/NoiseMap_std.fits')[0]
    HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')

    for i in speed:
        slab = cube.spectral_slab(i*u.km/u.s, (i+0.1)*u.km/u.s)
        mom0 = slab.moment0()
        mom0_HAWC = projectMap(mom0.hdu, HAWC[0])
        fldr = '{0}_kmpers/'.format(i)
        outName = '13CO/Cube/' + fldr
        label = '13CO {0}-{1} km/s'.format(i, i+0.1)
        if os.path.exists('Output_Plots/BandC/13CO/Cube/' + fldr)==False:
            os.mkdir('Output_Plots/BandC/13CO/Cube/' + fldr)
        Mask = HAWC[0].copy()
        Mask.data = np.ones(HAWC[0].data.shape)
        Mask.data[(mom0_HAWC / noise.data) < 3]=0
        fig = plt.figure(figsize=[12,8])
        fxx = aplpy.FITSFigure(Mask, figure=fig)
        fxx.show_colorscale(interpolation='nearest')
        fxx.add_colorbar()
        plt.savefig('Output_Plots/BandC/13CO/Cube/' + fldr + 'DataMask.png')
        # for i in range(mom0_HAWC.shape[0]):
        #     for j in range(mom0_HAWC.shape[1]):
        #         if (mom0_HAWC[i,j] / noise.data[i,j]) < 3:
        #             Mask.data[i,j]=0
        runAnalysis('', outName=outName, Mask=Mask, label=label, index=0, project=False, kernel='Gaussian', kstep=3, isSim=False, band=0, highRes=False, file=False, Map=mom0_HAWC)

def gifCube(fldr, pdfName, gifName):
    speed = np.round(np.linspace(3,10,71), 1)
    #cube = SpectralCube.read(fitsfl)
    images = []
    #_pdf = pdf.PdfPages('Output_Plots/BandC/12CO/Cube/' + pdfName)
    for i in speed:
        folder = fldr + '{0}_kmpers/'.format(i)
        #hist = folder + 'MASKED__phi_secthistogram.png'
        hist = folder + 'MASKED_phi.png'
        #hist = folder + 'DataMask.png'
        images.append(imageio.imread(hist))
        #_pdf.savefig(hist)
    #_pdf.close()
    imageio.mimsave('Output_Plots/BandC/13CO/Cube/' + gifName, images, fps=2)

def plotCube(fits_fl, index, pdfName, gifName, folder):
    data = fits.open(fits_fl)[index]
    speed = np.round(np.linspace(-1,13,141), 1)
    cube = SpectralCube.read(data)
    _pdf = pdf.PdfPages(folder + pdfName)
    images = []
    HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
    for i in speed:
        slab = cube.spectral_slab(i*u.km/u.s, (i+0.1)*u.km/u.s)
        mom0 = slab.moment0()
        mom0_HAWC = projectMap(mom0.hdu, HAWC[0])
        fig = plt.figure(figsize=[12,8])
        fxx = aplpy.FITSFigure(mom0_HAWC, figure=fig)
        fxx.show_colorscale(interpolation='nearest')
        fxx.add_colorbar()
        fxx.add_label(0.85, 0.85, text='{0} km/s'.format(i), relative=True, color='white', size='xx-large', weight='demibold')
        fldr = '{0}_kmpers/'.format(i)
        outName = folder + fldr
        if os.path.exists(outName)==False:
            os.mkdir(outName)
        figName = outName + 'mom0.png'
        plt.savefig(figName)
        images.append(imageio.imread(figName))
        _pdf.savefig(fig)
    _pdf.close()
    imageio.mimsave(folder + gifName, images, fps=2)

def calcNoise(fits_fl, index, fldr, start=-1, end=1):
    data = fits.open(fits_fl)[index]
    speed = np.round(np.linspace(start,end,21), 1)
    cube = SpectralCube.read(data)
    #_pdf = pdf.PdfPages('Output_Plots/BandC/12CO/Cube/' + pdfName)
    #images = []
    maps = []
    HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
    for i in speed:
        print(i)
        slab = cube.spectral_slab(i*u.km/u.s, (i+0.1)*u.km/u.s)
        mom0 = slab.moment0()
        mom0_HAWC = projectMap(mom0.hdu, HAWC[0])
        maps.append(mom0_HAWC)
        # fldr = '{0}_kmpers/'.format(i)
        # outName = 'Output_Plots/BandC/12CO/Cube/' + fldr
        # if os.path.exists(outName)==False:
        #     os.mkdir(outName)
    maps = np.asarray(maps)
    print(maps.shape)
    std = np.std(maps,0)
    print(std.shape)
    std_Map = HAWC[0].copy()
    std_Map.data = std
    std_Map.writeto(fldr + 'NoiseMap_std.fits')
    fig = plt.figure(figsize=[12,8])
    fxx = aplpy.FITSFigure(std_Map, figure=fig)
    fxx.show_colorscale(interpolation='nearest')
    fxx.add_colorbar()
    plt.savefig(fldr + 'NoiseMap_std.png')


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
    makePlots(hro, prefix, isSim, label, BinMap)

def runAnalysis(fits_fl, outName, Mask, label, index=0, project=True, kernel='Gaussian', kstep=1, isSim=False, band=0, highRes=False, file=True, Map=None, BinMap=None):
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

    elif band==1:
        fldr = '/BandE/'
        HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits')
        vecMask = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandE_polFlux_3.fits')[0]
        Qmap = HAWC['STOKES Q'].data
        Umap = HAWC['STOKES U'].data

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
        hro = h.HRO(Map.data, Qmap, Umap, Mask, vecMask, True, kernel, kstep, BinMap=BinMap)   
        prefix=prefix + 'MASKED_'    
    makePlots(hro, prefix, isSim, label, BinMap)
    return hro

def makePlots(hro, prefix, isSim, label, BinMap=None):
    if isSim: scale=15; step=20
    else: scale=5; step=6
    if BinMap is None:
        label=label
    else:
        label=label+' Binned to Column Density'

    #p.plot_Fields(hro, prefix, Bfield=True, Efield=True, step=step, scale=scale)
    p.plot_Map(hro, prefix='', norm=False)
    p.plot_Gradient(hro, prefix, norm=False)
    p.plot_GradientAmp(hro, prefix, norm=True)
    p.plot_vectors(hro, prefix, step=step, scale=scale)
    p.plot_regions(hro, prefix, step=step, scale=scale)
    p.plot_phi(hro, prefix, step=step, scale=scale)
    p.plot_histShaded(hro, label, prefix)
    #p.plot_FEWsect(hro, label, prefix)
    p.plot_secthist(hro, label, prefix)

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

#binHerschel()
runCube('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_13CO32.fits', 0)
#gifCube('/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/BandC/13CO/Cube/', '13CO_phi.pdf', '13CO_phi.gif')
#runAll()
#plotCube('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_13CO32.fits', 0, '13CO_Map.pdf', '13CO_Map.gif', folder='Output_Plots/BandC/13CO/Cube/')
#runBinHersch()
#calcNoise('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_13CO32.fits', 0, fldr='Output_Plots/BandC/13CO/Cube/', start=-1, end=2)
#runChannelMaps(fl='/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_12CO32.fits', fldr='12CO/')
#runNoise('Data/Observations/Herschel_ColumnDensity.fits', outName='ColumnDensity/', label='Herschel Column Density', index=1, project=True, kernel='Gaussian', kstep=5, isSim=False, band=0, highRes=False)
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



