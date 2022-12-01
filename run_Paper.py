import numpy as np
from astropy.io import fits
from reproject import reproject_interp, reproject_exact
import astropy.units as u
import astropy.constants as c
import HRO as h
import plots as p
import run_General as r
import os
from scipy import ndimage
import aplpy
import matplotlib.pyplot as plt

# import maps
folder = '/Users/akankshabij/Documents/MSc/Research/Data/'
HAWC = fits.open(folder + 'HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
HAWE = fits.open(folder + 'HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits')
vecMask_C = fits.open(folder + 'HAWC/masks/rereduced/BandC_polFlux_3.fits')[0]
vecMask_E = fits.open(folder + 'HAWC/masks/rereduced/BandE_polFlux_3.fits')[0]

CO12 = fits.open(folder + 'CO_LarsBonne/CO/RCW36_integratedIntensity12CO.fits')[0]
CO13 = fits.open(folder + 'CO_LarsBonne/CO/RCW36_integratedIntensity13CO.fits')[0]
CII = fits.open(folder + 'CO_LarsBonne/CII/RCW36_integratedIntensityCII.fits')[0]
Spitz_Ch1 = fits.open(folder + 'Spitzer/RCW36-4-selected_Post_BCDs/r15990016/ch1/pbcd/SPITZER_I1_15990016_0000_3_E8591943_maic.fits')[0]
Spitz_Ch3 = fits.open(folder + 'Spitzer/RCW36-4-selected_Post_BCDs/r15990016/ch3/pbcd/SPITZER_I3_15990016_0000_3_E8592062_maic.fits')[0]
#ALMA_12m = fits.open(folder + 'ALMA/')
ALMA_ACA = fits.open(folder + 'ALMA/VelaC_CR_ALMA2D.fits')[0]
OI = fits.open(folder + 'CO_LarsBonne/O/RCW36_OI_Integrated.fits')[0]
Halpha = fits.open(folder + 'Halpha/Halpha_corrHead.fits')[0]

#Herschel
Ncol = fits.open(folder + 'Herschel/HighresNmap_herschel_velac_high_av.fits')[0]
Ncol_Low = fits.open(folder + 'Herschel/VelaTvN_w160_bgSUB_PLWRes_20150920_Nmap_masked.fits')[1]
Temp_High = fits.open(folder + 'Herschel/HighresTmap_velac_temperature_cf_r500_medsmo3.fits')[0]
Temp_Low = fits.open(folder + 'Herschel/VelaTvN_w160_bgSUB_PLWRes_20150920_Tmap.fits')[0]
H_70 = fits.open(folder + 'Herschel/vela_herschel_70_flat.fits')[0]
H_160 = fits.open(folder + 'Herschel/vela_herschel_160_flat.fits')[0]
H_250 = fits.open(folder + 'Herschel/vela_herschel_hipe11_250.fits')[0]
H_350 = fits.open(folder + 'Herschel/vela_herschel_hipe11_350_10as.fits')[0]
H_500 = fits.open(folder + 'Herschel/vela_herschel_hipe11_500.fits')[0]

#Mopra
Mop_HNC_less = fits.open(folder + 'Mopra/HNC_Integrated_mom0_0to12kmpers.fits')[0]
Mop_HNC_more = fits.open(folder + 'Mopra/HNC_Integrated_mom0_n10to30kmpers.fits')[0]
Mop_N2H = fits.open(folder + 'Mopra/N2H_Integrated_mom0_n10to30kmpers.fits')[0]
Mop_C18O = fits.open(folder + 'Mopra/C18O_Integrated_mom0_n10to30kmpers.fits')[0]

#BLASTPol Maps
BLAST_Q = fits.open(folder + 'BLASTPol/Estimate_Q.fits')[0]
BLAST_U = fits.open(folder + 'BLASTPol/Estimate_U.fits')[0]
BLAST_mask = fits.open(folder + 'BLASTPol/BLAST_mask.fits')[0]
BLAST_Bmap = fits.open(folder + 'BLASTPol/BLASTPol_500_intermediate_BPOSang.fits')[0]

#BLASTPol Cutouts
CO12_B = fits.open(folder + 'Cutouts/12CO.fits')[0]
H70_B = fits.open(folder + 'Cutouts/Hersh_70.fits')[0]
Mop_HNC_B = fits.open(folder + 'Cutouts/Mopra_HNC.fits')[0]
Ncol_B = fits.open(folder + 'Cutouts/colDen.fits')[0]

def runHRO(Map, Bmap, kstep, vecMask, location, Bproj):
    hro = h.HRO(Map, Qmap=None, Umap=None, Bmap=Bmap, hdu=vecMask, vecMask=vecMask, msk=True, kstep=kstep, Bproj=Bproj) 
    prefix='/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/' + location
    if os.path.exists(prefix)==False:
        os.mkdir(prefix)
    prefix=prefix+'kstep{0}/'.format(kstep)
    if os.path.exists(prefix)==False:
        os.mkdir(prefix)
    r.makePlots(hro, prefix, isSim=False, label='')

def runHRO_QU(Map, Qmap, Umap, kstep, vecMask, location):
    hro = h.HRO(Map, Qmap=Qmap, Umap=Umap, Bmap=None, hdu=vecMask, vecMask=vecMask, msk=True, kstep=kstep) 
    prefix='/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/' + location
    if os.path.exists(prefix)==False:
        os.mkdir(prefix)
    prefix=prefix+'kstep{0}/'.format(kstep)
    if os.path.exists(prefix)==False:
        os.mkdir(prefix)
    r.makePlots(hro, prefix, isSim=False, label='')


# def prepHAWC(Map, Bmap, ksize):
#     # 1. Smooth Map
#     #Map_smooth = Map.copy()
#     #Map_smooth.data = ndimage.filters.gaussian_filter(Map.data, [ksize, ksize], order=[0,0], mode='nearest')
#     # 2. Project Map
#     #Map_proj = projectMap(Map_smooth, Bmap)
#     # 3. Convert to Bfield angle to radians
#     Bmap.data = ((Bmap.copy().data)*u.deg).to(u.rad).value 

#     return Map, Bmap#, Map_smooth

def projectMap(mapOrigin, ref):
    '''This function projects a given map 'mapOrigin' onto the same WCS coordinates as a reference map and returns the map in the same shape'''
    New = ref.copy()
    proj, footprint = reproject_exact(mapOrigin, ref.header)
    proj[np.isnan(ref.data)] = np.nan
    proj[0].data = proj
    New.data = proj
    return New

kstep=2
runHRO(H70_B, BLAST_Bmap, kstep=kstep, vecMask=BLAST_mask, location='BLASTPol/Hersch_70/', Bproj=True)
kstep=3
runHRO(H70_B, BLAST_Bmap, kstep=kstep, vecMask=BLAST_mask, location='BLASTPol/Hersch_70/', Bproj=True)
kstep=4
runHRO(H70_B, BLAST_Bmap, kstep=kstep, vecMask=BLAST_mask, location='BLASTPol/Hersch_70/', Bproj=True)

# kstep=3
# runHRO_QU(Ncol_B, Qmap=BLAST_Q, Umap=BLAST_U, kstep=kstep, vecMask=BLAST_mask, location='BLASTPol/Ncol/') 
# kstep=4
# runHRO_QU(Ncol_B, Qmap=BLAST_Q, Umap=BLAST_U, kstep=kstep, vecMask=BLAST_mask, location='BLASTPol/Ncol/') 

# kstep=2
# runHRO_QU(ALMA_ACA, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/ALMA_cont/') 
# kstep=3
# runHRO_QU(ALMA_ACA, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/ALMA_cont/') 
# kstep=4
# runHRO_QU(ALMA_ACA, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/ALMA_cont/') 
# kstep=5
# runHRO_QU(ALMA_ACA, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/ALMA_cont/') 
# kstep=6
# runHRO_QU(ALMA_ACA, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/ALMA_cont/') 
# kstep=7
# runHRO_QU(ALMA_ACA, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/ALMA_cont/') 
# kstep=8
# runHRO_QU(ALMA_ACA, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/ALMA_cont/') 

# kstep=2
# runHRO_QU(Spitz_Ch1, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch1/')
# runHRO_QU(Spitz_Ch3, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch3/')  
# kstep=3
# runHRO_QU(Spitz_Ch1, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch1/')
# runHRO_QU(Spitz_Ch3, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch3/')  
# kstep=4
# runHRO_QU(Spitz_Ch1, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch1/')
# runHRO_QU(Spitz_Ch3, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch3/')  
# kstep=5
# runHRO_QU(Spitz_Ch1, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch1/')
# runHRO_QU(Spitz_Ch3, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch3/')   
# kstep=6
# runHRO_QU(Spitz_Ch1, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch1/')
# runHRO_QU(Spitz_Ch3, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch3/')   
# kstep=7
# runHRO_QU(Spitz_Ch1, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch1/')
# runHRO_QU(Spitz_Ch3, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch3/')  
# kstep=8
# runHRO_QU(Spitz_Ch1, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch1/')
# runHRO_QU(Spitz_Ch3, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch3/')      

#runHRO(Ncol, HAWE[11], kstep=3, vecMask=vecMask_E, location='BandE/Ncol_Av/kstep3/')
# runHRO(CO12, HAWE[11], kstep=3, vecMask=vecMask_E, location='BandE/12CO/kstep3/')
# runHRO(CO13, HAWE[11], kstep=3, vecMask=vecMask_E, location='BandE/13CO/kstep3/')
# runHRO(CII, HAWE[11], kstep=3, vecMask=vecMask_E, location='BandE/CII/kstep3/')
#runHRO(HAWE[0], HAWE[11], kstep=2, vecMask=vecMask_E, location='BandE/BandE_I/')
#runHRO(HAWC[0], HAWE[11], kstep=2, vecMask=vecMask_E, location='BandE/BandC_I/')
#runHRO(Ncol_Low, HAWE[11], kstep=3, vecMask=vecMask_E, location='BandE/Ncol_Av_Low/kstep3/')
#runHRO(Temp_Low, HAWE[11], kstep=3, vecMask=vecMask_E, location='BandE/Temp_Low/kstep3/')
#runHRO(Temp_High, HAWE[11], kstep=3, vecMask=vecMask_E, location='BandE/Temp_High/kstep3/')
# runHRO(H_70, HAWE[11], kstep=2, vecMask=vecMask_E, location='BandE/Hersh_70/kstep2/')
# runHRO(H_160, HAWE[11], kstep=2, vecMask=vecMask_E, location='BandE/Hersh_160/kstep2/')
# runHRO(H_250, HAWE[11], kstep=2, vecMask=vecMask_E, location='BandE/Hersh_250/kstep2/')
# runHRO(H_350, HAWE[11], kstep=2, vecMask=vecMask_E, location='BandE/Hersh_350/kstep2/')
# runHRO(H_500, HAWE[11], kstep=2, vecMask=vecMask_E, location='BandE/Hersh_500/kstep2/')
# runHRO(Mop_HNC_less, HAWE[11], kstep=3, vecMask=vecMask_E, location='BandE/Mopra_HNC_0to12/kstep3/')
# runHRO(Mop_HNC_more, HAWE[11], kstep=3, vecMask=vecMask_E, location='BandE/Mopra_HNC_n10to30/kstep3/')
# runHRO(Mop_C18O, HAWE[11], kstep=3, vecMask=vecMask_E, location='BandE/Mopra_C18O_n10to30/kstep3/')
# runHRO(Mop_N2H, HAWE[11], kstep=3, vecMask=vecMask_E, location='BandE/Mopra_N2H_n10to30/kstep3/')
# runHRO(Mop_HNC_less, HAWE[11], kstep=4, vecMask=vecMask_E, location='BandE/Mopra_HNC_0to12/kstep4/')
# runHRO(Mop_HNC_more, HAWE[11], kstep=4, vecMask=vecMask_E, location='BandE/Mopra_HNC_n10to30/kstep4/')
# runHRO(Mop_C18O, HAWE[11], kstep=4, vecMask=vecMask_E, location='BandE/Mopra_C18O_n10to30/kstep4/')
# runHRO(Mop_N2H, HAWE[11], kstep=4, vecMask=vecMask_E, location='BandE/Mopra_N2H_n10to30/kstep4/')
# runHRO(Spitz_Ch1, HAWE[11], kstep=3, vecMask=vecMask_E, location='BandE/Spitz_Ch1/kstep3/')
# runHRO(Spitz_Ch3, HAWE[11], kstep=3, vecMask=vecMask_E, location='BandE/Spitz_Ch3/kstep3/')
# runHRO(Spitz_Ch1, HAWE[11], kstep=4, vecMask=vecMask_E, location='BandE/Spitz_Ch1/kstep4/')
# runHRO(Spitz_Ch3, HAWE[11], kstep=4, vecMask=vecMask_E, location='BandE/Spitz_Ch3/kstep4/')
# runHRO(Spitz_Ch1, HAWE[11], kstep=6, vecMask=vecMask_E, location='BandE/Spitz_Ch1/kstep6/')
# runHRO(Spitz_Ch3, HAWE[11], kstep=6, vecMask=vecMask_E, location='BandE/Spitz_Ch3/kstep6/')
# runHRO(OI, HAWE[11], kstep=3, vecMask=vecMask_E, location='BandE/OI/kstep3/')
# runHRO(Halpha, HAWE[11], kstep=3, vecMask=vecMask_E, location='BandE/Halpha/kstep3/')
# runHRO(ALMA_ACA, HAWE[11], kstep=3, vecMask=vecMask_E, location='BandE/ALMA_cont/kstep3/')
# runHRO(OI, HAWE[11], kstep=4, vecMask=vecMask_E, location='BandE/OI/kstep4/')
# runHRO(Halpha, HAWE[11], kstep=4, vecMask=vecMask_E, location='BandE/Halpha/kstep4/')
# runHRO(ALMA_ACA, HAWE[11], kstep=4, vecMask=vecMask_E, location='BandE/ALMA_cont/kstep4/')

#kstep=7
# runHRO(Ncol, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Ncol_Av/')
# runHRO(CO12, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/12CO/')
# runHRO(CO13, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/13CO/')
# runHRO(CII, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/CII/')
# runHRO(HAWE[0], HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/BandE_I/')
# runHRO(HAWC[0], HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/BandC_I/')
# runHRO(Ncol_Low, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Ncol_Av_Low/')
# runHRO(Temp_Low, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Temp_Low/')
# runHRO(Temp_High, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Temp_High/')
# runHRO(H_70, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Hersh_70/')
# runHRO(H_160, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Hersh_160/')
# runHRO(H_250, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Hersh_250/')
# runHRO(H_350, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Hersh_350/')
# runHRO(H_500, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Hersh_500/')
# runHRO(Mop_HNC_less, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Mopra_HNC_0to12/')
# runHRO(Mop_HNC_more, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Mopra_HNC_n10to30/')
# runHRO(Mop_C18O, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Mopra_C18O_n10to30/')
# runHRO(Mop_N2H, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Mopra_N2H_n10to30/')
# runHRO(Spitz_Ch1, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch1/')
# runHRO(Spitz_Ch3, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch3/')
# #runHRO(OI, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/OI/')
# runHRO(Halpha, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Halpha/')
# runHRO(ALMA_ACA, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/ALMA_cont/')
# kstep=6
# runHRO(Ncol, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Ncol_Av/')
# runHRO(CO12, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/12CO/')
# runHRO(CO13, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/13CO/')
# runHRO(CII, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/CII/')
# runHRO(HAWE[0], HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/BandE_I/')
# runHRO(HAWC[0], HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/BandC_I/')
# runHRO(Ncol_Low, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Ncol_Av_Low/')
# runHRO(Temp_Low, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Temp_Low/')
# runHRO(Temp_High, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Temp_High/')
# runHRO(H_70, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Hersh_70/')
# runHRO(H_160, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Hersh_160/')
# runHRO(H_250, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Hersh_250/')
# runHRO(H_350, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Hersh_350/')
# runHRO(H_500, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Hersh_500/')
# runHRO(Mop_HNC_less, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Mopra_HNC_0to12/')
# runHRO(Mop_HNC_more, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Mopra_HNC_n10to30/')
# runHRO(Mop_C18O, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Mopra_C18O_n10to30/')
# runHRO(Mop_N2H, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Mopra_N2H_n10to30/')
# runHRO(Spitz_Ch1, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch1/')
# runHRO(Spitz_Ch3, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch3/')
#runHRO(OI, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/OI/')
# runHRO(Halpha, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Halpha/')
#runHRO(ALMA_ACA, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/ALMA_cont/')


# runHRO(HAWE[0], HAWC[11], kstep=1, vecMask=vecMask_C, location='BandC/BandE_I/kstep1/')
# runHRO(HAWE[0], HAWC[11], kstep=1.5, vecMask=vecMask_C, location='BandC/BandE_I/kstep1_5/')
# runHRO(HAWE[0], HAWC[11], kstep=2, vecMask=vecMask_C, location='BandC/BandE_I/kstep2/')
# runHRO(HAWE[0], HAWC[11], kstep=2.5, vecMask=vecMask_C, location='BandC/BandE_I/kstep2_5/')
# runHRO(HAWE[0], HAWC[11], kstep=3, vecMask=vecMask_C, location='BandC/BandE_I/kstep3/')

# runHRO(HAWC[0], HAWC[11], kstep=1, vecMask=vecMask_C, location='BandC/BandC_I/kstep1/')
# runHRO(HAWC[0], HAWC[11], kstep=1.5, vecMask=vecMask_C, location='BandC/BandC_I/kstep1_5/')
# runHRO(HAWC[0], HAWC[11], kstep=2, vecMask=vecMask_C, location='BandC/BandC_I/kstep2/')
# runHRO(HAWC[0], HAWC[11], kstep=2.5, vecMask=vecMask_C, location='BandC/BandC_I/kstep2_5/')
# runHRO(HAWC[0], HAWC[11], kstep=3, vecMask=vecMask_C, location='BandC/BandC_I/kstep3/')

# runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=1, vecMask=vecMask_C, location='BandC/12CO/kstep1/')
# runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=1.5, vecMask=vecMask_C, location='BandC/12CO/kstep1_5/')
# runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=2, vecMask=vecMask_C, location='BandC/12CO/kstep2/')
# runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=2.5, vecMask=vecMask_C, location='BandC/12CO/kstep2_5/')
# runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=3, vecMask=vecMask_C, location='BandC/12CO/kstep3/')
# runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=3.5, vecMask=vecMask_C, location='BandC/12CO/kstep3_5/')
# runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=4, vecMask=vecMask_C, location='BandC/12CO/kstep4/')
# runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=4.5, vecMask=vecMask_C, location='BandC/12CO/kstep4_5/')
# runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=5, vecMask=vecMask_C, location='BandC/12CO/kstep5/')
# runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=5.5, vecMask=vecMask_C, location='BandC/12CO/kstep5_5/')
# runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=6, vecMask=vecMask_C, location='BandC/12CO/kstep6/')

# runHRO(Ncol, HAWC[11], kstep=1, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep1/')
# runHRO(Ncol, HAWC[11], kstep=1.5, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep1_5/')
# runHRO(Ncol, HAWC[11], kstep=2, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep2/')
# runHRO(Ncol, HAWC[11], kstep=2.5, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep2_5/')
# runHRO(Ncol, HAWC[11], kstep=3, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep3/')
# runHRO(Ncol, HAWC[11], kstep=3.5, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep3_5/')
# runHRO(Ncol, HAWC[11], kstep=4, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep4/')
# runHRO(Ncol, HAWC[11], kstep=4.5, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep4_5/')
# runHRO(Ncol, HAWC[11], kstep=5, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep5/')
# runHRO(Ncol, HAWC[11], kstep=5.5, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep5_5/')
# runHRO(Ncol, HAWC[11], kstep=6, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep6/')

# runHRO(Ncol, HAWC[11], kstep=1, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep1/')
# runHRO(Ncol, HAWC[11], kstep=1.5, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep1_5/')
# runHRO(Ncol, HAWC[11], kstep=2, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep2/')
# runHRO(Ncol, HAWC[11], kstep=2.5, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep2_5/')
# runHRO(Ncol, HAWC[11], kstep=3, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep3/')
# runHRO(Ncol, HAWC[11], kstep=3.5, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep3_5/')
# runHRO(Ncol, HAWC[11], kstep=4, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep4/')
# runHRO(Ncol, HAWC[11], kstep=4.5, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep4_5/')
# runHRO(Ncol, HAWC[11], kstep=5, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep5/')
# runHRO(Ncol, HAWC[11], kstep=5.5, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep5_5/')
# runHRO(Ncol, HAWC[11], kstep=6, vecMask=vecMask_C, location='BandC/Ncol_Av/kstep6/')

