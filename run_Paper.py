import numpy as np
from astropy.io import fits
from reproject import reproject_interp, reproject_exact
import astropy.units as u
import astropy.constants as c
import HRO as h
import plots as p
import run_General as r
import os

# import maps
folder = '/Users/akankshabij/Documents/MSc/Research/Data/'

HAWC = fits.open(folder + 'HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
HAWE = fits.open(folder + 'HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits')
vecMask_C = fits.open(folder + 'HAWC/masks/rereduced/BandC_polFlux_3.fits')[0]
vecMask_E = fits.open(folder + 'HAWC/masks/rereduced/BandE_polFlux_3.fits')[0]

CO12 = fits.open(folder + 'CO_LarsBonne/CO/RCW36_integratedIntensity12CO.fits')[0]
CO13 = fits.open(folder + 'CO_LarsBonne/CO/RCW36_integratedIntensity13CO.fits')[0]
CII = fits.open(folder + 'CO_LarsBonne/CII/RCW36_integratedIntensityCII.fits')[0]
Ncol = fits.open(folder + 'Herschel/HighresNmap_herschel_velac_high_av.fits')[0]
Ncol_Low = fits.open(folder + 'Herschel/VelaTvN_w160_bgSUB_PLWRes_20150920_Nmap_masked.fits')[1]
Spitz_Ch1 = fits.open(folder + 'Spitzer/RCW36-4-selected_Post_BCDs/r15990016/ch1/pbcd/SPITZER_I1_15990016_0000_3_E8591943_maic.fits')[0]

def runHRO(Map, Bmap, kstep, vecMask, location):
    # convert to radians:
    if np.nanmax(Bmap.data) > 2*np.pi:
        Bmap.data = ((Bmap.data)*u.deg).to(u.rad).value  

    hro = h.HRO(Map.data, Qmap=None, Umap=None, Bmap=Bmap.data, hdu=vecMask, vecMask=vecMask, msk=True, kstep=kstep) 
    prefix='/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/' + location
    if os.path.exists(prefix)==False:
        os.mkdir(prefix)
    r.makePlots(hro, prefix, isSim=False, label='')

def projectMap(mapOrigin, ref):
    '''This function projects a given map 'mapOrigin' onto the same WCS coordinates as a reference map and returns the map in the same shape'''
    New = ref.copy()
    proj, footprint = reproject_exact(mapOrigin, ref.header)
    proj[np.isnan(ref.data)] = np.nan
    proj[0].data = proj
    New.data = proj
    return New

# runHRO(projectMap(HAWE[0], HAWC[11]), HAWC[11], kstep=1, vecMask=vecMask_C, location='BandC/BandE_I/kstep1/')
# runHRO(projectMap(HAWE[0], HAWC[11]), HAWC[11], kstep=1.5, vecMask=vecMask_C, location='BandC/BandE_I/kstep1_5/')
# runHRO(projectMap(HAWE[0], HAWC[11]), HAWC[11], kstep=2, vecMask=vecMask_C, location='BandC/BandE_I/kstep2/')
# runHRO(projectMap(HAWE[0], HAWC[11]), HAWC[11], kstep=2.5, vecMask=vecMask_C, location='BandC/BandE_I/kstep2_5/')
# runHRO(projectMap(HAWE[0], HAWC[11]), HAWC[11], kstep=3, vecMask=vecMask_C, location='BandC/BandE_I/kstep3/')

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

mask=HAWC[11].copy()
mask.data = np.ones(HAWC[11].data.shape)
runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=1, vecMask=mask, location='BandC/12CO/kstep1/')
runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=1.5, vecMask=mask, location='BandC/12CO/kstep1_5/')
runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=2, vecMask=mask, location='BandC/12CO/kstep2/')
runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=2.5, vecMask=mask, location='BandC/12CO/kstep2_5/')
runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=3, vecMask=mask, location='BandC/12CO/kstep3/')
runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=3.5, vecMask=mask, location='BandC/12CO/kstep3_5/')
runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=4, vecMask=mask, location='BandC/12CO/kstep4/')
runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=4.5, vecMask=mask, location='BandC/12CO/kstep4_5/')
runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=5, vecMask=mask, location='BandC/12CO/kstep5/')
runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=5.5, vecMask=mask, location='BandC/12CO/kstep5_5/')
runHRO(projectMap(CO12, HAWC[11]), HAWC[11], kstep=6, vecMask=mask, location='BandC/12CO/kstep6/')

