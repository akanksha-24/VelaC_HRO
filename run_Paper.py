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
from spectral_cube import SpectralCube
from pathlib import Path

# import maps
folder = '/Users/akankshabij/Documents/MSc/Research/Data/'
HAWC = fits.open(folder + 'HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
HAWC_Q = HAWC['STOKES Q']
HAWC_U = HAWC['STOKES U']
vecMask_C = fits.open(folder + 'HAWC/masks/rereduced/BandC_polFlux_3.fits')[0]
HAWE = fits.open(folder + 'HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits')
HAWE_Q = HAWE['STOKES Q']
HAWE_U = HAWE['STOKES U']
vecMask_E = fits.open(folder + 'HAWC/masks/rereduced/BandE_polFlux_3.fits')[0]
BandE_boundary = HAWE[0].copy()
BandE_boundary.data = np.ones(HAWE[0].shape)
BandE_boundary.data[np.isnan(HAWE[0].data)]=np.nan
BandC_boundary = HAWC[0].copy()
BandC_boundary.data = np.ones(HAWC[0].shape)
BandC_boundary.data[np.isnan(HAWC[0].data)]=np.nan

#import cubes
CO12_cube = fits.open(folder + 'CO_LarsBonne/CO/RCW36_12CO32.fits')[0]
CO13_cube = fits.open(folder + 'CO_LarsBonne/CO/RCW36_13CO32.fits')[0]
CII_cube = fits.open(folder + 'CO_LarsBonne/CII/07_0077_RCW36_CII_L.fits')[0]
OI_cube = fits.open(folder + 'CO_LarsBonne/O/RCW36_OI_30_15.fits')[0]
HNC_cube = fits.open(folder + 'Mopra/HNC_3mm_Vela_C_T_MB.fits')[0]
N2H_cube = fits.open(folder + 'Mopra/M401_3mm_N2H+_hann2_paul.fits')[0]
C18O_cube = fits.open(folder + 'Mopra/C18O_corrHead_cube.fits')[0]

CO12 = fits.open(folder + 'CO_LarsBonne/CO/RCW36_integratedIntensity12CO.fits')[0]
CO13 = fits.open(folder + 'CO_LarsBonne/CO/RCW36_integratedIntensity13CO.fits')[0]
CII = fits.open(folder + 'CO_LarsBonne/CII/RCW36_integratedIntensityCII.fits')[0]
Spitz_Ch1 = fits.open(folder + 'Spitzer/RCW36-4-selected_Post_BCDs/r15990016/ch1/pbcd/SPITZER_I1_15990016_0000_3_E8591943_maic.fits')[0]
Spitz_Ch3 = fits.open(folder + 'Spitzer/RCW36-4-selected_Post_BCDs/r15990016/ch3/pbcd/SPITZER_I3_15990016_0000_3_E8592062_maic.fits')[0]
Spitz_fix1 = fits.open(folder + 'Spitzer/CH1_factor5.fits')[0]
Spitz_fix3 = fits.open(folder + 'Spitzer/CH3_interp.fits')[0]
Spitz_rez1 = fits.open(folder + 'Spitzer/Spitz_ch1_rezmatch.fits')[0]
Spitz_rez2 = fits.open(folder + 'Spitzer/Spitz_ch2_rezmatch.fits')[0]
Spitz_rez3 = fits.open(folder + 'Spitzer/Spitz_ch3_rezmatch.fits')[0]
Spitz1_HAWC = fits.open(folder + 'Spitzer/CH1_HAWCproj.fits')[0]
Spitz3_HAWC = fits.open(folder + 'Spitzer/CH3_HAWCproj.fits')[0]

#ALMA_12m = fits.open(folder + 'ALMA/')
ALMA_ACA = fits.open(folder + 'ALMA/VelaC_CR_ALMA2D.fits')[0]
ALMA_mask = fits.open(folder + 'ALMA/ALMAcont_mask_3sig.fits')[0]
ALMA_12m = fits.open(folder + 'ALMA/Cycle8_12m/VelaC_CR1_12m_Cont_flat2D.fits')[0]
ALMA_12m_mask = fits.open(folder + 'ALMA/ALMA_12m_cont_mask_3sig.fits')[0]
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
Mop_HNC = fits.open(folder + 'Mopra/HNC_Integrated_mom0_2to10kmpers.fits')[0]
Mop_C18O = fits.open(folder + 'Mopra/C18O_Integrated_mom0_2to10kmpers.fits')[0]
Mop_N2H = fits.open(folder + 'Mopra/N2H_Integrated_mom0_5to15kmpers.fits')[0]
HNC_BandE_msk = fits.open(folder + 'Mopra/HNC_BandE_TotalI_Mask_thres4.fits')[1]
N2H_BandE_msk = fits.open(folder + 'Mopra/N2H_BandE_TotalI_Mask_thres4.fits')[1]
C18O_BandE_msk = fits.open(folder + 'Mopra/C18O_BandE_TotalI_Mask_thres4.fits')[1]
HNC_BandC_msk = fits.open(folder + 'Mopra/HNC_TotalI_Mask_thres4.fits')[1]
N2H_BandC_msk = fits.open(folder + 'Mopra/N2H_TotalI_Mask_thres4.fits')[1]
C18O_BandC_msk = fits.open(folder + 'Mopra/C18O_TotalI_Mask_thres4.fits')[1]

#BLASTPol Maps
BLAST_Q = fits.open(folder + 'BLASTPol/Estimate_Q.fits')[0]
BLAST_U = fits.open(folder + 'BLASTPol/Estimate_U.fits')[0]
BLAST_mask = fits.open(folder + 'BLASTPol/BLAST_mask.fits')[0]
BLAST_Bmap = fits.open(folder + 'BLASTPol/BLASTPol_500_intermediate_BPOSang.fits')[0]

#BLASTPol Cutouts
CO12_B = fits.open(folder + 'Cutouts/12CO.fits')[0]
CO13_B = fits.open(folder + 'Cutouts/13CO.fits')[0]
CII_B = fits.open(folder + 'Cutouts/CII.fits')[0]
OI_B = fits.open(folder + 'Cutouts/OI.fits')[0]
H500_B = fits.open(folder + 'Cutouts/Hersh_500.fits')[0]
H350_B = fits.open(folder + 'Cutouts/Hersh_350.fits')[0]
H250_B = fits.open(folder + 'Cutouts/Hersh_250.fits')[0]
H160_B = fits.open(folder + 'Cutouts/Hersh_160.fits')[0]
H70_B = fits.open(folder + 'Cutouts/Hersh_70.fits')[0]
Ncol_B = fits.open(folder + 'Cutouts/colDen.fits')[0]
Mop_HNC_B = fits.open(folder + 'Cutouts/Mopra_HNC.fits')[0]
Mop_C18O_B = fits.open(folder + 'Cutouts/Mopra_C18O.fits')[0]
Mop_N2H_B = fits.open(folder + 'Cutouts/Mopra_N2H.fits')[0]
TempHigh_B = fits.open(folder + 'Cutouts/Temp.fits')[0]
TempLow_B = fits.open(folder + 'Cutouts/Temp_LowRes.fits')[0]
NcolLow_B = fits.open(folder + 'Cutouts/ColDen_LowRes.fits')[0]
Halpha_B = fits.open(folder + 'Cutouts/Halpha.fits')[0]

def smoothing(FWHM, ref, kstep=0, frac=4, thres=2):
    if kstep==0:
        frac_beam = FWHM / frac
        pix = (frac_beam*u.arcsec).to(u.deg).value / ref.header['CDELT2']
        print('inital', pix)
        if pix < thres:
            kstep = thres
        else:
            kstep = pix

    kstep_arc = (kstep * ref.header['CDELT2']*u.deg).to(u.arcsec).value
    print('smoothing pix', kstep)
    print('smoothing arcsecs: ', kstep_arc)
    return kstep, kstep_arc

def runHRO(Map, Bmap, vecMask, location, FWHM=0, kstep=0, Qmap=None, Umap=None, hdu=None, Bproj=False):
    #print(hdu.data.shape, "****")
    print(location)

    if hdu is None:
        hdu = vecMask

    if Qmap is None:
        ref = Bmap
    else:
        ref = Map
    kstep_pix, kstep_arc = smoothing(FWHM, ref, kstep=kstep, frac=3, thres=3)
    
    hro = h.HRO(Map, Qmap=Qmap, Umap=Umap, Bmap=Bmap, hdu=hdu, vecMask=vecMask, msk=True, kstep=kstep_pix, Bproj=Bproj)
    fldr = '/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/FWHM_3/'
    prefix= fldr + location
    if os.path.exists(prefix)==False:
        os.mkdir(prefix)

    # prefix=prefix+'kstep{0}/'.format(np.round(kstep_pix, 1))
    # if os.path.exists(prefix)==False:
    #    os.mkdir(prefix)

    if os.path.exists(prefix+'Plots/')==False:
        os.mkdir(prefix+'Plots/')
    r.makePlots(hro, prefix+'Plots/', isSim=False, label='', step=20, scale=15)

    np.savez(prefix+'HRO_results.npz', MapProj=hro.Map_proj.data, vecMask=hro.vecMask.data, Bfield=hro.Bfield, Efield=hro.Efield, grad=hro.grad, contour=hro.contour, 
            grad_mag=hro.mag, phi=hro.phi, hist=hro.hist, histbins=hro.histbins, Zx_unCorr=hro.Zx, #Zx_corr=PRS_corr, 
            kstep_pix=kstep_pix, kstep_arcsec=kstep_arc)
    return hro, prefix

def runWhiteNoise_HRO(Map, Bmap, kstep, vecMask, hdu, location, Qmap=None, Umap=None, Bproj=False):
    prefix='/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/FWHM_3/' + location
    mean = np.nanmean(Map.data)
    std = np.nanstd(Map.data)
    noise = np.random.normal(mean, std, size=Map.data.shape)
    noise_Map = Map.copy()
    noise_Map.data = noise
    #noise_Map.writeto(prefix + 'Whitenoise.fits')
    hro = h.HRO(noise_Map, Qmap=Qmap, Umap=Umap, Bmap=Bmap, hdu=hdu, vecMask=vecMask, msk=True, kstep=kstep, Bproj=Bproj) 
    if os.path.exists(prefix)==False:
        os.mkdir(prefix)
    #r.makePlots(hro, prefix, isSim=False, label='')
    return hro

def loopNoise(Map, Bmap, vecMask, location, counter, hdu, Qmap=None, Umap=None, plot=False):
    PRS = []
    fldr = '/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/FWHM_3/'
    #prefix = fldr + 'WhiteNoise/' + location
    prefix = fldr + location
    if os.path.exists(prefix)==False:
        os.mkdir(prefix)
    output = fldr + location
    HRO = np.load(output + 'HRO_results.npz')
    kstep = HRO['kstep_pix']
    print(kstep)
    for i in range(counter):
        print(i)
        hro = runWhiteNoise_HRO(Map, Bmap, kstep, vecMask, hdu, location, Qmap=Qmap, Umap=Umap, Bproj=False)
        PRS.append(hro.Zx)
        if plot:
            r.makePlots(hro, prefix, isSim=False, label='', step=20, scale=15)
    np.save(prefix + 'PRS_{0}.npy'.format(counter), PRS)
    return PRS
    
    # Zx_mean = np.nanmean(PRS)
    # Zx_std = np.nanstd(PRS)
    # run = '/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/BandC/' + location
    # HRO = np.load(run + 'HRO_results.npz')

    # np.savez(run+'HRO_results.npz', MapProj=HRO['MapProj'], vecMask=HRO['vecMask'], Bfield=HRO['Bfield'], Efield=HRO['Efield'], 
    #          grad=HRO['grad'], contour=HRO['contour'], grad_mag=HRO['grad_mag'], phi=HRO['phi'], hist=HRO['hist'], 
    #          histbins=HRO['histbins'], Zx=HRO[''], Zx_p=PRS_corr, 
    #     kstep_pix=hro.kstep_pix, kstep_arcsec=hro.kstep_arcsec)


def runHRO_QU(Map, Qmap, Umap, kstep, vecMask, location):
    #hro = h.HRO(Map, Qmap=Qmap, Umap=Umap, Bmap=None, hdu=ALMA_mask, vecMask=vecMask, msk=True, kstep=kstep) 
    hro = h.HRO(Map, Qmap=Qmap, Umap=Umap, Bmap=None, hdu=vecMask, vecMask=vecMask, msk=True, kstep=kstep) 
    prefix='/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/' + location
    if os.path.exists(prefix)==False:
        os.mkdir(prefix)
    prefix=prefix+'kstep{0}/'.format(kstep)
    if os.path.exists(prefix)==False:
        os.mkdir(prefix)
    r.makePlots(hro, prefix, isSim=False, label='')

def CompareVectors(Map, Bmap, Qmap1, Umap1, Qmap2, Umap2, vecMask, location):
    hro = h.HRO(Map=Map, Bmap=Bmap, Qmap=Qmap1, Umap=Umap1, Qmap2=Qmap2, Umap2=Umap2, hdu=vecMask, vecMask=vecMask, msk=True, compare=True) 
    prefix='/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/CompareBfield/' + location
    if os.path.exists(prefix)==False:
        os.mkdir(prefix)
    prefix=prefix
    if os.path.exists(prefix)==False:
        os.mkdir(prefix)
    r.makePlots_Compare(hro, prefix, isSim=False, label='')

# def MaskALMA(Map, region, thres):        

#     #for region in regions:
#     Mask = Map.copy()
#     size=15
#     std = np.nanstd(Map.data[region[0]-size:region[0]+size, region[1]-size:region[1]+size])
#     mean = np.nanmean(Map.data[region[0]-size:region[0]+size, region[1]-size:region[1]+size])
#     #Std.append(s)
#     #Mean.append(m)
#     #std = np.nanmean(np.asarray(Std))
#     #mean = np.nanmean(np.asarray(Mean))
#     Mask.data = np.ones(Map.shape)
#     Mask.data[((Map.data - mean) / std) < thres]=np.nan
#     Mask.data[np.isnan(Map.data)]=np.nan
#     Mask.writeto('Output_plots/Paper/ALMA/ALMAcont_mask_3sig.fits')
        
#     plt.close('all')
#     fig = plt.figure(figsize=[5,4], dpi=300)
#     fxx = aplpy.FITSFigure(Map, figure=fig, slices=[0])
#     fxx.show_colorscale(interpolation='nearest', cmap='Spectral_r')
#     fxx.show_rectangles(region[0], region[1], size, size, coords_frame='pixel')
#     fxx.add_colorbar()
#     plt.savefig('Output_plots/Paper/ALMA/Region.png') 
    
#     plt.close('all')
#     fig = plt.figure(figsize=[5,4], dpi=300)
#     fxx = aplpy.FITSFigure(Mask, figure=fig, slices=[0])
#     fxx.show_colorscale(interpolation='nearest', cmap='binary')
#     #fxx.show_contour(Hersch, levels=[20,50,80], colors='white')
#     fxx.add_colorbar()
#     plt.savefig('Output_plots/Paper/ALMA/Mask_values.png') 

#     # plt.close('all')
#     # fig = plt.figure(figsize=[5,4], dpi=300)
#     # fxx = aplpy.FITSFigure(hro.Map, figure=fig, slices=[0])
#     # fxx.show_colorscale(interpolation='nearest', cmap='Spectral_r')
#     # #fxx.show_contour(Hersch, levels=[20,50,80], colors='white')
#     # fxx.add_colorbar()
#     # plt.savefig(prefix+'Map_proj.png') 

def runCube_Maskregions(data, Bmap, vecMask, kstep, location, Qmap=None, Umap=None, regions=None, start=0, end=15, mask=False, thres=3, proj_Map=None):
    incr = (int(end) - int(start))*2 + 1
    speed = np.round(np.linspace(start,end,incr), 1)
    cube = SpectralCube.read(data)

    PRS = np.zeros((3,len(speed)))
    PRS[0] = speed
    hists = np.zeros((len(speed),20))
    Phi = []
    j = 0
    
    for i in speed:
        cube_speed = cube.with_spectral_unit(unit=u.km/u.s, velocity_convention='radio')
        slab = cube_speed.spectral_slab((i-0.5)*u.km/u.s, (i+0.5)*u.km/u.s)
        mom0 = slab.moment0().hdu

        if (proj_Map is None)==False:
            mom0 = h.projectMap(mom0, proj_Map)

        Mask = mom0.copy()
        Mask.data = np.ones(mom0.data.shape)
        
        if mask == True:
            Std = []; Mean = []
            for region in regions:
                s = np.nanstd(mom0.data[region[0]-5:region[0]+5, region[1]-5:region[1]+5])
                m = np.nanmean(mom0.data[region[0]-5:region[0]+5, region[1]-5:region[1]+5])
                Std.append(s)
                Mean.append(m)
            std = np.nanmean(np.asarray(Std))
            mean = np.nanmean(np.asarray(Mean))
            
            Mask.data[((mom0.data - mean) / std) < thres]=0  
            Mask.data[np.isnan(mom0.data)]=np.nan

        fldr = location + '{0}/'.format(i)
        hro, loc = runHRO(mom0, Bmap, kstep=kstep, Qmap=Qmap, Umap=Umap, vecMask=vecMask, hdu=Mask, location=fldr)
        PRS[1,j] = hro.Zx
        PRS[2,j] = hro.meanPhi
        hists[j] = hro.hist
        Phi.append(hro.phi)
        j = j+1

        plt.close('all')
        fig = plt.figure(figsize=[12,8])
        fxx = aplpy.FITSFigure(h.projectMap(mom0, Bmap), figure=fig)
        fxx.show_colorscale(vmax=1, vmin=0)
        fxx.add_colorbar()
        fxx.show_vectors(vecMask, Bmap, step=6, scale =4, units='degrees', color = 'White', linewidth=3)
        fxx.add_label(0.85, 0.85, text='{0} km/s'.format(i), relative=True, color='white', size='xx-large', weight='demibold')
        plt.savefig(loc + 'DataMask.png')

    out_fldr = '/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/' + location 
    np.save(out_fldr + 'PRS_speed.npy', PRS)
    np.save(out_fldr + 'hist_speed.npy', hists)
    np.save(out_fldr + 'total_phi.npy', Phi)
    r.plotPRS_speed(out_fldr)
        
    plt.close('all')
    fig = plt.figure(figsize=[12,8])
    fxx = aplpy.FITSFigure(mom0, figure=fig)
    fxx.show_colorscale(interpolation='nearest')
    fxx.add_colorbar()
    fxx.add_label(0.85, 0.85, text='{0} km/s'.format(i), relative=True, color='black', size='xx-large', weight='demibold')
    for region in regions:
        fxx.show_rectangles(region[0], region[1], height=10, width=10, coords_frame='pixel', edgecolor='cyan', linewidth=1.8)
    plt.savefig(out_fldr + 'mom0_regions.png') 

    # plt.close('all')
    # fig = plt.figure(figsize=[12,8])
    # fxx = aplpy.FITSFigure(mom0_HAWC, figure=fig)
    # fxx.show_colorscale(interpolation='nearest')
    # fxx.add_colorbar()
    # fxx.add_label(0.85, 0.85, text='{0} km/s'.format(i), relative=True, color='black', size='xx-large', weight='demibold')
    # fxx.show_vectors(vecMask, HAWC[11], step=6, scale =4, units='degrees', color = 'White', linewidth=3)
    # fxx.show_vectors(vecMask, HAWC[11], step=6, scale =4, units='degrees', color = 'Black', linewidth=1)
    # fxx.show_contour(Hersch, levels=[15,20,50,80], colors='black')
    # fxx.show_contour(Mask, levels=[0.9], colors='red')
    # for region in regions:
    #     fxx.show_rectangles(region[0], region[1], height=20, width=20, coords_frame='pixel', edgecolor='cyan', linewidth=1.8)
    # plt.savefig(folder + fldr + 'mom0_regions.png') 

def CompareVectors(Map, Bmap, Qmap1, Umap1, Qmap2, Umap2, vecMask, location):
    hro = h.HRO(Map=Map, Bmap=Bmap, Qmap=Qmap1, Umap=Umap1, Qmap2=Qmap2, Umap2=Umap2, hdu=vecMask, vecMask=vecMask, msk=True, compare=True) 
    prefix='/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/FWHM_3/CompareBfield/' + location
    if os.path.exists(prefix)==False:
        os.mkdir(prefix)
    prefix=prefix
    if os.path.exists(prefix)==False:
        os.mkdir(prefix)
    np.savez(prefix+'HRO_results.npz', phi=hro.phi, hist=hro.hist, histbins=hro.histbins, Zx_unCorr=hro.Zx)
    #r.makePlots_Compare(hro, prefix, isSim=False, label='')

def MaskALMA(Map, regions, thres):        

    Std = []; Mean = []
    for region in regions:
        s = np.nanstd(Map.data[region[0]-10:region[0]+10, region[1]-10:region[1]+10])
        print(s)
        m = np.nanmean(Map.data[region[0]-10:region[0]+10, region[1]-10:region[1]+10])
        Std.append(s)
        Mean.append(m)
    
    std = np.nanmean(np.asarray(Std))
    mean = np.nanmean(np.asarray(Mean))
    
    Mask = Map.copy()
    Mask.data = np.ones(Map.shape)
    Mask.data[((Map.data - mean) / std) < thres]=np.nan
    Mask.data[np.isnan(Map.data)]=np.nan
    Mask.writeto('Output_plots/Paper/BandC/ALMA_12m_cont/ALMAcont_mask_3sig.fits', overwrite=True)
        
    plt.close('all')
    fig = plt.figure(figsize=[5,4], dpi=300)
    fxx = aplpy.FITSFigure(Map, figure=fig, slices=[0])
    fxx.show_colorscale(interpolation='nearest', cmap='Spectral_r')
    for region in regions:
        fxx.show_rectangles(region[0], region[1], height=20, width=20, coords_frame='pixel', edgecolor='black', linewidth=1)
    fxx.add_colorbar()
    plt.savefig('Output_plots/Paper/BandC/ALMA_12m_cont/Region.png') 
    
    plt.close('all')
    fig = plt.figure(figsize=[5,4], dpi=300)
    fxx = aplpy.FITSFigure(Mask, figure=fig, slices=[0])
    fxx.show_colorscale(interpolation='nearest', cmap='binary')
    #fxx.show_contour(Hersch, levels=[20,50,80], colors='white')
    fxx.add_colorbar()
    plt.savefig('Output_plots/Paper/BandC/ALMA_12m_cont/Mask_values.png') 

    # plt.close('all')
    # fig = plt.figure(figsize=[5,4], dpi=300)
    # fxx = aplpy.FITSFigure(hro.Map, figure=fig, slices=[0])
    # fxx.show_colorscale(interpolation='nearest', cmap='Spectral_r')
    # #fxx.show_contour(Hersch, levels=[20,50,80], colors='white')
    # fxx.add_colorbar()
    # plt.savefig(prefix+'Map_proj.png') 

def runCube(data, Bmap, vecMask, location, Qmap=None, Umap=None, regions=None, start=0, end=15, mask=False, thres=3, proj_Map=None, Noise=False, counter=10):
    incr = (int(end) - int(start))*2 + 1
    print(incr)
    speed = np.round(np.linspace(start,end,incr), 1)
    print(speed)
    cube = SpectralCube.read(data)
    cube_speed = cube.with_spectral_unit(unit=u.km/u.s, velocity_convention='radio')

    PRS = np.zeros((3,len(speed)))
    PRS[0] = speed
    PRS_corr = np.zeros((4,len(speed)))
    PRS_corr[0] = speed
    hists = np.zeros((len(speed),20))
    Phi = []
    j = 0


    folder = '/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/FWHM_3/'
    path = Path(folder + location)
    hro_path = str(path.parent.absolute())
    single_map = np.load(hro_path + '/HRO_results.npz')
    kstep = single_map['kstep_pix']
    #WN_folder = '/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/FWHM_3/Whitenoise/'

    if mask==True:
        noise = cube_speed.spectral_slab(speed[0]*u.km/u.s, speed[1]*u.km/u.s)
        mean = np.nanmean(noise)
        std = np.nanstd(noise)
        location = location + 'Masked_{0}/'.format(thres)
        out_fldr = folder + location 

        if os.path.exists(out_fldr)==False:
            os.mkdir(out_fldr)
        

    location = location + 'kstep{0}/'.format(np.round(kstep, 1))
    out_fldr = folder + location
    if os.path.exists(out_fldr)==False:
        os.mkdir(out_fldr)        
    
    for i in speed:
        
        slab = cube_speed.spectral_slab((i-0.5)*u.km/u.s, (i+0.5)*u.km/u.s)
        mom0 = slab.moment0().hdu

        if (proj_Map is None)==False:
            mom0 = h.projectMap(mom0, proj_Map)

        Mask = mom0.copy()
        Mask.data = np.ones(mom0.data.shape)
        
        if mask == True:
            Mask.data[((mom0.data - mean) / std) < thres]=0  
            Mask.data[np.isnan(mom0.data)]=np.nan

        fldr = location + '{0}/'.format(i)
        if Noise==False:
            hro, loc = runHRO(mom0, Bmap, kstep=kstep, Qmap=Qmap, Umap=Umap, vecMask=vecMask, hdu=Mask, location=fldr)
            Mask.writeto(loc + 'Mask.fits', overwrite=True)
            PRS[1,j] = hro.Zx
            PRS[2,j] = hro.meanPhi
            hists[j] = hro.hist
            Phi.append(hro.phi)

            plt.close('all')
            fig = plt.figure(figsize=[12,8])
            if proj_Map is None:
                fxx = aplpy.FITSFigure(h.projectMap(Mask, Bmap), figure=fig)
            else:
                fxx = aplpy.FITSFigure(Mask, figure=fig)
            fxx.show_colorscale(vmax=1, vmin=0)
            fxx.add_colorbar()
            fxx.show_vectors(vecMask, h.projectMap(Bmap, vecMask), step=6, scale =4, units='degrees', color = 'White', linewidth=2)
            fxx.show_vectors(vecMask, h.projectMap(Bmap, vecMask), step=6, scale =4, units='degrees', color = 'Black', linewidth=1)
            fxx.add_label(0.85, 0.85, text='{0} km/s'.format(i), relative=True, color='white', size='xx-large', weight='demibold')
            plt.savefig(loc + 'DataMask.png')
        else: 
            PRS_vel = loopNoise(mom0, Bmap, vecMask=vecMask, hdu=Mask, location=fldr, counter=counter)
            Zx_std = np.nanstd(PRS_vel)
            Zx_mean = np.nanmean(PRS_vel)
            Zx_summary = np.asarray([Zx_mean, Zx_std])
            np.save(folder + fldr+ 'PRS_{0}_stats.npy'.format(counter), Zx_summary)
            np.save(folder + fldr + 'PRS_{0}_corr.npy'.format(counter), Zx_std)
            hro = np.load(folder + fldr + 'HRO_results.npz')
            PRS_corr[1,j] = hro['Zx_unCorr']
            PRS_corr[2,j] = hro['Zx_unCorr'] / Zx_std
            PRS_corr[3,j] = Zx_std
        j = j+1


    if Noise==False:
        np.save(out_fldr + 'PRS_speed.npy', PRS)
        np.save(out_fldr + 'hist_speed.npy', hists)
        np.save(out_fldr + 'total_phi.npy', Phi)
        r.plotPRS_speed(out_fldr)
        if mask==False:
            sigma = single_map['Zx_std']
            print(sigma)
            plt.figure(figsize=[12,8], dpi=200)
            plt.plot(speed, PRS[1]/sigma)
            plt.ylabel('Projected Rayleigh Statistic')
            plt.xlabel('Speed')
            plt.title('Statistic vs Speed')
            plt.savefig(out_fldr + 'PRScorr_velocity.png')

    else:
        print()
        np.save(out_fldr + 'PRScorr{0}_speed.npy'.format(counter), PRS_corr)
        plt.figure(figsize=[12,8], dpi=200)
        plt.plot(speed, PRS_corr[2])
        plt.ylabel('Corrected Projected Rayleigh Statistic')
        plt.xlabel('Speed')
        plt.title('Statistic vs Speed')
        plt.savefig(out_fldr + 'PRS_WNcorr{0}_velocity.png'.format(counter))
            
    
def PRScorr_single(location):
    prefix = '/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/FWHM_3/' + location
    Zx_WN = np.load(prefix + 'PRS_1000.npy')
    Zx_mean = np.nanmean(Zx_WN)
    Zx_std = np.nanstd(Zx_WN)
    Zx_summary = np.asarray([Zx_mean, Zx_std])
    np.save(prefix + 'PRS_1000_stats.npy', Zx_summary)
    np.save(prefix + 'PRS_1000_corr.npy', Zx_std)
    print('*** {0}: , {1}'.format(location, np.round(Zx_std, 2)))
    hro = np.load(prefix+'HRO_results.npz')
    Zx_uncorr = hro['Zx_unCorr']
    Zx_corr = Zx_uncorr / Zx_std
    np.savez(prefix+'HRO_results.npz', MapProj=hro['MapProj'], vecMask=hro['vecMask'], Bfield=hro['Bfield'], Efield=hro['Efield'], 
                 grad=hro['grad'], contour=hro['contour'], grad_mag=hro['grad_mag'], phi=hro['phi'], hist=hro['hist'], histbins=hro['histbins'], 
                 Zx_unCorr=hro['Zx_unCorr'], Zx_corr=Zx_corr, Zx_mean=Zx_mean, Zx_std=Zx_std, kstep_pix=hro['kstep_pix'], kstep_arcsec=hro['kstep_arcsec'])
    return Zx_mean, Zx_std

def PRScorr_all(sub_fldr='BandC/'):
    folder = '/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/FWHM_3/' + sub_fldr
    runs = os.listdir(folder)
    if '.DS_Store' in runs:
        runs.remove('.DS_Store')
    for run in runs:
        print(run)
        Zx_mean, Zx_std = PRScorr_single(location=sub_fldr+run+'/')
        hro = np.load(folder+run+'/HRO_results.npz')
        Zx_uncorr = hro['Zx_unCorr']
        Zx_corr = Zx_uncorr / Zx_std
        np.savez(folder+run+'/HRO_results.npz', MapProj=hro['MapProj'], vecMask=hro['vecMask'], Bfield=hro['Bfield'], Efield=hro['Efield'], 
                 grad=hro['grad'], contour=hro['contour'], grad_mag=hro['grad_mag'], phi=hro['phi'], hist=hro['hist'], histbins=hro['histbins'], 
                 Zx_unCorr=hro['Zx_unCorr'], Zx_corr=Zx_corr, Zx_mean=Zx_mean, Zx_std=Zx_std, kstep_pix=hro['kstep_pix'], kstep_arcsec=hro['kstep_arcsec'])

#def PRScorr_velocity(location, start=0, end=15, counter=10):


def Mask_TotalI(data2D, data_cube, Bmap, vecMask, filename, location, thres=3, msk_speed=[-50,30]):
    cube = SpectralCube.read(data_cube)
    cube_speed = cube.with_spectral_unit(unit=u.km/u.s, velocity_convention='radio')
    
    noise = cube_speed.spectral_slab(msk_speed[0]*u.km/u.s, msk_speed[1]*u.km/u.s)
    noise = (h.projectMap(noise.moment0().hdu, Bmap)).data
    mean = np.nanmean(noise)
    std = np.nanstd(noise)
    
    data2D = h.projectMap(data2D, Bmap)
    Mask = data2D.copy()
    Mask.data = np.ones(data2D.shape)   
    Mask.data[((data2D.data - mean) / std) < thres]=np.nan  #scale mean by chanels
    Mask.data[np.isnan(data2D.data)]=np.nan
    print(Mask.shape, "***")

    Mask.writeto(folder + filename + '_thres{0}.fits'.format(thres), overwrite=True)

    plt.close('all')
    fig = plt.figure(figsize=[12,8])
    fxx = aplpy.FITSFigure(h.projectMap(Mask, Bmap), figure=fig)
    fxx.show_colorscale(vmax=1, vmin=0, cmap='Spectral_r')
    fxx.add_colorbar()
    fxx.show_vectors(vecMask, Bmap, step=6, scale =4, units='degrees', color = 'White', linewidth=2)
    fxx.show_vectors(vecMask, Bmap, step=6, scale =4, units='degrees', color = 'Black', linewidth=1)
    prefix='/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/'
    plt.savefig(prefix + location + 'DataMask_thres{0}.png'.format(thres))

PRScorr_single(location='BandE/Hersh_70/native/')
PRScorr_single(location='BandE/Hersh_160/native/')
PRScorr_single(location='BandE/BandC_I/native/')

#runCube(CO12_cube, HAWC[11], vecMask_C, location='BandC/12CO/Cube/', start=0, end=15, mask=False, thres=3)
# runCube(CO13_cube, HAWC[11], vecMask_C, location='BandC/13CO/Cube/', start=0, end=15, mask=True, thres=2)
# runCube(CII_cube, HAWC[11], vecMask_C, location='BandC/CII/Cube/', start=-5, end=15, mask=True, thres=2)
# runCube(OI_cube, HAWC[11], vecMask_C, location='BandC/OI/Cube/', start=0, end=15, mask=True, thres=2)
# runCube(HNC_cube, HAWC[11], vecMask_C, location='BandC/Mopra_HNC/Cube/', start=0, end=15, mask=True, thres=2)
# runCube(N2H_cube, HAWC[11], vecMask_C, location='BandC/Mopra_N2H/Cube/', start=-6, end=15, mask=True, thres=2)
# runCube(C18O_cube, HAWC[11], vecMask_C, location='BandC/Mopra_C18O/Cube/', start=0, end=15, mask=True, thres=2)

# runCube(CO12_cube, HAWC[11], vecMask_C, location='BandC/12CO/Cube/', start=0, end=15, mask=True, thres=3, Noise=True, counter=1000)
# runCube(CO13_cube, HAWC[11], vecMask_C, location='BandC/13CO/Cube/', start=0, end=15, mask=True, thres=3, Noise=True, counter=1000)
# runCube(CII_cube, HAWC[11], vecMask_C, location='BandC/CII/Cube/', start=-5, end=15, mask=True, thres=3, Noise=True, counter=1000)
# runCube(OI_cube, HAWC[11], vecMask_C, location='BandC/OI/Cube/', start=0, end=15, mask=True, thres=3, Noise=True, counter=1000)
# runCube(HNC_cube, HAWC[11], vecMask_C, location='BandC/Mopra_HNC/Cube/', start=0, end=15, mask=True, thres=3, Noise=True, counter=1000)
# runCube(N2H_cube, HAWC[11], vecMask_C, location='BandC/Mopra_N2H/Cube/', start=-6, end=15, mask=True, thres=3, Noise=True, counter=1000)
#runCube(C18O_cube, HAWC[11], vecMask_C, location='BandC/Mopra_C18O/Cube/', start=0, end=15, mask=True, thres=3, Noise=True, counter=1000)

#runCube(data, Bmap, vecMask, kstep, location, Qmap=None, Umap=None, regions=None, start=0, end=15, mask=False, thres=3, proj_Map=None, speed=[-50,30], Noise=False, counter=10)

# Mask_TotalI(Mop_HNC_more, HNC_cube, Bmap=HAWE[11], vecMask=vecMask_E, filename='Mopra/HNC_BandE_TotalI_Mask', location='BandE/Mopra_HNC/', thres=4, msk_speed=[20,40])
# Mask_TotalI(Mop_C18O, C18O_cube, Bmap=HAWE[11], vecMask=vecMask_E, filename='Mopra/C18O_BandE_TotalI_Mask', location='BandE/Mopra_C18O/', thres=4, msk_speed=[20,40])
# Mask_TotalI(Mop_N2H, N2H_cube, Bmap=HAWE[11], vecMask=vecMask_E, filename='Mopra/N2H_BandE_TotalI_Mask', location='BandE/Mopra_N2H/', thres=4, msk_speed=[20,40])

# PRScorr(location='BandE/Ncol_Av/')
# PRScorr(location='BandE/12CO/')
# PRScorr(location='BandE/13CO/')
# PRScorr(location='BandE/CII/')
# PRScorr(location='BandE/BandE_I/')
# PRScorr(location='BandE/BandC_I/')
# PRScorr(location='BandE/Ncol_Av_Low/')
# PRScorr(location='BandE/Temp_Low/')
# PRScorr(location='BandE/Temp_High/')
# PRScorr(location='BandE/Hersh_70/')
# PRScorr(location='BandE/Hersh_160/')
# PRScorr(location='BandE/Hersh_250/')
# PRScorr(location='BandE/Hersh_350/')
# PRScorr(location='BandE/Hersh_500/')
# PRScorr(location='BandE/Mopra_HNC/')
# PRScorr(location='BandE/Mopra_C18O/')
# PRScorr(location='BandE/Mopra_N2H/')
# PRScorr(location='BandE/OI/')
# PRScorr(location='BandE/Halpha/')

# runCube(CO12_cube, BLAST_Bmap, BandC_boundary, Qmap=BLAST_Q, Umap=BLAST_U, kstep=3, location='BLASTPol/12CO/Cube/')
# runCube(CO13_cube, BLAST_Bmap, BandC_boundary, Qmap=BLAST_Q, Umap=BLAST_U, kstep=3, location='BLASTPol/13CO/Cube/')
# runCube(CII_cube, BLAST_Bmap, BandC_boundary, Qmap=BLAST_Q, Umap=BLAST_U, kstep=3, location='BLASTPol/CII/Cube/')
# runCube(OI_cube, BLAST_Bmap, BandC_boundary, Qmap=BLAST_Q, Umap=BLAST_U, kstep=3, location='BLASTPol/OI/Cube/')
# runCube(HNC_cube, BLAST_Bmap, BandC_boundary, Qmap=BLAST_Q, Umap=BLAST_U, kstep=3, location='BLASTPol/Mopra_HNC/Cube/', proj_Map=BandC_boundary)
# runCube(C18O_cube, BLAST_Bmap, BandC_boundary, Qmap=BLAST_Q, Umap=BLAST_U, kstep=3, location='BLASTPol/Mopra_C18O/Cube/', proj_Map=BandC_boundary)
# runCube(N2H_cube, BLAST_Bmap, BandC_boundary, Qmap=BLAST_Q, Umap=BLAST_U, kstep=3, location='BLASTPol/Mopra_N2H/Cube/', proj_Map=BandC_boundary)

# runCube(CO12_cube, HAWC[11], vecMask_C, kstep=4, location='BandC/12CO/Cube/', mask=True, speed=[-50,-30])
# runCube(CO13_cube, HAWC[11], vecMask_C, kstep=4, location='BandC/13CO/Cube/', mask=True, speed=[-50,-30])
# runCube(CII_cube, HAWC[11], vecMask_C, kstep=4, location='BandC/CII/Cube/', mask=True, speed=[-40,-30])
# runCube(OI_cube, HAWC[11], vecMask_C, kstep=4, location='BandC/OI/Cube/', mask=True, speed=[-20,5])
# runCube(HNC_cube, HAWC[11], vecMask_C, kstep=4, location='BandC/Mopra_HNC/Cube/', mask=True, speed=[20,40])
# runCube(C18O_cube, HAWC[11], vecMask_C, kstep=4, location='BandC/Mopra_C18O/Cube/', mask=True, speed=[20,40])
# runCube(N2H_cube, HAWC[11], vecMask_C, kstep=4, location='BandC/Mopra_N2H/Cube/', mask=True, speed=[20,40])

# runCube(CO12_cube, HAWE[11], vecMask_E, kstep=4, location='BandE/12CO/Cube/', mask=True, speed=[-50,-30])
# runCube(CO13_cube, HAWE[11], vecMask_E, kstep=4, location='BandE/13CO/Cube/', mask=True, speed=[-50,-30])
# runCube(CII_cube, HAWE[11], vecMask_E, kstep=4, location='BandE/CII/Cube/',  mask=True, speed=[-40,-30])
# runCube(OI_cube, HAWE[11], vecMask_E, kstep=4, location='BandE/OI/Cube/', mask=True, speed=[-20,5])
# runCube(HNC_cube, HAWE[11], vecMask_E, kstep=4, location='BandE/Mopra_HNC/Cube/', mask=True, speed=[20,40])
# runCube(C18O_cube, HAWE[11], vecMask_E, kstep=4, location='BandE/Mopra_C18O/Cube/', mask=True, speed=[20,40])
# runCube(N2H_cube, HAWE[11], vecMask_E, kstep=4, location='BandE/Mopra_N2H/Cube/', mask=True, speed=[20,40])

# runCube(CO12_cube, BLAST_Bmap, BandC_boundary, Qmap=BLAST_Q, Umap=BLAST_U, kstep=3, location='BLASTPol/12CO/Cube/', mask=True, proj_Map=BandC_boundary, speed=[-50,-30])
# runCube(CO13_cube, BLAST_Bmap, BandC_boundary, Qmap=BLAST_Q, Umap=BLAST_U, kstep=3, location='BLASTPol/13CO/Cube/', mask=True, proj_Map=BandC_boundary, speed=[-50,-30])
# runCube(CII_cube, BLAST_Bmap, BandC_boundary, Qmap=BLAST_Q, Umap=BLAST_U, kstep=3, location='BLASTPol/CII/Cube/', mask=True, proj_Map=BandC_boundary, speed=[-40,-30])
# runCube(OI_cube, BLAST_Bmap, BandC_boundary, Qmap=BLAST_Q, Umap=BLAST_U, kstep=3, location='BLASTPol/OI/Cube/', mask=True, proj_Map=BandC_boundary, speed=[-20,5])
# runCube(HNC_cube, BLAST_Bmap, BandC_boundary, Qmap=BLAST_Q, Umap=BLAST_U, kstep=3, location='BLASTPol/Mopra_HNC/Cube/', mask=True, proj_Map=BandC_boundary, speed=[20,40])
# runCube(C18O_cube, BLAST_Bmap, BandC_boundary, Qmap=BLAST_Q, Umap=BLAST_U, kstep=3, location='BLASTPol/Mopra_C18O/Cube/', mask=True, proj_Map=BandC_boundary, speed=[20,40])
# runCube(N2H_cube, BLAST_Bmap, BandC_boundary, Qmap=BLAST_Q, Umap=BLAST_U, kstep=3, location='BLASTPol/Mopra_N2H/Cube/', mask=True, proj_Map=BandC_boundary, speed=[20,40])

# runCube(CO12_cube, BLAST_Bmap, BandC_boundary, Qmap=BLAST_Q, Umap=BLAST_U, kstep=4, location='BLASTPol/12CO/Cube/', mask=True, proj_Map=BandC_boundary, speed=[-50,-30])
# runCube(CO13_cube, BLAST_Bmap, BandC_boundary, Qmap=BLAST_Q, Umap=BLAST_U, kstep=4, location='BLASTPol/13CO/Cube/', mask=True, proj_Map=BandC_boundary, speed=[-50,-30])
# runCube(CII_cube, BLAST_Bmap, BandC_boundary, Qmap=BLAST_Q, Umap=BLAST_U, kstep=4, location='BLASTPol/CII/Cube/', mask=True, proj_Map=BandC_boundary, speed=[-40,-30])
# runCube(OI_cube, BLAST_Bmap, BandC_boundary, Qmap=BLAST_Q, Umap=BLAST_U, kstep=4, location='BLASTPol/OI/Cube/', mask=True, proj_Map=BandC_boundary, speed=[-20,5])
# runCube(HNC_cube, BLAST_Bmap, BandC_boundary, Qmap=BLAST_Q, Umap=BLAST_U, kstep=4, location='BLASTPol/Mopra_HNC/Cube/', mask=True, proj_Map=BandC_boundary, speed=[20,40])
# runCube(C18O_cube, BLAST_Bmap, BandC_boundary, Qmap=BLAST_Q, Umap=BLAST_U, kstep=4, location='BLASTPol/Mopra_C18O/Cube/', mask=True, proj_Map=BandC_boundary, speed=[20,40])
# runCube(N2H_cube, BLAST_Bmap, BandC_boundary, Qmap=BLAST_Q, Umap=BLAST_U, kstep=4, location='BLASTPol/Mopra_N2H/Cube/', mask=True, proj_Map=BandC_boundary, speed=[20,40])

# runCube(CO12_cube, HAWE[11], vecMask_E, kstep=2, location='BandE/12CO/Cube/kstep2/', project=True)
# runCube(CO13_cube, HAWE[11], vecMask_E, kstep=2, location='BandE/13CO/Cube/kstep2/', project=True)
# runCube(CII_cube, HAWE[11], vecMask_E, kstep=2, location='BandE/CII/Cube/kstep2/', project=True)
# runCube(OI_cube, HAWE[11], vecMask_E, kstep=2, location='BandE/OI/Cube/kstep2/', project=True)
# runCube(HNC_cube, HAWE[11], vecMask_E, kstep=2, location='BandE/Mopra_HNC/Cube/kstep2/', project=True)
# runCube(C18O_cube, HAWE[11], vecMask_E, kstep=2, location='BandE/Mopra_C18O/Cube/kstep2/', project=True)
# runCube(N2H_cube, HAWE[11], vecMask_E, kstep=2, location='BandE/Mopra_N2H/Cube/kstep2/', project=True)

#MaskALMA(ALMA_ACA, [150, 60], thres=3)
#MaskALMA(ALMA_12m, regions=[[100, 170], [260,450]], thres=3) 

#CompareVectors(HAWC[0], HAWC[11], HAWC_Q, HAWC_U, BLAST_Q, BLAST_U, vecMask_C, 'BandC_BLAST/')
#CompareVectors(HAWE[0], HAWE[11], HAWE_Q, HAWE_U, BLAST_Q, BLAST_U, vecMask_E, 'BandE_BLAST/')
#CompareVectors(HAWC[0], HAWC[11], HAWC_Q, HAWC_U, HAWE_Q, HAWE_U, vecMask_C, 'BandC_BandE/')

# kstep=2
# runHRO_QU( H70_B, BLAST_Q, BLAST_U, kstep=kstep, vecMask=BandC_boundary, location='BLASTPol/Hersch_70/')
# runHRO_QU(H160_B, BLAST_Q, BLAST_U, kstep=kstep, vecMask=BandC_boundary, location='BLASTPol/Hersch_160/')
# runHRO(H250_B, Bmap=BLAST_Bmap, Qmap=BLAST_Q, Umap=BLAST_U, kstep=3, vecMask=BandC_boundary, location='BLASTPol/Hersch_250/')
# runHRO(H350_B, Bmap=BLAST_Bmap, Qmap=BLAST_Q, Umap=BLAST_U, kstep=3, vecMask=BandC_boundary, location='BLASTPol/Hersch_350/')
# runHRO(H500_B, Bmap=BLAST_Bmap, Qmap=BLAST_Q, Umap=BLAST_U, kstep=3, vecMask=BandC_boundary, location='BLASTPol/Hersch_500/')
# runHRO_QU(Ncol_B, BLAST_Q, BLAST_U, kstep=kstep, vecMask=BandC_boundary, location='BLASTPol/Hersch_ColDen/')
# runHRO_QU(CO12_B, BLAST_Q, BLAST_U, kstep=kstep, vecMask=BandC_boundary, location='BLASTPol/12CO/')
# runHRO_QU(CO13_B, BLAST_Q, BLAST_U, kstep=kstep, vecMask=BandC_boundary, location='BLASTPol/13CO/')
# runHRO_QU(CII_B, BLAST_Q, BLAST_U, kstep=kstep, vecMask=BandC_boundary, location='BLASTPol/CII/')
# runHRO(OI_B, Bmap=BLAST_Bmap, Qmap=BLAST_Q, Umap=BLAST_U, kstep=2, vecMask=BandC_boundary, location='BLASTPol/OI/')
# runHRO(Mop_HNC_B, Bmap=BLAST_Bmap, Qmap=BLAST_Q, Umap=BLAST_U, kstep=4, vecMask=BandC_boundary, location='BLASTPol/Mopra_HNC/')
# runHRO(Mop_N2H_B, Bmap=BLAST_Bmap, Qmap=BLAST_Q, Umap=BLAST_U, kstep=4, vecMask=BandC_boundary, location='BLASTPol/Mopra_N2H/')
# runHRO(Mop_C18O_B, Bmap=BLAST_Bmap, Qmap=BLAST_Q, Umap=BLAST_U, kstep=4, vecMask=BandC_boundary, location='BLASTPol/Mopra_C18O/')
# runHRO(Spitz_Ch1, Bmap=BLAST_Bmap, Qmap=BLAST_Q, Umap=BLAST_U, kstep=2, vecMask=BandC_boundary, location='BLASTPol/Spitz_Ch1/')
# runHRO(Spitz_Ch3, Bmap=BLAST_Bmap, Qmap=BLAST_Q, Umap=BLAST_U, kstep=2, vecMask=BandC_boundary, location='BLASTPol/Spitz_Ch3/')
# runHRO(Halpha_B, Bmap=BLAST_Bmap, Qmap=BLAST_Q, Umap=BLAST_U, kstep=2, vecMask=BandC_boundary, location='BLASTPol/Halpha/')
# runHRO(ALMA_ACA, Bmap=BLAST_Bmap, Qmap=BLAST_Q, Umap=BLAST_U, kstep=2, vecMask=BandC_boundary, location='BLASTPol/ALMA_cont/')
# runHRO(NcolLow_B, Bmap=BLAST_Bmap, Qmap=BLAST_Q, Umap=BLAST_U, kstep=3, vecMask=BandC_boundary, location='BLASTPol/Ncol_Av_Low/')
# runHRO(TempLow_B, Bmap=BLAST_Bmap, Qmap=BLAST_Q, Umap=BLAST_U, kstep=3, vecMask=BandC_boundary, location='BLASTPol/Temp_Low/')
# runHRO(TempHigh_B, Bmap=BLAST_Bmap, Qmap=BLAST_Q, Umap=BLAST_U, kstep=3, vecMask=BandC_boundary, location='BLASTPol/Temp_High/')
# runHRO(HAWE[0], Bmap=BLAST_Bmap, Qmap=BLAST_Q, Umap=BLAST_U, kstep=2, vecMask=BandC_boundary, location='BLASTPol/BandE_I/')
# runHRO(HAWC[0], Bmap=BLAST_Bmap, Qmap=BLAST_Q, Umap=BLAST_U, kstep=2, vecMask=BandC_boundary, location='BLASTPol/BandC_I/')



# kstep=2
# runHRO(H70_B, BLAST_Bmap, kstep=kstep, vecMask=BandE_boundary, location='BLASTPol/Hersch_70/', Bproj=True)
# runHRO(H160_B, BLAST_Bmap, kstep=kstep, vecMask=BandE_boundary, location='BLASTPol/Hersch_160/', Bproj=True)
# runHRO(H250_B, BLAST_Bmap, kstep=kstep, vecMask=BandE_boundary, location='BLASTPol/Hersch_250/', Bproj=True)
# runHRO(H350_B, BLAST_Bmap, kstep=kstep, vecMask=BandE_boundary, location='BLASTPol/Hersch_350/', Bproj=True)
# runHRO(H500_B, BLAST_Bmap, kstep=kstep, vecMask=BandE_boundary, location='BLASTPol/Hersch_500/', Bproj=True)
# runHRO(Ncol_B, BLAST_Bmap, kstep=kstep, vecMask=BandE_boundary, location='BLASTPol/Hersch_ColDen/', Bproj=True)
# kstep=3
# runHRO(H70_B, BLAST_Bmap, kstep=kstep, vecMask=BLAST_mask, location='BLASTPol/Hersch_70/', Bproj=True)
# kstep=4
# runHRO(H70_B, BLAST_Bmap, kstep=kstep, vecMask=BLAST_mask, location='BLASTPol/Hersch_70/', Bproj=True)

# kstep=3
# runHRO_QU(Ncol_B, Qmap=BLAST_Q, Umap=BLAST_U, kstep=kstep, vecMask=BLAST_mask, location='BLASTPol/Ncol/') 
# kstep=4
# runHRO_QU(Ncol_B, Qmap=BLAST_Q, Umap=BLAST_U, kstep=kstep, vecMask=BLAST_mask, location='BLASTPol/Ncol/') 

# kstep=2
# runHRO_QU(ALMA_ACA, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=ALMA_mask, location='BandC/ALMA_cont/') 
# kstep=3
# runHRO_QU(ALMA_ACA, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=ALMA_mask, location='BandC/ALMA_cont/') 
# kstep=4
# runHRO_QU(ALMA_ACA, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=ALMA_mask, location='BandC/ALMA_cont/') 
# kstep=5
# runHRO_QU(ALMA_ACA, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=ALMA_mask, location='BandC/ALMA_cont/') 
# kstep=6
# runHRO_QU(ALMA_ACA, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/ALMA_cont/') 
# kstep=7
# runHRO_QU(ALMA_ACA, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/ALMA_cont/') 
# kstep=8
# runHRO_QU(ALMA_ACA, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/ALMA_cont/') 

# kstep=3
# runHRO_QU(Spitz_fix1, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch1/')
# runHRO_QU(Spitz_fix3, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch3/') 
# kstep=4
# runHRO_QU(Spitz_fix1, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch1/')
# runHRO_QU(Spitz_fix3, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch3/') 
# kstep=5
# runHRO_QU(Spitz_fix1, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch1/')
# runHRO_QU(Spitz_fix3, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch3/') 
# kstep=3
# runHRO_QU(Spitz1_HAWC, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch1/')
# runHRO_QU(Spitz3_HAWC, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch3/')  
# kstep=4
#runHRO_QU(Spitz1_HAWC, Qmap=HAWC[2], Umap=HAWC[4], kstep=4, vecMask=vecMask_C, location='BandC/Spitz_Ch1/')
# runHRO_QU(Spitz3_HAWC, Qmap=HAWC[2], Umap=HAWC[4], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch3/')  
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

#runCube(CO12_cube, HAWC[11], vecMask=vecMask_C, kstep=2, location='BandC/12CO/Cube/', mask=True, thres=3, speed=[-50,30], Noise=True, counter=100)
#runWhiteNoise_HRO(Ncol, HAWC[11], kstep=2, vecMask=vecMask_C, location='BandC/Ncol_Av/')

# # loopNoise(Ncol, HAWC[11], kstep=2, vecMask=vecMask_C, mask=vecMask_C, location='BandC/Ncol_Av/', counter=1000)
# # loopNoise(CO12, HAWC[11], kstep=2, vecMask=vecMask_C, mask=vecMask_C, location='BandC/12CO/', counter=1000)
# # loopNoise(CO13, HAWC[11], kstep=2, vecMask=vecMask_C, mask=vecMask_C, location='BandC/13CO/', counter=1000)
# # loopNoise(CII, HAWC[11], kstep=2, vecMask=vecMask_C, mask=vecMask_C, location='BandC/CII/', counter=1000)
# # loopNoise(HAWE[0], HAWC[11], kstep=2, vecMask=vecMask_C, mask=vecMask_C, location='BandC/BandE_I/', counter=1000)
# # loopNoise(HAWC[0], HAWC[11], kstep=2, vecMask=vecMask_C, mask=vecMask_C, location='BandC/BandC_I/', counter=1000)
# # loopNoise(Ncol_Low, HAWC[11], kstep=3, vecMask=vecMask_C, mask=vecMask_C, location='BandC/Ncol_Av_Low/', counter=1000)
# # loopNoise(Temp_Low, HAWC[11], kstep=3, vecMask=vecMask_C, mask=vecMask_C, location='BandC/Temp_Low/', counter=1000)
# # loopNoise(Temp_High, HAWC[11], kstep=3, vecMask=vecMask_C, mask=vecMask_C, location='BandC/Temp_High/', counter=1000)
# loopNoise(H_70, HAWC[11], kstep=2, vecMask=vecMask_C, mask=vecMask_C, location='BandC/Hersh_70/', counter=1000)
# loopNoise(H_160, HAWC[11], kstep=2, vecMask=vecMask_C, mask=vecMask_C, location='BandC/Hersh_160/', counter=1000)
# loopNoise(H_250, HAWC[11], kstep=3, vecMask=vecMask_C, mask=vecMask_C, location='BandC/Hersh_250/', counter=1000)
# loopNoise(H_350, HAWC[11], kstep=3, vecMask=vecMask_C, mask=vecMask_C, location='BandC/Hersh_350/', counter=1000)
# loopNoise(H_500, HAWC[11], kstep=3, vecMask=vecMask_C, mask=vecMask_C, location='BandC/Hersh_500/', counter=1000)
# # loopNoise(Mop_HNC_more, HAWC[11], kstep=4, vecMask=vecMask_C, mask=vecMask_C, location='BandC/Mopra_HNC_n10to30/', counter=1000)
# # loopNoise(Mop_C18O, HAWC[11], kstep=4, vecMask=vecMask_C, mask=vecMask_C, location='BandC/Mopra_C18O_n10to30/', counter=1000)
# # loopNoise(Mop_N2H, HAWC[11], kstep=4, vecMask=vecMask_C, mask=vecMask_C, location='BandC/Mopra_N2H_n10to30/', counter=1000)
# # loopNoise(OI, HAWC[11], kstep=2, vecMask=vecMask_C, mask=vecMask_C, location='BandC/OI/', counter=1000)
# loopNoise(Halpha, HAWC[11], kstep=2, vecMask=vecMask_C, mask=vecMask_C, location='BandC/Halpha/', counter=1000)

# loopNoise(Ncol, HAWE[11], kstep=2, vecMask=vecMask_E, mask=vecMask_E, location='BandE/Ncol_Av/', counter=1000)
# loopNoise(CO12, HAWE[11], kstep=2, vecMask=vecMask_E, mask=vecMask_E, location='BandE/12CO/', counter=1000)
# loopNoise(CO13, HAWE[11], kstep=2, vecMask=vecMask_E, mask=vecMask_E, location='BandE/13CO/', counter=1000)
# loopNoise(CII, HAWE[11], kstep=2, vecMask=vecMask_E, mask=vecMask_E, location='BandE/CII/', counter=1000)
# #loopNoise(HAWE[0], HAWE[11], kstep=2, vecMask=vecMask_E, mask=vecMask_E, location='BandE/BandE_I/', counter=1000)
# #loopNoise(HAWC[0], HAWE[11], kstep=2, vecMask=vecMask_E, mask=vecMask_E, location='BandE/BandC_I/', counter=1000)
# #loopNoise(Ncol_Low, HAWE[11], kstep=3, vecMask=vecMask_E, mask=vecMask_E, location='BandE/Ncol_Av_Low/', counter=1000)
# #loopNoise(Temp_Low, HAWE[11], kstep=3, vecMask=vecMask_E, mask=vecMask_E, location='BandE/Temp_Low/', counter=1000)
# loopNoise(Temp_High, HAWE[11], kstep=3, vecMask=vecMask_E, mask=vecMask_E, location='BandE/Temp_High/', counter=1000)
# loopNoise(H_70, HAWE[11], kstep=2, vecMask=vecMask_E, mask=vecMask_E, location='BandE/Hersh_70/', counter=1000)
# loopNoise(H_160, HAWE[11], kstep=2, vecMask=vecMask_E, mask=vecMask_E, location='BandE/Hersh_160/', counter=1000)
# loopNoise(H_250, HAWE[11], kstep=3, vecMask=vecMask_E, mask=vecMask_E, location='BandE/Hersh_250/', counter=1000)
# loopNoise(H_350, HAWE[11], kstep=3, vecMask=vecMask_E, mask=vecMask_E, location='BandE/Hersh_350/', counter=1000)
# loopNoise(H_500, HAWE[11], kstep=3, vecMask=vecMask_E, mask=vecMask_E, location='BandE/Hersh_500/', counter=1000)
# loopNoise(Mop_HNC_more, HAWE[11], kstep=4, vecMask=vecMask_E, mask=vecMask_E, location='BandE/Mopra_HNC/', counter=1000)
# loopNoise(Mop_C18O, HAWE[11], kstep=4, vecMask=vecMask_E, mask=vecMask_E, location='BandE/Mopra_C18O/', counter=1000)
# loopNoise(Mop_N2H, HAWE[11], kstep=4, vecMask=vecMask_E, mask=vecMask_E, location='BandE/Mopra_N2H/', counter=1000)
# # loopNoise(Mop_HNC_less, HAWE[11], kstep=4, vecMask=vecMask_E, location='BandE/Mopra_HNC_0to12/kstep4/')
# # loopNoise(Mop_HNC_more, HAWE[11], kstep=4, vecMask=vecMask_E, location='BandE/Mopra_HNC_n10to30/kstep4/')
# # loopNoise(Mop_C18O, HAWE[11], kstep=4, vecMask=vecMask_E, location='BandE/Mopra_C18O_n10to30/kstep4/')
# # loopNoise(Mop_N2H, HAWE[11], kstep=4, vecMask=vecMask_E, location='BandE/Mopra_N2H_n10to30/kstep4/')
# # loopNoise(Spitz_Ch1, HAWE[11], kstep=3, vecMask=vecMask_E, location='BandE/Spitz_Ch1/kstep3/')
# # loopNoise(Spitz_Ch3, HAWE[11], kstep=3, vecMask=vecMask_E, location='BandE/Spitz_Ch3/kstep3/')
# # loopNoise(Spitz_Ch1, HAWE[11], kstep=4, vecMask=vecMask_E, location='BandE/Spitz_Ch1/kstep4/')
# # loopNoise(Spitz_Ch3, HAWE[11], kstep=4, vecMask=vecMask_E, location='BandE/Spitz_Ch3/kstep4/')
# # loopNoise(Spitz_Ch1, HAWE[11], kstep=6, vecMask=vecMask_E, location='BandE/Spitz_Ch1/kstep6/')
# # loopNoise(Spitz_Ch3, HAWE[11], kstep=6, vecMask=vecMask_E, location='BandE/Spitz_Ch3/kstep6/')
# loopNoise(OI, HAWE[11], kstep=2, vecMask=vecMask_E, mask=vecMask_E, location='BandE/OI/', counter=1000)
# loopNoise(Halpha, HAWE[11], kstep=2, vecMask=vecMask_E, mask=vecMask_E, location='BandE/Halpha/', counter=1000)
# # loopNoise(ALMA_ACA, HAWE[11], kstep=3, vecMask=vecMask_E, location='BandE/ALMA_cont/kstep3/')
# # loopNoise(OI, HAWE[11], kstep=4, vecMask=vecMask_E, location='BandE/OI/kstep4/')
# # loopNoise(Halpha, HAWE[11], kstep=4, vecMask=vecMask_E, location='BandE/Halpha/kstep4/')
# # loopNoise(ALMA_ACA, HAWE[11], kstep=4, vecMask=vecMask_E, location='BandE/ALMA_cont/kstep4/')

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
# runHRO(Mop_HNC_more, HAWE[11], kstep=4, vecMask=vecMask_E, hdu=HNC_BandE_msk, location='BandE/Mopra_HNC/Masked/kstep4/')
# runHRO(Mop_C18O, HAWE[11], kstep=4, vecMask=vecMask_E, hdu=C18O_BandE_msk, location='BandE/Mopra_C18O/Masked/step4/')
# runHRO(Mop_N2H, HAWE[11], kstep=4, vecMask=vecMask_E, hdu=N2H_BandE_msk, location='BandE/Mopra_N2H/Masked/kstep4/')
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

# runHRO(Mop_HNC_more, HAWC[11], kstep=4, vecMask=vecMask_C, hdu=HNC_BandC_msk, location='BandC/Mopra_HNC/Masked/kstep4/')
# runHRO(Mop_C18O, HAWC[11], kstep=4, vecMask=vecMask_C, hdu=C18O_BandC_msk, location='BandC/Mopra_C18O/Masked/step4/')
# runHRO(Mop_N2H, HAWC[11], kstep=4, vecMask=vecMask_C, hdu=N2H_BandC_msk, location='BandC/Mopra_N2H/Masked/kstep4/')

# runHRO(ALMA_12m, HAWE[11], Qmap=HAWE[2], Umap=HAWE[4],  kstep=2, vecMask=vecMask_E, hdu=ALMA_12m_mask, location='BandE/ALMA_12m_cont/Masked/kstep5/')
# runHRO(ALMA_12m, HAWE[11], Qmap=HAWE[2], Umap=HAWE[4],  kstep=5, vecMask=vecMask_E, hdu=ALMA_12m_mask, location='BandE/ALMA_12m_cont/Masked/kstep5/')
# runHRO(ALMA_12m, HAWE[11], Qmap=HAWE[2], Umap=HAWE[4],  kstep=7, vecMask=vecMask_E, hdu=ALMA_12m_mask, location='BandE/ALMA_12m_cont/Masked/kstep7/')

# kstep=2
#runHRO(Spitz_fix1, HAWC[11], kstep=2, vecMask=vecMask_C, location='BandC/Spitz_Ch1/')
#runHRO(Spitz_Ch1, HAWC[11], kstep=2, vecMask=vecMask_C, location='BandC/Spitz_Ch1/Spitz_projHAWC/')
#runHRO(Spitz_fix1, Qmap=HAWC[2], Umap=HAWC[4], Bmap=HAWC[11], kstep=6, vecMask=vecMask_C, location='BandC/Spitz_Ch1/')
# runHRO(Spitz_rez1, Qmap=HAWC[2], Umap=HAWC[4], Bmap=HAWC[11], kstep=2, vecMask=vecMask_C, location='BandC/Spitz_Ch1/')
# runHRO(Spitz_rez2, Qmap=HAWC[2], Umap=HAWC[4], Bmap=HAWC[11], kstep=2, vecMask=vecMask_C, location='BandC/Spitz_Ch2/')
# runHRO(Spitz_rez3, Qmap=HAWC[2], Umap=HAWC[4], Bmap=HAWC[11], kstep=2, vecMask=vecMask_C, location='BandC/Spitz_Ch3/')
# runHRO(Spitz_fix3, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch3/Spitz_proj/')
# kstep=3
# runHRO(Spitz_fix1, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch1/Spitz_proj/')
# runHRO(Spitz_fix3, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch3/Spitz_proj/')
# kstep=4
# runHRO(Spitz_fix1, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch1/Spitz_proj/')
# runHRO(Spitz_fix3, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch3/Spitz_proj/')
# kstep=5
#runHRO(Spitz_fix1, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch1/Spitz_proj/')
#runHRO(Spitz_fix3, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Spitz_Ch3/Spitz_proj/')
# ksteps=[2,3,4,5]
# for kstep in ksteps:
#     runHRO_QU(Spitz_fix1, Qmap=HAWE[2], Umap=HAWE[4], kstep=kstep, vecMask=vecMask_E, location='BandE/Spitz_Ch1/')
#     runHRO_QU(Spitz_fix3, Qmap=HAWE[2], Umap=HAWE[4], kstep=kstep, vecMask=vecMask_E, location='BandE/Spitz_Ch3/')
#     runHRO_QU(ALMA_ACA, Qmap=HAWE[2], Umap=HAWE[4], kstep=kstep, vecMask=ALMA_mask, location='BandE/ALMA_cont/')
#     runHRO(Spitz_fix1, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Spitz_Ch1/Spitz_proj/')
#     runHRO(Spitz_fix3, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Spitz_Ch3/Spitz_proj/') 


# for kstep in ksteps:
#     runHRO(Ncol, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Ncol_Av/')
#     runHRO(CO12, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/12CO/')
#     runHRO(CO13, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/13CO/')
#     runHRO(CII, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/CII/')
#     runHRO(HAWE[0], HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/BandE_I/')
#     runHRO(HAWC[0], HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/BandC_I/')
#     runHRO(Ncol_Low, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Ncol_Av_Low/')
#     runHRO(Temp_Low, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Temp_Low/')
#     runHRO(Temp_High, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Temp_High/')
#     runHRO(H_70, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Hersh_70/')
#     runHRO(H_160, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Hersh_160/')
#     runHRO(H_250, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Hersh_250/')
#     runHRO(H_350, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Hersh_350/')
#     runHRO(H_500, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Hersh_500/')
#     runHRO(Mop_HNC_less, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Mopra_HNC_0to12/')
#     runHRO(Mop_HNC_more, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Mopra_HNC_n10to30/')
#     runHRO(Mop_C18O, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Mopra_C18O_n10to30/')
#     runHRO(Mop_N2H, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Mopra_N2H_n10to30/')
#     runHRO(OI, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/OI/')
#     runHRO(Halpha, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Halpha/')

# for kstep in ksteps:
#     runHRO(Ncol, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Ncol_Av/')
#     runHRO(CO12, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/12CO/')
#     runHRO(CO13, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/13CO/')
#     runHRO(CII, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/CII/')
#     runHRO(HAWE[0], HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/BandE_I/')
#     runHRO(HAWC[0], HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/BandC_I/')
#     runHRO(Ncol_Low, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Ncol_Av_Low/')
#     runHRO(Temp_Low, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Temp_Low/')
#     runHRO(Temp_High, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Temp_High/')
#     runHRO(H_70, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Hersh_70/')
#     runHRO(H_160, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Hersh_160/')
#     runHRO(H_250, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Hersh_250/')
#     runHRO(H_350, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Hersh_350/')
#     runHRO(H_500, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Hersh_500/')
#     runHRO(Mop_HNC_less, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Mopra_HNC_0to12/')
#     runHRO(Mop_HNC_more, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Mopra_HNC_n10to30/')
#     runHRO(Mop_C18O, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Mopra_C18O_n10to30/')
#     runHRO(Mop_N2H, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Mopra_N2H_n10to30/')
#     runHRO(OI, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/OI/')
#     runHRO(Halpha, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Halpha/')
# kstep=2
# runHRO(ALMA_ACA, HAWC[11], kstep=kstep, vecMask=ALMA_mask, location='BandC/ALMA_cont/ALMA_proj/')
# kstep=3
# runHRO(ALMA_ACA, HAWC[11], kstep=kstep, vecMask=ALMA_mask, location='BandC/ALMA_cont/ALMA_proj/')
# kstep=4
# runHRO(ALMA_ACA, HAWC[11], kstep=kstep, vecMask=ALMA_mask, location='BandC/ALMA_cont/ALMA_proj/')
# kstep=5
# runHRO(ALMA_ACA, HAWC[11], kstep=kstep, vecMask=ALMA_mask, location='BandC/ALMA_cont/ALMA_proj/')
# kstep=2
# runHRO(Mop_N2H_B, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Mopra_N2H_cutout/')
# runHRO(Mop_N2H_B, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Mopra_N2H_cutout/')
# runHRO(Mop_C18O_B, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Mopra_C18O_cutout/')
# runHRO(Mop_C18O_B, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Mopra_C18O_cutout/')
# kstep=3
# runHRO(Mop_N2H_B, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Mopra_N2H_cutout/')
# runHRO(Mop_N2H_B, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Mopra_N2H_cutout/')
# runHRO(Mop_C18O_B, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Mopra_C18O_cutout/')
# runHRO(Mop_C18O_B, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Mopra_C18O_cutout/')
#kstep=2
# runHRO(Mop_N2H_B, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Mopra_N2H_cutout/')
# runHRO(Mop_N2H_B, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Mopra_N2H_cutout/')
# runHRO(Mop_C18O_B, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Mopra_C18O_cutout/')
# runHRO(Mop_C18O_B, HAWE[11], kstep=kstep, vecMask=vecMask_E, location='BandE/Mopra_C18O_cutout/')
# runHRO(Ncol, HAWC[11], kstep=kstep, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Ncol_Av/')
# runHRO(CO12, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/12CO/')
# runHRO(CO13, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/13CO/')
# runHRO(CII, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/CII/')
# runHRO(HAWE[0], HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/BandE_I/')
# runHRO(HAWC[0], HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/BandC_I/')
# runHRO(Ncol_Low, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Ncol_Av_Low/')
# runHRO(Temp_Low, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Temp_Low/')
# runHRO(Temp_High, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Temp_High/')
# runHRO(H_70, HAWC[11], kstep=kstep, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Hersh_70/')
# runHRO(H_160, HAWC[11], kstep=kstep, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Hersh_160/')
# runHRO(H_250, HAWC[11], kstep=kstep, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Hersh_250/')
# runHRO(H_350, HAWC[11], kstep=kstep, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Hersh_350/')
# runHRO(H_500, HAWC[11], kstep=kstep, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Hersh_500/')
# runHRO(Mop_HNC_less, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Mopra_HNC_0to12/')
# runHRO(Mop_HNC_more, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Mopra_HNC_n10to30/')
# runHRO(Mop_HNC_B, HAWC[11], kstep=kstep, vecMask=vecMask_C, location='BandC/Mopra_HNC_cutout/')
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

# runHRO(Ncol, HAWC[11], FWHM=18, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Ncol_Av/')
# runHRO(Temp_High, HAWC[11], kstep=4, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Temp_High/')
# runHRO(H_70, HAWC[11], FWHM=5, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Hersh_70/')
# runHRO(H_70, Qmap=HAWC[2], Umap=HAWC[4], Bmap=HAWC[11], FWHM=5, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Hersh_70/native/')
# runHRO(H_160, HAWC[11], FWHM=12, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Hersh_160/')
# runHRO(H_250, HAWC[11], FWHM=18, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Hersh_250/')
# runHRO(H_350, HAWC[11], FWHM=25, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Hersh_350/')
# runHRO(H_500, HAWC[11], FWHM=36, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Hersh_500/')
# runHRO(HAWC[0], HAWC[11], FWHM=7.8, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/BandC_I/')
# runHRO(HAWE[0], HAWC[11], FWHM=18.2, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/BandE_I/')
# runHRO(Mop_C18O, HAWC[11], FWHM=33, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Mopra_C18O/')
# runHRO(Mop_HNC, HAWC[11], FWHM=36, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Mopra_HNC/')
# runHRO(Mop_N2H, HAWC[11], FWHM=36, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Mopra_N2H/')
# runHRO(CO12, HAWC[11], FWHM=18.2, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/12CO/')
# runHRO(CO13, HAWC[11], FWHM=18.2, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/13CO/')
# runHRO(CII, HAWC[11], FWHM=20, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/CII/')
# runHRO(OI, HAWC[11], FWHM=30, vecMask=vecMask_C, hdu=vecMask_C, location='BandC/OI/')
# runHRO(ALMA_12m, FWHM=1.4, Qmap=HAWC[2], Umap=HAWC[4], Bmap=HAWC[11], vecMask=vecMask_C, location='BandC/ALMA_12m/', hdu=ALMA_12m_mask)
# runHRO(ALMA_ACA, FWHM=5.4, Qmap=HAWC[2], Umap=HAWC[4], Bmap=HAWC[11], vecMask=vecMask_C, location='BandC/ALMA_ACA/', hdu=ALMA_mask)
# runHRO(Spitz_rez1, FWHM=1.66, Qmap=HAWC[2], Umap=HAWC[4], Bmap=HAWC[11], vecMask=vecMask_C, location='BandC/Spitz_Ch1/')
# runHRO(Spitz_rez2, FWHM= 1.7, Qmap=HAWC[2], Umap=HAWC[4], Bmap=HAWC[11], vecMask=vecMask_C, location='BandC/Spitz_Ch2/')
# runHRO(Spitz_rez3, FWHM=1.88, Qmap=HAWC[2], Umap=HAWC[4], Bmap=HAWC[11], vecMask=vecMask_C, location='BandC/Spitz_Ch3/')

# loopNoise(Ncol, HAWC[11], vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Ncol_Av/', counter=1000)
# loopNoise(Temp_High, HAWC[11], vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Temp_High/', counter=1000)
# loopNoise(H_70, HAWC[11], vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Hersh_70/', counter=1000)
# loopNoise(H_160, HAWC[11], vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Hersh_160/', counter=1000)
# loopNoise(H_250, HAWC[11], vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Hersh_250/', counter=1000)
# loopNoise(H_350, HAWC[11], vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Hersh_350/', counter=1000)
# loopNoise(H_500, HAWC[11], vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Hersh_500/', counter=1000)
# loopNoise(HAWC[0], HAWC[11], vecMask=vecMask_C, hdu=vecMask_C, location='BandC/BandC_I/', counter=1000)
# loopNoise(HAWE[0], HAWC[11], vecMask=vecMask_C, hdu=vecMask_C, location='BandC/BandE_I/', counter=1000)
# loopNoise(Mop_C18O, HAWC[11], vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Mopra_C18O/', counter=1000)
# loopNoise(Mop_HNC, HAWC[11], vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Mopra_HNC/', counter=1000)
# loopNoise(Mop_N2H, HAWC[11], vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Mopra_N2H/', counter=1000)
# loopNoise(CO12, HAWC[11], vecMask=vecMask_C, hdu=vecMask_C, location='BandC/12CO/', counter=1000)
# loopNoise(CO13, HAWC[11], vecMask=vecMask_C, hdu=vecMask_C, location='BandC/13CO/', counter=1000)
# loopNoise(CII, HAWC[11], vecMask=vecMask_C, hdu=vecMask_C, location='BandC/CII/', counter=1000)
# loopNoise(OI, HAWC[11], vecMask=vecMask_C, hdu=vecMask_C, location='BandC/OI/', counter=1000)
# loopNoise(H_70, Qmap=HAWC[2], Umap=HAWC[4], Bmap=HAWC[11], vecMask=vecMask_C, hdu=vecMask_C, location='BandC/Hersh_70/native/', counter=1000)
# loopNoise(ALMA_12m, Qmap=HAWC[2], Umap=HAWC[4], Bmap=HAWC[11], vecMask=vecMask_C, location='BandC/ALMA_12m/', hdu=ALMA_12m_mask, counter=1000)
# loopNoise(ALMA_ACA, Qmap=HAWC[2], Umap=HAWC[4], Bmap=HAWC[11], vecMask=vecMask_C, location='BandC/ALMA_ACA/', hdu=ALMA_mask, counter=1000)
# loopNoise(Spitz_rez2, Qmap=HAWC[2], Umap=HAWC[4], Bmap=HAWC[11], vecMask=vecMask_C, location='BandC/Spitz_Ch2/', hdu=vecMask_C, counter=1000)
# loopNoise(Spitz_rez3, Qmap=HAWC[2], Umap=HAWC[4], Bmap=HAWC[11], vecMask=vecMask_C, location='BandC/Spitz_Ch3/', hdu=vecMask_C, counter=1000)
# loopNoise(Spitz_rez1, Qmap=HAWC[2], Umap=HAWC[4], Bmap=HAWC[11], vecMask=vecMask_C, location='BandC/Spitz_Ch1/', hdu=vecMask_C, counter=1000)

#runHRO(Ncol, HAWE[11], FWHM=18, vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Ncol_Av/')
# runHRO(Temp_High, HAWE[11], kstep=4, vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Temp_High/')
# runHRO(H_70, HAWE[11], FWHM=5, vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Hersh_70/')
# runHRO(H_70, Qmap=HAWE[2], Umap=HAWE[4], Bmap=HAWE[11], FWHM=5, vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Hersh_70/native/')
# runHRO(H_160, HAWE[11], FWHM=12, vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Hersh_160/')
# runHRO(H_250, HAWE[11], FWHM=18, vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Hersh_250/')
# runHRO(H_350, HAWE[11], FWHM=25, vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Hersh_350/')
# runHRO(H_500, HAWE[11], FWHM=36, vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Hersh_500/')
# runHRO(HAWC[0], HAWE[11], FWHM=7.8, vecMask=vecMask_E, hdu=vecMask_E, location='BandE/BandC_I/')
# runHRO(HAWE[0], HAWE[11], FWHM=18.2, vecMask=vecMask_E, hdu=vecMask_E, location='BandE/BandE_I/')
# runHRO(Mop_C18O, HAWE[11], FWHM=33, vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Mopra_C18O/')
# runHRO(Mop_HNC, HAWE[11], FWHM=36, vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Mopra_HNC/')
# runHRO(Mop_N2H, HAWE[11], FWHM=36, vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Mopra_N2H/')
# runHRO(CO12, HAWE[11], FWHM=18.2, vecMask=vecMask_E, hdu=vecMask_E, location='BandE/12CO/')
# runHRO(CO13, HAWE[11], FWHM=18.2, vecMask=vecMask_E, hdu=vecMask_E, location='BandE/13CO/')
# runHRO(CII, HAWE[11], FWHM=20, vecMask=vecMask_E, hdu=vecMask_E, location='BandE/CII/')
# runHRO(OI, HAWE[11], FWHM=30, vecMask=vecMask_E, hdu=vecMask_E, location='BandE/OI/')
#runHRO(H_160, Qmap=HAWE[2], Umap=HAWE[4], Bmap=HAWE[11], FWHM=12, vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Hersh_160/native/')
#runHRO(HAWC[0], Qmap=HAWE[2], Umap=HAWE[4], Bmap=HAWE[11], FWHM=36, vecMask=vecMask_E, hdu=vecMask_E, location='BandE/BandC_I/native/')
# runHRO(ALMA_12m, FWHM=1.4, Qmap=HAWE[2], Umap=HAWE[4], Bmap=HAWE[11], vecMask=vecMask_E, location='BandE/ALMA_12m/', hdu=ALMA_12m_mask)
# runHRO(ALMA_ACA, FWHM=5.4, Qmap=HAWE[2], Umap=HAWE[4], Bmap=HAWE[11], vecMask=vecMask_E, location='BandE/ALMA_ACA/', hdu=ALMA_mask)
# runHRO(Spitz_rez1, FWHM=1.66, Qmap=HAWE[2], Umap=HAWE[4], Bmap=HAWE[11], vecMask=vecMask_E, location='BandE/Spitz_Ch1/')
# runHRO(Spitz_rez2, FWHM= 1.7, Qmap=HAWE[2], Umap=HAWE[4], Bmap=HAWE[11], vecMask=vecMask_E, location='BandE/Spitz_Ch2/')
# runHRO(Spitz_rez3, FWHM=1.88, Qmap=HAWE[2], Umap=HAWE[4], Bmap=HAWE[11], vecMask=vecMask_E, location='BandE/Spitz_Ch3/')

# loopNoise(Ncol, HAWE[11], vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Ncol_Av/', counter=1000)
# loopNoise(Temp_High, HAWE[11], vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Temp_High/',  counter=1000)
# loopNoise(H_70, HAWE[11], vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Hersh_70/',  counter=1000)
# loopNoise(H_160, HAWE[11], vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Hersh_160/',  counter=1000)
# loopNoise(H_250, HAWE[11], vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Hersh_250/', counter=1000)
# loopNoise(H_350, HAWE[11], vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Hersh_350/', counter=1000)
# loopNoise(H_500, HAWE[11], vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Hersh_500/', counter=1000)
# loopNoise(HAWC[0], HAWE[11], vecMask=vecMask_E, hdu=vecMask_E, location='BandE/BandC_I/', counter=1000)
# loopNoise(HAWE[0], HAWE[11], vecMask=vecMask_E, hdu=vecMask_E, location='BandE/BandE_I/', counter=1000)
# loopNoise(Mop_C18O, HAWE[11], vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Mopra_C18O/', counter=1000)
# loopNoise(Mop_HNC, HAWE[11], vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Mopra_HNC/', counter=1000)
# loopNoise(Mop_N2H, HAWE[11], vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Mopra_N2H/', counter=1000)
# loopNoise(CO12, HAWE[11], vecMask=vecMask_E, hdu=vecMask_E, location='BandE/12CO/', counter=1000)
# loopNoise(CO13, HAWE[11], vecMask=vecMask_E, hdu=vecMask_E, location='BandE/13CO/', counter=1000)
# loopNoise(CII, HAWE[11], vecMask=vecMask_E, hdu=vecMask_E, location='BandE/CII/', counter=1000)
# loopNoise(OI, HAWE[11], vecMask=vecMask_E, hdu=vecMask_E, location='BandE/OI/', counter=1000)
# loopNoise(H_70, Qmap=HAWE[2], Umap=HAWE[4], Bmap=HAWE[11], vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Hersh_70/native/', counter=1000)
#loopNoise(HAWC[0], Qmap=HAWE[2], Umap=HAWE[4], Bmap=HAWE[11], vecMask=vecMask_E, hdu=vecMask_E, location='BandE/BandC_I/native/', counter=1000)
#loopNoise(H_160, Qmap=HAWE[2], Umap=HAWE[4], Bmap=HAWE[11], vecMask=vecMask_E, hdu=vecMask_E, location='BandE/Hersh_160/native/',  counter=1000)
# loopNoise(ALMA_12m, Qmap=HAWE[2], Umap=HAWE[4], Bmap=HAWE[11], vecMask=vecMask_E, location='BandE/ALMA_12m/', hdu=ALMA_12m_mask, counter=1000)
# loopNoise(ALMA_ACA, Qmap=HAWE[2], Umap=HAWE[4], Bmap=HAWE[11], vecMask=vecMask_E, location='BandE/ALMA_ACA/', hdu=ALMA_mask, counter=1000)
# loopNoise(Spitz_rez2, Qmap=HAWE[2], Umap=HAWE[4], Bmap=HAWE[11], vecMask=vecMask_E, location='BandE/Spitz_Ch2/', hdu=vecMask_E, counter=1000)
# loopNoise(Spitz_rez3, Qmap=HAWE[2], Umap=HAWE[4], Bmap=HAWE[11], vecMask=vecMask_E, location='BandE/Spitz_Ch3/', hdu=vecMask_E, counter=1000)
# loopNoise(Spitz_rez1, Qmap=HAWE[2], Umap=HAWE[4], Bmap=HAWE[11], vecMask=vecMask_E, location='BandE/Spitz_Ch1/', hdu=vecMask_E, counter=1000)

#PRScorr_all(sub_fldr='BandC/')