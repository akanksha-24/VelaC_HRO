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

# Maps 
HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
HAWE = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_031-040_POL_70060912_HAWE_HWPE_PMP.fits')
Hersh_70 = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_70_flat.fits')[0]
Hersh_160 = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_160_flat.fits')[0]
Hersh_250 = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_hipe11_250.fits')[0]
Hersh_350 = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_hipe11_350_10as.fits')[0]
Ncol = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits')[0]
Temp = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresTmap_velac_temperature_cf_r500_medsmo3.fits')[0]
Hersh_500 = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/vela_herschel_hipe11_500.fits')[0]
vecMask = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/masks/rereduced/BandC_polFlux_3.fits')[0]
CO12 = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_integratedIntensity12CO.fits')[0]
CO13 = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CO/RCW36_integratedIntensity13CO.fits')[0]
CII = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/CII/RCW36_integratedIntensityCII.fits')[0]
OI = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/CO_LarsBonne/O/RCW36_OI_Integrated.fits')[0]
HNC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Mopra/HNC_Integrated_mom0_2to10kmpers.fits')[0]
C18O = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Mopra/C18O_Integrated_mom0_2to10kmpers.fits')[0]
N2H = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Mopra/N2H_Integrated_mom0_5to15kmpers.fits')[0]
spitzer_CH2 = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Spitzer/Spitz_ch2_rezmatch.fits')[0]
spitzer_CH1 = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Spitzer/Spitz_ch1_rezmatch.fits')[0]
ALMA_ACA = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/VelaC_CR_ALMA2D.fits')[0]
ALMA_12m = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/ALMA/Cycle8_12m/VelaC_CR1_12m_Cont_flat2D.fits')[0]
Tmap = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresTmap_velac_temperature_cf_r500_medsmo3.fits')[0]

All_Maps = [Hersh_500, Hersh_350, Hersh_250, HAWE[0], Hersh_160, HAWC[0], Hersh_70, spitzer_CH2, spitzer_CH1, 
            ALMA_ACA, ALMA_12m, Ncol, Tmap, CO12, CO13, CII, OI, HNC, C18O, N2H]

All_vmax = [1.520e-03, 16765, 10, 30, 180, 100, 6000, 9000, 9000, 100, 500, 100000]
All_vmin = [-1.362e-04, -244, -0.255, 18, -5, -5, -2200, -2000, -200, -4, -50, -20000]
All_vmid = [0, -3978, -2.6]

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 11.7}

plt.rc('font', **font)

ra_center = HAWC[0].header['CRVAL1']
dec_center = HAWC[0].header['CRVAL2']

def projectMap(mapOrigin, ref):
    '''This function projects a given map 'mapOrigin' onto the same WCS coordinates as a reference map and returns the map in the same shape'''
    New = ref.copy()
    proj, footprint = reproject_exact(mapOrigin, ref.header)
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

def Results_Flip4(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, 
                title, highrez=[False, False, False, False], interpolate=False):
    nrows=4
    ncols=3
    panels=nrows*ncols

    x = 0.97; dx = 0.009 ; dy = 0.06
    boxes = [[x, 0.72, dx, dy], [x, 0.53, dx, dy], [x, 0.34, dx, dy], [x, 0.15, dx, dy]]
    fig = plt.figure(figsize=(7, 9.5))

    for i in range(len(location)):
        hro = np.load(prefix+location[i]+'HRO_results.npz', allow_pickle=True)
        mapProj = projectMap(Maps[i], HAWC[0])

        ax = plt.subplot(nrows,ncols,i*ncols+1)
        angle = np.linspace(0, 90, hro['histbins'])
        plt.plot(angle, hro['hist'], linewidth=2.5, color='black', linestyle='--')
        plt.xlim(0,90)
        plt.ylim(0, 1.7)
        asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
        print("aspect is ", asp)
        ax.set_aspect(asp)
        #ax.yaxis.set_label_position("right")
        #ax.yaxis.tick_right()
        ax.set_yticks([0.5, 1.0, 1.5])
        ax.text(10, 1.4, Map_labels[i])
        rainbow_fill_between(ax, angle, np.zeros(angle.shape), hro['hist'], alpha=0.6)
        ax.text(10, 1.2, 'Z$_{\mathrm{x}}$'+' = {0}'.format(np.round(hro['Zx_corr'], 1)))
        ax.text(10, 1.0, 'Z\'$_{\mathrm{x}}$'+' = {0}'.format(np.round(hro['Zx_unCorr'], 1)))
        # ax.fill_between(angle[(np.where(angle<30))],hro['hist'][(np.where(angle<30))], facecolor='red', alpha=0.5)
        # ax.fill_between(angle[(np.where(angle>60))],hro['hist'][(np.where(angle>60))], facecolor='blue', alpha=0.5)
        # plt.text(5, 0.10, 'Parallel', fontsize=7)
        # plt.text(60, 0.10, 'Perpendicular', fontsize=7)
        if i==0:
            ax.set_title(title, fontsize=14, pad=17)

        # if (i!=2):
        #     ax.set_xticks([0.5, 1, 1.5])
        if (i==(nrows-1)):
            ax.set_xticks([0, 30, 60])
            plt.xlabel('Relative Angle $\phi$ ($^{\circ}$)')
        else:
            ax.set_xticks([])
        if (i==1):
            plt.ylabel('Histogram density', labelpad=11)
            # f2.ticks.hide_y()
            # f2.ticks.hide_x()

        if highrez[i]:
            phi = AddHeader((hro['phi']*u.rad).to(u.deg).value, Maps[i])
        else:
            phi = AddHeader((hro['phi']*u.rad).to(u.deg).value, HAWC[0])
        f2 = aplpy.FITSFigure(phi, figure=fig, subplot=(nrows,ncols,i*ncols+2))
        f2.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'white', linewidth=1.6) #linewidth=2.5)
        f2.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'black', linewidth=1.1) #linewidth=1.8)
        f2.show_colorscale(vmax=90, vmin=0, stretch='linear', cmap='Spectral')
        #f2.show_contour(Ncol, levels=[20, 50], colors='white', linewidth=1)
        f2.show_contour(Ncol, levels=[15, 50], colors='black', linewidth=1,  alpha=0.6)
        f2.axis_labels.hide_y()
        f2.tick_labels.hide_y()
        #f2.ticks.set_xspacing(0.05)
        if i==0:
            #f1.add_colorbar(location='top', width=0.1, pad=-0.07, axis_label_pad=6, log_format=True, axis_label_text='Intensity (Jy/pixel)')
            f2.add_colorbar(location='top', width=0.1, pad=-0.07, axis_label_pad=6, ticks=[0,30,60,90], axis_label_text='Relative Angle $\phi$ ($^{\circ}$)')
        
        f1 = aplpy.FITSFigure(mapProj, figure=fig, subplot=(nrows,ncols,i*ncols+3), constrained_layout=True)
        if interpolate:
            f1.show_colorscale(interpolation='nearest', cmap='BuPu')
        else:
            f1.show_colorscale(vmax=Vmax[i], vmin=Vmin[i], vmid=Vmid[i], cmap='BuPu', stretch=stretch[i])
        f1.show_contour(Ncol, levels=[15, 50], colors='black', linewidth=1, alpha=0.6)
        f1.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'white', linewidth=1.6) #linewidth=2.5)
        f1.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'black', linewidth=1.1) #linewidth=1.8)
        f1.axis_labels.set_yposition('right')
        f1.tick_labels.set_yposition('right')
        #f1.show_contour(Ncol, levels=[20, 50], colors='white', linewidth=0.5)
        #f1.ticks.set_tick_direction('in')#, log_format=True)
        # if i==0:
        #     f1.add_colorbar(box=boxes[i], axis_label_pad=3, axis_label_text='Jy/pixel', ticks=ticks[i], log)
        # else:
        f1.add_colorbar(box=boxes[i], ticks=ticks[i], log_format=log_format[i])#, axis_label_text='Jy/pixel')
        f1.ticks.set_yspacing(0.04)
        #f1.ticks.set_xspacing(0.06)
        # if i==0:
        #     f1.add_label(2, 2, 'Jy/pixel', horizontalalignment='center', size=11)
        #colorbar.set_box(box=, box_orientation='vertical')
        
        if (i!=1):
            f1.axis_labels.hide_y()
        else:
            f1.axis_labels.set_ypad(2.7)

        if (i!=(nrows-1)):
            f1.axis_labels.hide_x()
            f1.tick_labels.hide_x()
            f2.axis_labels.hide_x()
            f2.tick_labels.hide_x()

    #plt.tight_layout()
    plt.subplots_adjust(wspace=-0.07, hspace=-0.01)
    #fig.tight_layout()
    plt.savefig(plot_name, bbox_inches='tight')
    #plt.show()

def Results_ALMA(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, 
                title, highrez=[False, False, False, False], interpolate=False):
    nrows=4
    ncols=3
    panels=nrows*ncols

    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 9}

    x = 0.998; dx = 0.009 ; dy = 0.06
    boxes = [[x, 0.76, dx, dy], [x, 0.57, dx, dy], [x, 0.34, dx, dy], [x, 0.15, dx, dy]]
    fig = plt.figure(figsize=(7, 9.5))

    for i in range(len(location)):
        hro = np.load(prefix+location[i]+'HRO_results.npz', allow_pickle=True)
        #mapProj = projectMap(Maps[i], HAWC[0])
        mapProj = Maps[i]

        ax = plt.subplot(nrows,ncols,i*ncols+1)
        angle = np.linspace(0, 90, hro['histbins'])
        plt.plot(angle, hro['hist'], linewidth=2.5, color='black', linestyle='--')
        plt.xlim(0,90)
        plt.ylim(0, 1.7)
        asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
        print("aspect is ", asp)
        ax.set_aspect(asp)
        #ax.yaxis.set_label_position("right")
        #ax.yaxis.tick_right()
        ax.set_yticks([0.5, 1.0, 1.5])
        ax.text(10, 1.4, Map_labels[i])
        rainbow_fill_between(ax, angle, np.zeros(angle.shape), hro['hist'], alpha=0.6)
        ax.text(10, 1.2, 'Z$_{\mathrm{x}}$'+' = {0}'.format(np.round(hro['Zx_corr'], 1)))
        ax.text(10, 1.0, 'Z\'$_{\mathrm{x}}$'+' = {0}'.format(np.round(hro['Zx_unCorr'], 1)))
        # ax.fill_between(angle[(np.where(angle<30))],hro['hist'][(np.where(angle<30))], facecolor='red', alpha=0.5)
        # ax.fill_between(angle[(np.where(angle>60))],hro['hist'][(np.where(angle>60))], facecolor='blue', alpha=0.5)
        # plt.text(5, 0.10, 'Parallel', fontsize=7)
        # plt.text(60, 0.10, 'Perpendicular', fontsize=7)
        if i==0:
            ax.set_title(title, fontsize=14, pad=17)

        # if (i!=2):
        #     ax.set_xticks([0.5, 1, 1.5])
        
        ax.set_xticks([0, 30, 60, 90])
        plt.xlabel('Relative Angle $\phi$ ($^{\circ}$)')
        if i==1:
            plt.ylabel('Histogram density', labelpad=11)
            # f2.ticks.hide_y()
            # f2.ticks.hide_x()

        if highrez[i]:
            phi = AddHeader((hro['phi']*u.rad).to(u.deg).value, Maps[i])
        else:
            phi = AddHeader((hro['phi']*u.rad).to(u.deg).value, HAWC[0])
        f2 = aplpy.FITSFigure(phi, figure=fig, subplot=(nrows,ncols,i*ncols+2))
        #f2.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'white', linewidth=1.6) #linewidth=2.5)
        #f2.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'black', linewidth=1.1) #linewidth=1.8)
        f2.show_colorscale(vmax=90, vmin=0, stretch='linear', cmap='Spectral')
        #f2.show_contour(Ncol, levels=[20, 50], colors='white', linewidth=1)
        f2.show_contour(Ncol, levels=[15, 50], colors='black', linewidth=1,  alpha=0.6)
        f2.axis_labels.hide_y()
        f2.tick_labels.hide_y()

        if i==0:
            #f1.add_colorbar(location='top', width=0.1, pad=-0.07, axis_label_pad=6, log_format=True, axis_label_text='Intensity (Jy/pixel)')
            f2.add_colorbar(location='top', width=0.1, pad=-0.07, axis_label_pad=6, ticks=[0,30,60,90], axis_label_text='Relative Angle $\phi$ ($^{\circ}$)')
        
        f1 = aplpy.FITSFigure(mapProj, figure=fig, subplot=(nrows,ncols,i*ncols+3), constrained_layout=True)
        if interpolate:
            f1.show_colorscale(interpolation='nearest', cmap='BuPu')
        else:
            f1.show_colorscale(vmax=Vmax[i], vmin=Vmin[i], vmid=Vmid[i], cmap='BuPu', stretch=stretch[i])
        f1.show_contour(Ncol, levels=[15, 50], colors='black', linewidth=1, alpha=0.6)
        if Maps[i] == ALMA_12m:
            f1.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'white', linewidth=1) #linewidth=2.5)
            f1.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'black', linewidth=0.7) #linewidth=1.8)
            f2.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'white', linewidth=1) #linewidth=2.5)
            f2.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'black', linewidth=0.7) #linewidth=1.8)
        else:
            f1.show_vectors(vecMask, HAWC[11], step=8, scale=6, units = 'degrees', color = 'white', linewidth=1) #linewidth=2.5)
            f1.show_vectors(vecMask, HAWC[11], step=8, scale=6, units = 'degrees', color = 'black', linewidth=0.7) #linewidth=1.8)
            f2.show_vectors(vecMask, HAWC[11], step=8, scale=6, units = 'degrees', color = 'white', linewidth=1) #linewidth=2.5)
            f2.show_vectors(vecMask, HAWC[11], step=8, scale=6, units = 'degrees', color = 'black', linewidth=0.7) #linewidth=1.8)
        f1.axis_labels.set_yposition('right')
        f1.tick_labels.set_yposition('right')
        #f1.show_contour(Ncol, levels=[20, 50], colors='white', linewidth=0.5)
        #f1.ticks.set_tick_direction('in')#, log_format=True)
        # if i==0:
        #     f1.add_colorbar(box=boxes[i], axis_label_pad=3, axis_label_text='Jy/pixel', ticks=ticks[i], log)
        # else:
        f1.add_colorbar(box=boxes[i], ticks=ticks[i], log_format=log_format[i])#, axis_label_text='Jy/pixel')
        if Maps[i]==ALMA_ACA:
            f1.ticks.set_xspacing(0.05)
            f2.ticks.set_xspacing(0.05)
            f1.recenter(ra_center, dec_center, height=0.095, width=0.0855)
            f2.recenter(ra_center, dec_center, height=0.095, width=0.0855)
        else:
            f1.ticks.set_xspacing(0.03)
            f2.ticks.set_xspacing(0.03)
            f1.ticks.set_yspacing(0.01)
            f2.ticks.set_yspacing(0.01)
        #f1.ticks.set_yspacing(0.005)
        #f1.ticks.set_xspacing(0.06)
        # if i==0:
        #     f1.add_label(2, 2, 'Jy/pixel', horizontalalignment='center', size=11)
        #colorbar.set_box(box=, box_orientation='vertical')
        
        if (i!=1):
            f1.axis_labels.hide_y()
        else:
            f1.axis_labels.set_ypad(3.5)

        # if (i!=(nrows-1)):
        #     f1.axis_labels.hide_x()
        #     f1.tick_labels.hide_x()
        #     f2.axis_labels.hide_x()
        #     f2.tick_labels.hide_x()

    #plt.tight_layout()
    #plt.subplots_adjust(wspace=-0.35)
    #fig.tight_layout()
    plt.savefig(plot_name, bbox_inches='tight')
    #plt.show()

def Results_Flip5(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, 
                title, highrez=[False, False, False, False, False], interpolate=False):#
    nrows=5
    ncols=3
    panels=nrows*ncols

    x = 0.91; dx = 0.009 ; dy = 0.05
    boxes = [[x, 0.78, dx, dy], [x, 0.63, dx, dy], [x, 0.48, dx, dy], [x, 0.33, dx, dy], [x, 0.17, dx, dy]]
    fig = plt.figure(figsize=(7.5, 11))

    for i in range(len(location)):
        hro = np.load(prefix+location[i]+'HRO_results.npz', allow_pickle=True)
        mapProj = projectMap(Maps[i], HAWC[0])

        ax = plt.subplot(nrows,ncols,i*ncols+1)
        angle = np.linspace(0, 90, hro['histbins'])
        plt.plot(angle, hro['hist'], linewidth=2.5, color='black', linestyle='--')
        plt.xlim(0,90)
        plt.ylim(0, 1.7)
        asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
        print("aspect is ", asp)
        ax.set_aspect(asp)
        #ax.yaxis.set_label_position("right")
        #ax.yaxis.tick_right()
        ax.set_yticks([0.5, 1.0, 1.5])
        ax.text(10, 1.4, Map_labels[i])
        ax.text(10, 1.2, 'Z$_{\mathrm{x}}$'+' = {0}'.format(np.round(hro['Zx_corr'], 1)))
        ax.text(10, 1.0, 'Z\'$_{\mathrm{x}}$'+' = {0}'.format(np.round(hro['Zx_unCorr'], 1)))
        cmap = plt.get_cmap("Spectral")

        #ax.fill_between(angle[(np.where(angle<30))],hro['hist'][(np.where(angle<30))], facecolor=cmap(0), alpha=0.5)
        #ax.fill_between(angle[(np.where(angle>60))],hro['hist'][(np.where(angle>60))], facecolor=cmap(1), alpha=0.5)
        #plt.text(2, 0.10, 'Parallel', fontsize=8)
        #plt.text(60, 0.10, 'Perpendicular', fontsize=8)
        rainbow_fill_between(ax, angle, np.zeros(angle.shape), hro['hist'], alpha=0.6)
        # if i==(nrows-1):
        #     plt.text(2, 0.10, 'Parallel', fontsize=7)
        #     plt.text(60, 0.10, 'Perpendicular', fontsize=7)
        if i==0:
            ax.set_title(title, fontsize=14, pad=17)

        # if (i!=2):
        #     ax.set_xticks([0.5, 1, 1.5])
        if (i==4):
            ax.set_xticks([0, 30, 60])
            plt.xlabel('Relative Angle $\phi$ ($^{\circ}$)')
        else:
            ax.set_xticks([])
        if (i==2):
            plt.ylabel('Histogram density', labelpad=11)
            # f2.ticks.hide_y()
            # f2.ticks.hide_x()

        if highrez[i]:
            phi = AddHeader((hro['phi']*u.rad).to(u.deg).value, Maps[i])
        else:
            phi = AddHeader((hro['phi']*u.rad).to(u.deg).value, HAWC[0])
        f2 = aplpy.FITSFigure(phi, figure=fig, subplot=(nrows,ncols,i*ncols+2))
        f2.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'white', linewidth=1.6) #linewidth=2.5)
        f2.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'black', linewidth=1.1) #linewidth=1.8)
        f2.show_colorscale(vmax=90, vmin=0, stretch='linear', cmap='Spectral')
        #f2.show_contour(Ncol, levels=[20, 50], colors='white', linewidth=1)
        f2.show_contour(Ncol, levels=[15, 50], colors='black', linewidth=1,  alpha=0.6)
        f2.axis_labels.hide_y()
        f2.tick_labels.hide_y()
        #f2.ticks.set_xspacing(0.05)
        if i==0:
            #f1.add_colorbar(location='top', width=0.1, pad=-0.07, axis_label_pad=6, log_format=True, axis_label_text='Intensity (Jy/pixel)')
            f2.add_colorbar(location='top', width=0.1, pad=-0.07, axis_label_pad=6, ticks=[0,30,60,90], axis_label_text='Relative Angle $\phi$ ($^{\circ}$)')
        
        f1 = aplpy.FITSFigure(mapProj, figure=fig, subplot=(nrows,ncols,i*ncols+3), constrained_layout=True)
        if interpolate:
            f1.show_colorscale(interpolation='nearest', cmap='BuPu')
        else:
            f1.show_colorscale(vmax=Vmax[i], vmin=Vmin[i], vmid=Vmid[i], cmap='BuPu', stretch=stretch[i])
        f1.show_contour(Ncol, levels=[15, 50], colors='black', linewidth=1, alpha=0.6)
        f1.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'white', linewidth=1.6) #linewidth=2.5)
        f1.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'black', linewidth=1.1) #linewidth=1.8)
        f1.axis_labels.set_yposition('right')
        f1.tick_labels.set_yposition('right')
        #f1.show_contour(Ncol, levels=[20, 50], colors='white', linewidth=0.5)
        #f1.ticks.set_tick_direction('in')#, log_format=True)
        # if i==0:
        #     f1.add_colorbar(box=boxes[i], axis_label_pad=3, axis_label_text='Jy/pixel', ticks=ticks[i], log)
        # else:
        f1.add_colorbar(box=boxes[i], ticks=ticks[i], log_format=log_format[i])#, axis_label_text='Jy/pixel')
        f1.ticks.set_yspacing(0.045)
        #f1.ticks.set_xspacing(0.06)
        # if i==0:
        #     f1.add_label(2, 2, 'Jy/pixel', horizontalalignment='center', size=11)
        #colorbar.set_box(box=, box_orientation='vertical')
        
        if (i!=2):
            f1.axis_labels.hide_y()
        else:
            f1.axis_labels.set_ypad(3.3)

        if (i!=4):
            f1.axis_labels.hide_x()
            f1.tick_labels.hide_x()
            f2.axis_labels.hide_x()
            f2.tick_labels.hide_x()

    #plt.tight_layout()
    plt.subplots_adjust(wspace=-0.33, hspace=-0.01)
    #fig.tight_layout()
    plt.savefig(plot_name, bbox_inches='tight')
    #plt.show()

def Results_Flip(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title, highrez=[False, False, False]):
    nrows=3
    ncols=3
    panels=nrows*ncols

    x = 0.95; dx = 0.009 ; dy = 0.1
    boxes = [[x, 0.69, dx, dy], [x, 0.432, dx, dy], [x, 0.17, dx, dy]]
    fig = plt.figure(figsize=(7, 7))

    for i in range(len(location)):
        hro = np.load(prefix+location[i]+'HRO_results.npz', allow_pickle=True)
        mapProj = projectMap(Maps[i], HAWC[0])

        ax = plt.subplot(nrows,ncols,i*nrows+1)
        angle = np.linspace(0, 90, hro['histbins'])
        plt.plot(angle, hro['hist'], linewidth=1.4, color='black')
        plt.xlim(0,90)
        plt.ylim(0, 1.7)
        asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
        print("aspect is ", asp)
        ax.set_aspect(asp)
        #ax.yaxis.set_label_position("right")
        #ax.yaxis.tick_right()
        ax.set_yticks([0.5, 1.0, 1.5])
        ax.text(10, 1.4, Map_labels[i])
        ax.text(10, 1.2, 'Z$_{\mathrm{x}}$'+' = {0}'.format(np.round(hro['Zx_corr'], 1)))
        ax.text(10, 1.0, 'Z$_{\mathrm{x}}$\''+' = {0}'.format(np.round(hro['Zx_unCorr'], 1)))
        if i==0:
            ax.set_title(title, fontsize=13, pad=20)

        # if (i!=2):
        #     ax.set_xticks([0.5, 1, 1.5])
        if (i==2):
            ax.set_xticks([0, 30, 60])
            plt.xlabel('Relative Angle $\phi$ ($^{\circ}$)')
        else:
            ax.set_xticks([])
        if (i==1):
            plt.ylabel('Histogram density', labelpad=5)
            # f2.ticks.hide_y()
            # f2.ticks.hide_x()

        if highrez[i]:
            phi = AddHeader((hro['phi']*u.rad).to(u.deg).value, Maps[i])
        else:
            phi = AddHeader((hro['phi']*u.rad).to(u.deg).value, HAWC[0])
        f2 = aplpy.FITSFigure(phi, figure=fig, subplot=(nrows,ncols,i*nrows+2))
        f2.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'white', linewidth=1.6) #linewidth=2.5)
        f2.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'black', linewidth=1.1) #linewidth=1.8)
        f2.show_colorscale(vmax=90, vmin=0, stretch='linear', cmap='Spectral')
        #f2.show_contour(Ncol, levels=[20, 50], colors='white', linewidth=1)
        f2.show_contour(Ncol, levels=[20, 50], colors='black', linewidth=1,  alpha=0.6)
        f2.axis_labels.hide_y()
        f2.tick_labels.hide_y()
        #f2.ticks.set_xspacing(0.05)
        if i==0:
            #f1.add_colorbar(location='top', width=0.1, pad=-0.07, axis_label_pad=6, log_format=True, axis_label_text='Intensity (Jy/pixel)')
            f2.add_colorbar(location='top', width=0.1, pad=-0.07, axis_label_pad=6, ticks=[0,30,60,90], axis_label_text='Relative Angle $\phi$ ($^{\circ}$)')
        
        f1 = aplpy.FITSFigure(mapProj, figure=fig, subplot=(nrows,ncols,i*nrows+3), constrained_layout=True)
        
        f1.show_colorscale(vmax=Vmax[i], vmin=Vmin[i], vmid=Vmid[i], cmap='BuPu', stretch=stretch[i])
        f1.show_contour(Ncol, levels=[20, 50], colors='black', linewidth=1, alpha=0.6)
        f1.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'white', linewidth=1.6) #linewidth=2.5)
        f1.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'black', linewidth=1.1) #linewidth=1.8)
        f1.axis_labels.set_yposition('right')
        f1.tick_labels.set_yposition('right')
        #f1.show_contour(Ncol, levels=[20, 50], colors='white', linewidth=0.5)
        #f1.ticks.set_tick_direction('in')#, log_format=True)
        # if i==0:
        #     f1.add_colorbar(box=boxes[i], axis_label_pad=3, axis_label_text='Jy/pixel', ticks=ticks[i], log)
        # else:
        f1.add_colorbar(box=boxes[i], ticks=ticks[i], log_format=log_format[i])#, axis_label_text='Jy/pixel')
        f1.ticks.set_yspacing(0.04)
        #f1.ticks.set_xspacing(0.06)
        # if i==0:
        #     f1.add_label(2, 2, 'Jy/pixel', horizontalalignment='center', size=11)
        #colorbar.set_box(box=, box_orientation='vertical')
        
        if (i!=1):
            f1.axis_labels.hide_y()
        else:
            f1.axis_labels.set_ypad(2.7)

        if (i!=2):
            f1.axis_labels.hide_x()
            f1.tick_labels.hide_x()
            f2.axis_labels.hide_x()
            f2.tick_labels.hide_x()

    #plt.tight_layout()
    plt.subplots_adjust(wspace=-0.07, hspace=-0.01)
    #fig.tight_layout()
    plt.savefig(plot_name, bbox_inches='tight')
    #plt.show()


def Results_Fig(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title, highrez=[False, False, False]):
    nrows=3
    ncols=3
    panels=nrows*ncols

    x = 0.047; dx = 0.009 ; dy = 0.1
    boxes = [[x, 0.69, dx, dy], [x, 0.432, dx, dy], [x, 0.17, dx, dy]]
    fig = plt.figure(figsize=(7, 7))

    for i in range(len(location)):
        hro = np.load(prefix+location[i]+'HRO_results.npz', allow_pickle=True)
        # if Maps[i]== spitzer_CH1 and Maps[i] == spitzer_CH2:
        #     mapProj = Maps[i]
        # else:
        mapProj = projectMap(Maps[i], HAWC[0])
        
        f1 = aplpy.FITSFigure(mapProj, figure=fig, subplot=(nrows,ncols,i*nrows+1))
        
        f1.show_colorscale(vmax=Vmax[i], vmin=Vmin[i], vmid=Vmid[i], cmap='BuPu', stretch=stretch[i])
        f1.show_contour(Ncol, levels=[20, 50], colors='black', linewidth=1, alpha=0.6)
        f1.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'white', linewidth=1.6) #linewidth=2.5)
        f1.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'black', linewidth=1.1) #linewidth=1.8)
        if i==0:
            f1.set_title(title, pad=20, fontsize=13, horizontalalignment='center')
        #f1.show_contour(Ncol, levels=[20, 50], colors='white', linewidth=0.5)
        #f1.ticks.set_tick_direction('in')#, log_format=True)
        # if i==0:
        #     f1.add_colorbar(box=boxes[i], axis_label_pad=3, axis_label_text='Jy/pixel', ticks=ticks[i], log)
        # else:
        f1.add_colorbar(box=boxes[i], ticks=ticks[i], log_format=log_format[i])#, axis_label_text='Jy/pixel')
        f1.ticks.set_yspacing(0.04)
        # if i==0:
        #     f1.add_label(2, 2, 'Jy/pixel', horizontalalignment='center', size=11)
        #colorbar.set_box(box=, box_orientation='vertical')
        
        if highrez[i]:
            phi = AddHeader((hro['phi']*u.rad).to(u.deg).value, Maps[i])
        else:
            phi = AddHeader((hro['phi']*u.rad).to(u.deg).value, HAWC[0])
        f2 = aplpy.FITSFigure(phi, figure=fig, subplot=(nrows,ncols,i*nrows+2))
        f2.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'white', linewidth=1.6) #linewidth=2.5)
        f2.show_vectors(vecMask, HAWC[11], step=6, scale=4, units = 'degrees', color = 'black', linewidth=1.1) #linewidth=1.8)
        f2.show_colorscale(vmax=90, vmin=0, stretch='linear', cmap='Spectral')
        #f2.show_contour(Ncol, levels=[20, 50], colors='white', linewidth=1)
        f2.show_contour(Ncol, levels=[20, 50], colors='black', linewidth=1,  alpha=0.6)
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
        ax.text(10, 1.0, 'Z$_{\mathrm{x}}$\''+' = {0}'.format(np.round(hro['Zx_unCorr'], 1)))
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
    plt.savefig(plot_name)
    #plt.show()


def rect(ax, x, y, w, h, c,**kwargs):
    #Varying only in x
    if len(c.shape) is 1:
        rect = plt.Rectangle((x, y), w, h, color=c, ec=c,**kwargs)
        ax.add_patch(rect)
    #Varying in x and y
    else:
        #Split into a number of bins
        N = c.shape[0]
        hb = h/float(N); yl = y
        for i in range(N):
            yl += hb
            rect = plt.Rectangle((x, yl), w, hb, 
                                 color=c[i,:], ec=c[i,:],**kwargs)
            ax.add_patch(rect)

def rainbow_fill_between(ax, X, Y1, Y2, colors=None, 
                         cmap=plt.get_cmap("Spectral"),**kwargs):
    plt.plot(X,Y1,lw=0)  # Plot so the axes scale correctly

    dx = (X[1]-X[0])
    N  = X.size

    #Pad a float or int to same size as x
    if (type(Y2) is float or type(Y2) is int):
        Y2 = np.array([Y2]*N)

    #No colors -- specify linear
    if colors is None:
        colors = []
        for n in range(N):
            colors.append(cmap(n/float(N)))
    #Varying only in x
    elif len(colors.shape) is 1:
        colors = cmap((colors-colors.min())
                      /(colors.max()-colors.min()))
    #Varying only in x and y
    else:
        cnp = np.array(colors)
        colors = np.empty([colors.shape[0],colors.shape[1],4])
        for i in range(colors.shape[0]):
            for j in range(colors.shape[1]):
                colors[i,j,:] = cmap((cnp[i,j]-cnp[:,:].min())
                                    /(cnp[:,:].max()-cnp[:,:].min()))

    colors = np.array(colors)

    #Create the patch objects
    for (color,x,y1,y2) in zip(colors,X,Y1,Y2):
        rect(ax,x,y2,dx,y1-y2,color,**kwargs)

# dust maps
prefix = '/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/FWHM_3/'
Maps = [Hersh_500, Hersh_250, Hersh_70]
stretch = ['log', 'log', 'log']
log_format = [True, True, True]
location = ['BandC/Hersh_500/',  'BandC/Hersh_160/', 'BandC/Hersh_70/']
ticks = [[1e2,1e3], [0,1e4], [0, 1e1]]
Vmax = [2800, 16765, 12]
Vmin = [80, -244, -0.255]
Vmid = [10, -3978, -2.6]
Map_labels = ['Herschel 500 $\mu$m', 'Herschel 250 $\mu$m', 'Herschel 70 $\mu$m']
Ncol_HAWC = projectMap(Ncol, HAWC[0])
plot_name = 'Results_wavelength.pdf'
title='Dust Maps'
#Results_Fig(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title)
plot_name = 'Results_DustFlip.pdf'
#Results_Flip(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title)

# gas maps
Maps = [CO12, CO13,  C18O]
stretch = ['linear', 'linear', 'linear']
log_format = [False, False, False]
location = ['BandC/12CO/',  'BandC/13CO/', 'BandC/Mopra_C18O/']
ticks = [[0,150], [0,80], [0, 5000]]
Vmax = [180, 100, 6000]
Vmin = [-5, -5, -100]
Vmid = [None, None, None]
Map_labels = ['APEX $^{12}$CO', 'APEX $^{13}$CO', 'Mopra C$^{18}$O']
plot_name = 'Results_Gas3.pdf'
title='Gas Tracers'
#Results_Fig(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title)
plot_name = 'Results_GasFlip.pdf'
#Results_Flip(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title)

# gas maps
Maps = [CO12, CO13,  C18O, N2H]
stretch = ['linear', 'linear', 'linear', 'linear']
log_format = [False, False, True, False]
location = ['BandC/12CO/',  'BandC/13CO/', 'BandC/Mopra_C18O/', 'BandC/Mopra_N2H/']
ticks = [[0,150], [0,60], [0, 5000], [0,6000]]
Vmax = [180, 80, 6000, 6000]
Vmin = [-5, -5, -100, -100]
Vmid = [None, None, None, None]
Map_labels = ['APEX $^{12}$CO', 'APEX $^{13}$CO', 'Mopra C$^{18}$O', 'Mopra N$_{2}$H$^{+}$']
plot_name = 'Results_Gas4.pdf'
title='Gas Tracers'
#Results_Fig(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title)
plot_name = 'Results_GasFlip4.pdf'
#Results_Flip4(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title)

# pdr
Maps = [spitzer_CH2, spitzer_CH1, CII, OI]
stretch = ['linear', 'linear', 'linear', 'linear']
log_format = [False, False, False, True]
ticks = [[0, 60], [0, 60], [0,400], [0, 1e5]]
location = ['BandC/Spitz_CH2/', 'BandC/Spitz_CH1/',  'BandC/CII/', 'BandC/OI/']
Vmax = [80, 80, 500, 100000]
Vmin = [-4, -4, 0, -20000]
Vmid = [-10, -10, -3978, -2.6]
Map_labels = ['Spitzer 4.5 $\mu$m', 'Spitzer 3.6 $\mu$m', 'SOFIA [CII]', 'SOFIA [OI]']
plot_name = 'Results_PDR.pdf'
title='PDR Tracers'
#Results_Fig(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title, highrez=[True, False, False])
plot_name = 'Results_PDR4.pdf'
#Results_Flip(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title, highrez=[True, False, False])
#Results_Flip4(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title, highrez=[True, True, False, False])

Maps = [Hersh_500, Hersh_350, Hersh_250, Hersh_160, Hersh_70]
stretch = ['log', 'log', 'log', 'log', 'log']
log_format = [True, True, True, True, True]
location = ['BandC/Hersh_500/', 'BandC/Hersh_350/', 'BandC/Hersh_250/', 'BandC/Hersh_160/', 'BandC/Hersh_70/']
ticks = [[1e2,1e3], [0,1e3], [0,1e4], [0,1e1], [0, 1e1]]
Vmax = [2800, 8000, 16765,  50, 12]
Vmin = [80, -100, -244, -1, -0.255]
Vmid = [10, -1000, -3978, -5, -2.6]
Map_labels = ['Herschel 500 $\mu$m', 'Herschel 350 $\mu$m', 'Herschel 250 $\mu$m', 'Herschel 160 $\mu$m', 'Herschel 70 $\mu$m']
Ncol_HAWC = projectMap(Ncol, HAWC[0])
plot_name = 'Results_Dust5pdf'
title='Dust Maps'
#Results_Fig(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title)
plot_name = 'Results_Herschel5.pdf'
#Results_Flip5(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title)

Maps = [Hersh_500, Hersh_350, Hersh_250, Hersh_160, Hersh_70]
stretch = ['log', 'log', 'log', 'log', 'log']
log_format = [True, True, True, True, True]
location = ['BandC/Hersh_500/', 'BandC/Hersh_350/', 'BandC/Hersh_250/', 'BandC/Hersh_160/', 'BandC/Hersh_70/']
ticks = [[1e2,1e3], [0,1e3], [0,1e4], [0,1e1], [0, 1e1]]
Vmax = [2800, 8000, 16765,  50, 12]
Vmin = [80, -100, -244, -1, -0.255]
Vmid = [10, -1000, -3978, -5, -2.6]
Map_labels = ['Herschel 500 $\mu$m', 'Herschel 350 $\mu$m', 'Herschel 250 $\mu$m', 'Herschel 160 $\mu$m', 'Herschel 70 $\mu$m']
Ncol_HAWC = projectMap(Ncol, HAWC[0])
plot_name = 'Results_Dust5pdf'
title='Herschel Maps'
#Results_Fig(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title)
plot_name = 'Results_Herschel5.pdf'
#Results_Flip5(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title)

Maps = [CO12, CO13, C18O, HNC,  N2H]
stretch = ['linear', 'linear', 'linear', 'linear', 'linear']
log_format = [False, False, False, False, False]
location = ['BandC/12CO/', 'BandC/13CO/', 'BandC/Mopra_C18O/', 'BandC/Mopra_HNC/', 'BandC/Mopra_N2H/']
ticks = [[0, 150], [0,80], [0,4000], [0,6000], [0, 5000]]
Vmax = [180, 100, 5000, 7400, 6000]
Vmin = [-10, -5, -10, -5, -1000]
Vmid = [None, None, None, None, None]
Map_labels = ['APEX $^{12}$CO', 'APEX $^{13}$CO', 'Mopra C$^{18}$O', 'Mopra HNC', 'Mopra N$_{2}$H$^{+}$']
title='Gas Tracers'
plot_name = 'Results_Gas5.pdf'
#Results_Fig(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title)
Results_Flip5(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title)

Maps = [Hersh_500, Hersh_350, Hersh_250, HAWE]
stretch = None
log_format = [False, False, True, False, False]
Map_labels = ['Herschel 500 $\mu$m', 'Herschel 350 $\mu$m', 'Herschel 250 $\mu$m', 'SOFIA 214 $\mu$m']
ticks = [[1e2,1e3], [0,1e3], [0,1e4], [0,1e2]]
# Vmax = [2800, 8000, 16765,  ]
# Vmin = [80, -100, -244, ]
# Vmid = [10, -1000, -3978, ]
location = ['BandC/Hersh_500/', 'BandC/Hersh_350/', 'BandC/Hersh_250/', 'BandC/BandE_I/']
Ncol_HAWC = projectMap(Ncol, HAWC[0])
plot_name = 'Results_All1.pdf'
title='Dust Maps'
#Results_Fig(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title)
#Results_Flip4(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title, interpolate=True)

Maps = []
stretch = None
log_format = [False, False, True, False, False]
Map_labels = ['Herschel 500 $\mu$m', 'Herschel 350 $\mu$m', 'Herschel 250 $\mu$m', 'SOFIA 214 $\mu$m']
ticks = [[1e2,1e3], [0,1e3], [0,1e4], [0,1e2]]
# Vmax = [2800, 8000, 16765,  ]
# Vmin = [80, -100, -244, ]
# Vmid = [10, -1000, -3978, ]
location = ['BandC/Hersh_500/', 'BandC/Hersh_350/', 'BandC/Hersh_250/', 'BandC/BandE_I/']
Ncol_HAWC = projectMap(Ncol, HAWC[0])
plot_name = 'Results_All1.pdf'
title='All Maps'
#Results_Fig(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title)
#Results_Flip4(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title, interpolate=True)

Maps = [Hersh_500, Hersh_250, HAWC[0], Hersh_70]
stretch = ['log', 'log', 'linear', 'log']
log_format = [True, True, False, True]
location = ['BandC/Hersh_500/',  'BandC/Hersh_250/', 'BandC/BandC_I/', 'BandC/Hersh_70/']
Map_labels = ['Herschel 500 $\mu$m', 'Herschel 250 $\mu$m', 'SOFIA 89 $\mu$m', 'Herschel 70 $\mu$m']
ticks = [[1e2,1e3], [0,1e4], [0,4], [0, 1e1]]
Vmax = [2800, 16765, 5, 15]
Vmin = [80, -244, -0.5, -1]
Vmid = [10, -3978, -5, -4]
plot_name = 'Results_Dust4.pdf'
title='Dust Maps'
prefix = '/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/FWHM_3/'
#Results_Flip4(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title)

Maps = [Hersh_350, HAWE[0], Ncol, Temp]
stretch = ['log', 'log', 'linear', 'linear']
log_format = [False, False, False, False]
location = ['BandC/Hersh_350/',  'BandC/BandE_I/', 'BandC/Ncol_Av/', 'BandC/Temp_High/']
Map_labels = ['Herschel 350 $\mu$m', 'SOFIA 214 $\mu$m', 'Column Density', 'Temperature']
ticks = [[500,7000], [0,10], [0,70], [20, 30]]
Vmax = [10000, 20, 80, 32]
Vmin = [300, -1, -6, 20]
Vmid = [-10, -5, -8, 0]
plot_name = 'Results_Appendix4.pdf'
title=''
prefix = '/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/FWHM_3/'
#Results_Flip4(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title)

Maps = [Hersh_350, HAWE[0], Hersh_160, Ncol, Temp]
stretch = ['log', 'log', 'log', 'linear', 'linear']
log_format = [False, False, False, False, False]
location = ['BandC/Hersh_350/',  'BandC/BandE_I/', 'BandC/Hersh_160/',   'BandC/Ncol_Av/', 'BandC/Temp_High/']
Map_labels = ['Herschel 350 $\mu$m', 'SOFIA 214 $\mu$m', 'Hershcel 160 $\mu$m', 'Column Density', 'Temperature']
ticks = [[500,7000], [0,10], [0,30], [0,70], [20, 30]]
Vmax = [10000, 20, 40, 80, 32]
Vmin = [300, -1, -3, -6, 20]
Vmid = [-10, -5, -15, -8, 0]
plot_name = 'Results_Appendix5.pdf'
title=''
prefix = '/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/FWHM_3/'
Results_Flip5(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title)

Maps = [ALMA_ACA, ALMA_12m]
stretch = ['log', 'log']
log_format = [False, False]
location = ['BandC/ALMA_ACA/',  'BandC/ALMA_12m/']
Map_labels = ['ALMA ACA', 'ALMA 12m']
ticks = [[0,0.1], [0,0.01]]
Vmax = [0.11, 0.011]
Vmin = [-0.017, -0.0017]
Vmid = [-0.5, -0.05]
plot_name = 'Results_ALMA.pdf'
title='ALMA Maps'
prefix = '/Users/akankshabij/Documents/MSc/Research/Code/scripts/HRO/HRO_BijA/Output_Plots/Paper/FWHM_3/'
#Results_ALMA(prefix, location, Maps, Map_labels, Vmax, Vmin, Vmid, stretch, log_format, plot_name, ticks, title, interpolate=False, highrez=[True, True])