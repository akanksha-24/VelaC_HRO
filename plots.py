import numpy as np
import aplpy
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
from astropy.io import fits
from reproject import reproject_interp, reproject_exact
from matplotlib import cm
#import run as r

Hersch = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits')[0]

def projectMap(mapOrigin, ref):
    '''This function projects a given map 'mapOrigin' onto the same WCS coordinates as a reference map and returns the map in the same shape'''
    proj, footprint = reproject_exact(mapOrigin, ref.header)
    proj[np.isnan(ref.data)] = np.nan
    proj[0].data = proj
    return proj

def plot_Fields(hro, prefix='', Bfield=True, Efield=False, step=12, scale=8):
    '''This function plots the B-field and E-field, used for debugging purposes to make sure the result is as expected'''
    
    # Use fits header information to get maps in WCS coordinates
    Map = hro.hdu.copy()
    Map.data = hro.Map
    B_hdr = hro.hdu.copy()                                 
    B_hdr.data = hro.Bfield
    E_hdr = hro.hdu.copy()
    E_hdr.data = hro.Efield
    #E_hdr.data = hro.contour

    plt.close('all')
    fig = plt.figure(figsize=[5,4], dpi=300)
    fxx = aplpy.FITSFigure(Map, figure=fig, slices=[0])
    fxx.show_colorscale(cmap='binary', interpolation='nearest')#, vmin=0.5e22, vmax=4e22)
    fxx.add_colorbar()

    if Bfield==True:
        fxx.show_vectors(hro.vecMask, B_hdr, step=step, scale=scale, units = 'radians', color = 'blue', linewidth=2)     # plot B-field vectors 
        fxx.set_title('Magnetic Field')
        plt.savefig(prefix+'BField_ForTalk.png')
    if Efield==True:
        plt.close('all')
        fig = plt.figure(figsize=[5,4], dpi=300)
        fxx = aplpy.FITSFigure(Map, figure=fig, slices=[0])
        fxx.show_colorscale(cmap='binary', interpolation='nearest')
        fxx.add_colorbar()
        fxx.show_vectors(hro.vecMask, E_hdr, step=step, scale=scale, units='radians', color = 'red', linewidth=2)        # plot Polarization angle 
        fxx.set_title('Polarization Angle')
        plt.savefig(prefix+'Efield.png')
    # if Efield==True and Bfield==True:
    #     fxx.show_vectors(hro.vecMask, B_hdr, step=step, scale=scale, units = 'radians', color = 'blue', linewidth=1, label='Magnetic Field')     # Plot both, expect 90 deg offset sanity check
    #     fxx.show_vectors(hro.vecMask, E_hdr, step=step, scale=scale, units='radians', color = 'red', linewidth=1, label='Polarization Angle') 
    #     fxx.set_title('Comparing Magnetic Field and Polarization Angle')  
    #     plt.legend()  
    #     plt.savefig(prefix+'BandEfield.png')

def plot_Map(hro, prefix='', norm=False):
    '''hro is an HRO class object as defined in HRO.py'''

    Map = hro.hdu.copy()
    Map.data = hro.unmasked_Map

    Map_smooth = hro.hdu.copy()
    Map_smooth.data = hro.Map

    plt.close('all')
    fig = plt.figure(figsize=[5,4], dpi=300)
    fxx = aplpy.FITSFigure(Map, figure=fig, slices=[0])
    fxx.show_colorscale(interpolation='nearest', cmap='Spectral_r')
    #fxx.show_contour(Hersch, levels=[20,50,80], colors='white')
    fxx.add_colorbar()
    plt.savefig(prefix+'Map.png')

    # plt.close('all')
    # fig = plt.figure(figsize=[5,4], dpi=300)
    # fxx = aplpy.FITSFigure(Map, figure=fig, slices=[0])
    # fxx.show_colorscale(interpolation='nearest', cmap='Spectral_r')
    # #fxx.show_contour(Hersch, levels=[20,50,80], colors='white')
    # fxx.add_colorbar()
    # plt.savefig(prefix+'Map_Smooth.png')


def plot_Gradient(hro, prefix='', norm=False):
    '''hro is an HRO class object as defined in HRO.py'''

    gradx = hro.hdu.copy()
    grady = hro.hdu.copy()
    if norm==False:
        gradx.data = hro.dMdx
        grady.data = hro.dMdy
    else:
        gradx.data = hro.dMdx / hro.mag
        grady.data = hro.dMdy / hro.mag

    plt.close('all')
    fig = plt.figure(figsize=[5,4], dpi=300)
    fxx = aplpy.FITSFigure(gradx, figure=fig, slices=[0])
    fxx.show_colorscale(interpolation='nearest', cmap='binary')
    fxx.add_colorbar()
    if norm==False:
        fxx.set_title('Partial Derivative in x direction, dI/dx') 
        plt.savefig(prefix+'gradx.png')
    else:
        fxx.set_title('Normalized Partial Derivative in x direction, dI/dx')
        plt.savefig(prefix+'gradx_norm.png')

    fig = plt.figure(figsize=[5,4], dpi=300)
    fxx = aplpy.FITSFigure(grady, figure=fig, slices=[0])
    fxx.show_colorscale(interpolation='nearest', cmap='binary')
    fxx.add_colorbar()
    if norm==False:
        fxx.set_title('Partial Derivative in y direction, dI/dy') 
        plt.savefig(prefix+'grady.png')
    else:
        fxx.set_title('Normalized Partial Derivative in y direction, dI/dy')
        plt.savefig(prefix+'grady_norm.png')

def plot_GradientAmp(hro, prefix='', norm=False):
    '''hro is an HRO class object as defined in HRO.py'''

    gradAmp = hro.hdu.copy()
    gradAmp.data = hro.mag

    plt.close('all')
    fig = plt.figure(figsize=[5,4], dpi=300)
    fxx = aplpy.FITSFigure(gradAmp, figure=fig, slices=[0])
    fxx.show_colorscale(interpolation='nearest', cmap='Spectral')
    fxx.add_colorbar()
    fxx.set_title('Gradient Amplitude') 
    plt.savefig(prefix+'gradAmp.png')

def plot_vectors(hro, prefix='', step=20, scale=10):
    ''''''
    Map = hro.hdu.copy()
    Map.data = hro.Map
    Bfield = hro.hdu.copy()
    Bfield.data = hro.Bfield
    cont = hro.hdu.copy()
    cont.data = hro.contour

    plt.close('all')
    fig = plt.figure(figsize=[5,4], dpi=300)
    fxx = aplpy.FITSFigure(Map, figure=fig, slices=[0])
    fxx.show_colorscale(interpolation='nearest', cmap='binary')
    #fxx.show_vectors(hro.vecMask, Bfield, step=step, scale=scale, units = 'radians', color = 'white', linewidth=2)
    #fxx.show_vectors(hro.vecMask, cont, step=step, scale=scale, units='radians', color = 'white', linewidth=2)
    fxx.show_vectors(hro.vecMask, Bfield, step=step, scale=scale, units = 'radians', color = 'blue', linewidth=1.7, label='Magnetic Field')
    fxx.show_vectors(hro.vecMask, cont, step=step, scale=scale, units='radians', color = 'red', linewidth=1.7, label='Contour Orientation')
    plt.legend()
    fxx.add_colorbar()
    #fxx.set_title('Comparing the contour and magnetic field orientation')
    plt.savefig(prefix+'relative_vectors.png')

def plot_phi(hro, prefix='', label='', norm=False, step=12, scale=8):
    '''hro is an HRO class object as defined in HRO.py'''
    Map = hro.vecMask.copy()
    Map.data = hro.unmasked_Map

    phi = hro.vecMask.copy()
    phi.data = (hro.phi * u.rad).to(u.deg)

    Bfield = hro.vecMask.copy()
    Bfield.data = hro.Bfield

    #Hersch = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/Herschel/HighresNmap_herschel_velac_high_av.fits')[0]

    #HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')[0]
    #phi_C = HAWC.copy()
    #phi_C.data = projectMap(phi, HAWC)
    # Map_C = HAWC.copy()
    # Map_C.data = r.projectMap(Map, HAWC)

    plt.close('all')
    fig = plt.figure(figsize=[5,4], dpi=300)
    fxx = aplpy.FITSFigure(phi, figure=fig, slices=[0], figsize=[12,8])
    fxx.show_colorscale(vmin=0, vmax=90, cmap='Spectral')
    fxx.show_vectors(hro.vecMask, Bfield, step=step, scale=scale, units = 'radians', color = 'white', linewidth=2.5)
    fxx.show_vectors(hro.vecMask, Bfield, step=step, scale=scale, units = 'radians', color = 'black', linewidth=1.8)
    #fxx.show_contour(Hersch, levels=[20,50,80], colors='black')
    #fxx.show_vectors(hro.hdu, hro.Bfield2, step=step, scale=scale, units = 'radians', color = 'black', linewidth=1)
    #fxx.show_contour(Map, levels=hro.sections[-3:-1], cmap='plasma')
    # print(np.nanmin(Map.data) + np.nanstd(Map.data))
    # print(2*np.nanmean(Map.data))
    # print(np.nanmax(Map.data) - 3*np.nanstd(Map.data))
    # fxx.show_contour(Map, levels=[np.nanmin(Map.data) + np.nanstd(Map.data), np.nanmean(Map.data)*2, 3*np.nanmean(Map.data), np.nanmax(Map.data) - 2*np.nanstd(Map.data)], colors='Black')
    fxx.add_colorbar()
    #fxx.set_title('Relative Angle, $\phi$ ' + label)
    fxx.set_title('Relative Angle') #(PRS: Zx={0})'.format(int(hro.Zx)))
    #plt.legend(fontsize=12)
    plt.savefig(prefix+'phi.png')

def plot_regions(hro, prefix='', norm=False, step=12, scale=8):
    '''hro is an HRO class object as defined in HRO.py'''
    digitized = hro.hdu.copy()
    digitized.data = hro.digitized
    #HAWC = fits.open('/Users/akankshabij/Documents/MSc/Research/Data/HAWC/Rereduced_2018-07-14_HA_F487_004-065_POL_70060957_HAWC_HWPC_PMP.fits')
    #digitized.data[np.isnan(hro.Map)]=np.nan

    phi = hro.hdu.copy()
    phi.data = (hro.phi * u.rad).to(u.deg)

    plt.close('all')
    fig = plt.figure(figsize=[5,4], dpi=300)
    fxx = aplpy.FITSFigure(digitized, figure=fig, slices=[0], figsize=[12,8])
    fxx.show_colorscale(cmap='Spectral', vmax=5, vmin=0)
    fxx.show_vectors(hro.vecMask, hro.Bfield, step=step, scale=scale, units = 'radians', color = 'black', linewidth=2)
    fxx.show_vectors(hro.vecMask, hro.Bfield, step=step, scale=scale, units = 'radians', color = 'white', linewidth=1)

    #fxx.show_vectors(hro.hdu, hro.Bfield2, step=step, scale=scale, units = 'radians', color = 'black', linewidth=1)
    #fxx.show_contour(Map, levels=hro.sections[-3:-1], cmap='plasma')
    # print(np.nanmin(Map.data) + np.nanstd(Map.data))
    # print(2*np.nanmean(Map.data))
    # print(np.nanmax(Map.data) - 3*np.nanstd(Map.data))
    # fxx.show_contour(Map, levels=[np.nanmin(Map.data) + np.nanstd(Map.data), np.nanmean(Map.data)*2, 3*np.nanmean(Map.data), np.nanmax(Map.data) - 2*np.nanstd(Map.data)], colors='Black')
    fxx.add_colorbar()
    fxx.set_title('Relative Angle, $\phi$')
    plt.savefig(prefix+'regions.png')

def plot_hist(hro, label, prefix=''):
    '''hro is an HRO class object as defined in HRO.py'''
    
    plt.close('all')
    plt.figure(figsize=[5,4], dpi=300)
    angle = np.linspace(0, 90, hro.histbins)
    plt.plot(angle, hro.hist, linewidth=2, label=label+' PRS: Zx={0}'.format(np.round(hro.Zx, 2)))
    #plt.ylim(0,1)
    #plt.xlim(0,90)
    plt.xlabel('Relative angle, $\phi$')
    plt.ylabel('Histogram Density')
    plt.grid(True, linewidth=0.5)
    plt.title('Histogram of Relative Orientation', fontsize=10)
    plt.legend(fontsize=12)
    plt.savefig(prefix+'phi_histogram.png')

def plot_histShaded(hro, label, prefix=''):
    '''hro is an HRO class object as defined in HRO.py'''
    
    plt.close('all')
    plt.figure(figsize=[5,4], dpi=300)
    angle = np.linspace(0, 90, hro.histbins)
    plt.plot(angle, hro.hist, linewidth=2, color='black')
    plt.fill_between(angle[(np.where(angle<30))],hro.hist[(np.where(angle<30))], facecolor='orange', alpha=0.3)
    plt.fill_between(angle[(np.where(angle>60))],hro.hist[(np.where(angle>60))], facecolor='purple', alpha=0.2)
    plt.text(8, 0.18, 'Parallel')
    plt.text(65, 0.18, 'Perpendicular')
    plt.text(5, np.max(hro.hist)*0.9,  'Zx = {0}'.format(np.round(hro.Zx, 1)), fontsize=14)
    plt.text(5, np.max(hro.hist)*0.8,  '<$\phi$> = {0}'.format(np.round(hro.meanPhi*180/np.pi, 1)), fontsize=14)
    plt.xlabel('Relative angle, $\phi$')
    plt.ylabel('Histogram Density')
    plt.savefig(prefix+'phi_SHADEDhistogram.png')

def plot_secthist(hro, label, prefix=''):
    '''hro is an HRO class object as defined in HRO.py'''
    
    plt.close('all')
    plt.figure(figsize=[5,4], dpi=300)
    angle = np.linspace(0, 90, hro.histbins)
    colors = [cm.Spectral(x) for x in np.linspace(0, 1, 6)]
    for i in range(len(hro.hist_array)):
        if hro.sections[i] > 1e20: 
            digits=-22
        else: 
            digits=2
        if i==0:
                plt.plot(angle, hro.hist_array[i], linewidth=1, label='X<{0}, {1} phi points'.format(np.round(hro.sections[i], digits), hro.nvectors[i]), color=colors[i])
        else:
                plt.plot(angle, hro.hist_array[i], linewidth=1, label='{0}<X<{1}, {2} phi points'.format(np.round(hro.sections[i-1], digits), np.round(hro.sections[i], digits), hro.nvectors[i]), color=colors[i])
    plt.xlabel('Relative angle, $\phi$')
    plt.ylabel('Histogram Density')
    plt.grid(True, linewidth=0.5)
    plt.title(label, fontsize=10)
    plt.legend(fontsize=5, bbox_to_anchor=(1,0.8))
    plt.subplots_adjust(top=0.9,bottom=0.2,left=0.15,right=0.75)
    plt.savefig(prefix+'_phi_secthistogram.png')

def plot_FEWsect(hro, label, prefix=''):
    '''hro is an HRO class object as defined in HRO.py'''
    
    plt.close('all')
    plt.figure(figsize=[5,4], dpi=300)
    angle = np.linspace(0, 90, hro.histbins)
    maptyp=prefix.split('/')[3]

    linestyles=[':', '--', '-']
    
    for i in range(len(hro.hist_array)):
        if maptyp=='ColumnDensity':
            digits=-22
        elif maptyp=='ALMA': 
            digits=2
        else:
            digits=0

        if maptyp=='ColumnDensity':
            symbol='N$_{H}$'
            units='cm$^{-3}$'
        elif maptyp=='Temperature' or maptyp=='Highres_Temperature':
            symbol='T'
            units='K'
        elif maptyp=='Highres_ColumnDensity':
            symbol='A$_{v}$'
            units=''
        else:
            symbol='I'
            units=''

        if i==0:
            if digits==0:
                plt.plot(angle, hro.hist_array[i], linewidth=2, label=symbol + '<{0}'.format(int(hro.sections[i])) + units, linestyle=linestyles[i])
            else:
                plt.plot(angle, hro.hist_array[i], linewidth=2, label=symbol + '<{0}'.format(np.round(hro.sections[i], digits)) + units, linestyle=linestyles[i])
        else:
            if digits==0:
                plt.plot(angle, hro.hist_array[i], linewidth=2, label='{0}<'.format(int(hro.sections[i-1]))+symbol+'<{0}'.format(int(hro.sections[i]))+units, linestyle=linestyles[i])
            else:
                plt.plot(angle, hro.hist_array[i], linewidth=2, label='{0}<'.format(np.round(hro.sections[i-1], digits))+symbol+'<{0}'.format(np.round(hro.sections[i], digits))+units, linestyle=linestyles[i])
    plt.xlabel('Relative angle, $\phi$')
    plt.ylabel('Histogram Density')
    plt.legend(fontsize=11)
    #plt.subplots_adjust(top=0.9,bottom=0.2,left=0.15,right=0.75)
    plt.savefig(prefix+'phi_3Sects.png')