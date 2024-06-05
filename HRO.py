import numpy as np
from astropy.io import fits
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel, Box2DKernel, AiryDisk2DKernel, TrapezoidDisk2DKernel
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt
import aplpy
from scipy import ndimage
from reproject import reproject_interp, reproject_exact

class HRO:
    '''Class to perform 'Histogram of Relative Orientation' Analysis between a given Map and Inferred Magnetic Field Direction'''

    def __init__(self, Map, Qmap=None, Umap=None, Bmap=None, hdu=None, vecMask=None, 
                msk=False, kernel='Gaussian', kstep=0, convention=1, gstep=[1,1,1], 
                histbins=20, compare=False, Qmap2=None, Umap2=None, BinMap=None,Bproj=False):
        '''Intialization of HRO class'''

        # Initialize
        self.Map_full = Map if Map is None else Map.copy()
        self.Map = self.Map_full.data if ((Qmap is None)==False) else projectMap(Map, Bmap).data                                             # Original data before masking
        self.Qmap = Qmap if Qmap is None else Qmap.copy()
        self.Umap = Umap if Umap is None else Umap.copy()
        self.Bmap = Bmap if Bmap is None else Bmap.copy()
        self.hdu = hdu if hdu is None else hdu.copy() 
        self.vecMask = vecMask if vecMask is None else vecMask.copy()                        # hdu contains mask and header WCS information (use for fits plots)
        self.histbins = histbins
        self.BinMap =  self.Map.data if BinMap is None else BinMap.copy().data

         # Convert Bmap to radians
        if (Bmap is None)==False and np.nanmax(self.Bmap.data) > np.pi*2:
                self.Bmap.data = ((self.Bmap.data)*u.deg).to(u.rad).value 

        # Perform HRO analysis:
        if compare==False:
            self.HROanalysis(convention, msk, hdu, vecMask, gstep, kstep, kernel, histbins, Bproj)

        else:
            self.CompareVectors(convention, msk, hdu, vecMask, gstep, kstep, kernel, histbins, Qmap2, Umap2)


    def HROanalysis(self, convention, msk, hdu, vecMask, gstep, kstep, kernel, histbins, Bproj):

        # Qmap is given
        if (self.Qmap is None)==False:
        # #    assert self.Map.shape == self.Qmap.shape and self.Map.shape == self.Umap.shape and self.Qmap.shape == self.Umap.shape, "Dimensions of Map and Polarization data do not match"
            #Calculate the Polarization angle from Stokes Parameters
            self.Qmap = projectMap(self.Qmap, self.Map_full)
            self.Umap = projectMap(self.Umap, self.Map_full)
            self.vecMask = projectMap(self.vecMask, self.Map_full)
            self.hdu = projectMap(self.hdu, self.Map_full)

            if convention==1:                                                             # Standard convention used for most polarization data 
                self.Efield = 0.5*np.arctan2(self.Umap.data,self.Qmap.data)               # Use 4-quadrant arctan, gives result in radians
            else:
                self.Efield = 0.5*np.arctan2(-1*self.Umap.data,self.Qmap.data)            # Some observations use the Internantional Astronomical Union convention (e.g Planck)

            # Decompose Polarization angle vector into x and y components
            self.Ex = np.cos(self.Efield) 
            self.Ey = np.sin(self.Efield)

            # Find inferred magnetic field direction
            self.Bfield = self.Efield + np.pi/2                                           # Inferred B-field is 90 deg offset from polarization angle
            self.Bx = np.cos(self.Bfield) 
            self.By = np.sin(self.Bfield)
            self.Bmap = self.Map_full.copy()
            self.Bmap.data = self.Bfield

            self.Map_proj = self.Map_full.copy()
        
        else: 
            #1. project
            self.Map_proj = projectMap(self.Map_full, self.Bmap)
            if (self.Map_proj.data.shape != self.hdu.data.shape):
                self.hdu = projectMap(self.hdu, self.Bmap)
            if (self.Map_proj.data.shape != self.vecMask.data.shape):
                self.vecMask = projectMap(self.vecMask, self.Bmap)

            # B-field is given
            self.Bfield = self.Bmap.data                                      # Inferred B-field is 90 deg offset from polarization angle
            self.Bx = np.cos(self.Bfield) 
            self.By = np.sin(self.Bfield)

            #Infer E-field
            self.Efield = self.Bfield + np.pi/2  
            self.Ex = np.cos(self.Efield) 
            self.Ey = np.sin(self.Efield)

        #2. Smooth
        #self.kstep_pix, self.kstep_arcsec = self.KernelLength(self.Map_full, self.Bmap, set_pix=kstep)
        self.dMdx_proj = self.Map_proj.copy()
        self.dMdx = ndimage.filters.gaussian_filter(self.Map_proj.data, [kstep, kstep], order=[1,0], mode='nearest')
        self.dMdx_proj.data = self.dMdx

        self.dMdy_proj = self.Map_proj.copy()
        self.dMdy = ndimage.filters.gaussian_filter(self.Map_proj.data, [kstep, kstep], order=[0,1], mode='nearest')
        self.dMdy_proj.data = self.dMdy

        # Find Gradient magnitude and direction and contour direction
        self.mag = np.sqrt(self.dMdx**2 + self.dMdy**2)                               # Gradient magnitude
        self.grad = np.arctan2(self.dMdy, self.dMdx)                                  # Gradient direction
        self.contour = np.arctan2(self.dMdx, self.dMdy)                               # Contours direction is perpendicular to gradient vector (defined as angle psi in report)

        # Calculate Relative angle
        self.dot   = ((self.dMdy)*(self.Bx) + (self.dMdx)*(self.By))/self.mag         # normalized dot product between B-field and contour (note: x and y of gradient flipped for contour direction)
        self.cross = ((self.dMdy)*(self.By) - (self.dMdx)*(self.Bx))/self.mag         # normalized cross product between B-field and contour
        self.phi_unmasked = np.arctan2(np.abs(self.cross), self.dot)                  # relative angle phi as defined in report, ranges from 0 to 180 deg
        self.phi = self.phi_unmasked.copy()

        # mask phi
        if msk==True:
            for i in range(self.vecMask.data.shape[0]):
                for j in range(self.vecMask.data.shape[1]):
                    #if (hdu.data[i,j]!=1) or (vecMask.data[i,j]!=1) or np.isnan(self.Map[i,j]) or np.isnan(self.dMdx[i,j]) or np.isnan(self.dMdy[i,j]):
                    if (self.vecMask.data[i,j]!=1) or (self.hdu.data[i,j]!=1): #or (self.mag[i,j] < 0.2):
                        self.phi[i,j]=np.nan

        # Caluclate Rayleigh Statistic 
        self.Zx = self.RayleighStatistic(self.phi)

        # Wrap phi angle
        self.phi[self.phi > np.pi/2] = np.pi - self.phi[self.phi > np.pi/2]           # wrap angle so now it ranges from 0 to 90 deg (as relative angle phi=85 is the same as phi=95)

        # Calculate Mean relative angle
        self.MeanPhi()
        
        # Make full phi histogram
        self.hist, self.bin_edges = np.histogram(self.phi, bins=histbins, range=(0, np.pi/2), density=True)

        # Use percentile to split
        partitions = 5
        percentile = [(100/partitions)*(i+1) for i in range(partitions)]
        self.sections = np.nanpercentile(self.BinMap, percentile)
        try:
            self.digitized = np.digitize(self.BinMap, self.sections)
        except:
            self.digitized = np.digitize(self.BinMap, [self.sections[i]*i for i in range(len(self.sections))]) # to avoid monotonically increasing error
        phi_sections = [self.phi[self.digitized == i] for i in range(partitions)]
        self.nvectors = [phi_sect.shape[0] for phi_sect in phi_sections]
        self.hist_array = []; self.bin_edges_array = []
        for phi_sect in phi_sections:
            hist_, bin_edges_ = np.histogram(phi_sect, bins=histbins, range=(0, np.pi/2), density=True)
            self.hist_array.append(hist_)
            self.bin_edges_array.append(bin_edges_)
    
        
    def CompareVectors(self, convention, msk, hdu, vecMask, gstep, kstep, kernel, histbins, Qmap2, Umap2):
        ''' This function calculates the relative angles between magnetic field maps (i.e BLASTPol vs. HAWC'''    
        self.Qmap2 = projectMap(Qmap2, self.Qmap).data
        self.Umap2 = projectMap(Umap2, self.Umap).data
        self.Qmap = self.Qmap.data
        self.Umap = self.Umap.data

        if convention==1:                                                             # Standard convention used for most polarization data 
            self.Efield = 0.5*np.arctan2(self.Umap,self.Qmap)                         # Use 4-quadrant arctan, gives result in radians
            self.Efield2 = 0.5*np.arctan2(self.Umap2,self.Qmap2) 
        else:
            self.Efield = 0.5*np.arctan2(-1*self.Umap,self.Qmap)
            self.Efield2 = 0.5*np.arctan2(self.Umap2,self.Qmap2)                       # Some observations use the Internantional Astronomical Union convention (e.g Planck)

        # Decompose Polarization angle vector into x and y components
        self.Ex = np.cos(self.Efield) 
        self.Ex2 = np.cos(self.Efield2)
        self.Ey = np.sin(self.Efield)
        self.Ey2 = np.sin(self.Efield2)

        # Find inferred magnetic field direction
        self.Bfield = self.Efield + np.pi/2                                           # Inferred B-field is 90 deg offset from polarization angle
        self.Bfield2 = self.Efield2 + np.pi/2
        self.Bx = np.cos(self.Bfield) 
        self.Bx2 = np.cos(self.Bfield2) 
        self.By = np.sin(self.Bfield)
        self.By2 = np.sin(self.Bfield2) 

        # Calculate Relative angle
        self.dot   = ((self.Bx2)*(self.Bx) + (self.By2)*(self.By))                    # normalized dot product between B-field and contour (note: x and y of gradient flipped for contour direction)
        self.cross = ((self.Bx2)*(self.By) - (self.By2)*(self.Bx))                    # normalized cross product between B-field and contour
        self.phi = np.arctan2(np.abs(self.cross), self.dot)                           # relative angle phi as defined in report, ranges from 0 to 180 deg

        if msk==True:
            for i in range(self.vecMask.data.shape[0]):
                for j in range(self.vecMask.data.shape[1]):
                    if (self.vecMask.data[i,j]!=1): 
                        self.phi[i,j]=np.nan
        
        self.Zx = self.RayleighStatistic(self.phi)

        # Wrap phi angle
        self.phi[self.phi > np.pi/2] = np.pi - self.phi[self.phi > np.pi/2]           # wrap angle so now it ranges from 0 to 90 deg (as relative angle phi=85 is the same as phi=95)

        # Calculate Mean relative angle
        self.MeanPhi()
        
        # Make full phi histogram
        self.hist, self.bin_edges = np.histogram(self.phi, bins=histbins, range=(0, np.pi/2), density=True)

        # Use percentile to split
        partitions = 5
        percentile = [(100/partitions)*(i+1) for i in range(partitions)]
        self.sections = np.nanpercentile(self.BinMap, percentile)
        try:
            self.digitized = np.digitize(self.BinMap, self.sections)
        except:
            self.digitized = np.digitize(self.BinMap, [self.sections[i]*i for i in range(len(self.sections))]) # to avoid monotonically increasing error
        # print('digitized shape ', digitized.shape)
        # print(np.min(digitized))
        # print(np.max(digitized))
        phi_sections = [self.phi[self.digitized == i] for i in range(partitions)]
        self.nvectors = [phi_sect.shape[0] for phi_sect in phi_sections]
        # print(phi_1.shape)
        # print(phi_2.shape)
        # print(phi_3.shape)
        self.hist_array = []; self.bin_edges_array = []
        for phi_sect in phi_sections:
            hist_, bin_edges_ = np.histogram(phi_sect, bins=histbins, range=(0, np.pi/2), density=True)
            self.hist_array.append(hist_)
            self.bin_edges_array.append(bin_edges_)
            
    def RayleighStatistic(self, angles, weights=None):
        # angles needs to be on domain (-pi/2, pi/2) (input parameter theta is 2*phi 
        theta = np.arctan(np.tan(angles))*2

        # Set default weights to be 1
        if weights==None:
            weights = np.ones(theta.shape)
            weights[np.isnan(theta)] = np.nan

        # Calculate weighted Projected Rayleigh Statistic, as defined in report
        Zx = np.nansum(weights*np.cos(theta)) / np.sqrt(np.nansum(weights**2)/2)

        #var_Zx = np.sqrt(2*(np.nansum(np.cos(theta) - Zx**2))/n)

        return Zx

    def MeanPhi(self, weights=None):
        # Set default weights to be 1
        if weights==None:
            weights = np.ones(self.phi.shape)
            weights[np.isnan(self.phi)] = np.nan

        x = np.nansum(weights*np.cos(self.phi)) / np.nansum(weights)
        y = np.nansum(weights*np.sin(self.phi)) / np.nansum(weights)
        self.meanPhi = np.arctan2(y, x)

    def Gradient(self, array, step=[1,1,1], dim=None):
        '''Custom Gradient function using central difference, tested by comparing the results to numpy gradient
        '''
        if dim==None: dim=len(array.shape)
        assert dim <= 3, "Only able to perform Gradient in 3 dimensions or less"      # Assert that dimensions be less than 3

        gradx = np.zeros(array.shape)
        gradx[np.isnan(array)] = np.nan
        
        if dim==1:                                                                   # 1D Gradient
            for i in range(array.shape[0]):
                try:
                    if i != (array.shape[0]-1):
                        gradx[i] = (array[i+step[0]] - array[i-step[0]])/(step[0]*2)     # Use Central Difference
                except Exception as e:
                    print(e)
                    pass
            return gradx    

        if dim==2:                                                                   # 2D Gradient                   
            grady = gradx.copy()
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    if np.isnan(array[i,j])==False:
                        try:
                            if i != (array.shape[0]-1) and j != (array.shape[1]-1):
                                gradx[i,j] = (array[i+step[0], j] - array[i-step[0], j])/(step[0]*2)
                                grady[i,j] = (array[i, j+step[1]] - array[i, j-step[1]])/(step[1]*2)
                        except Exception as e:
                            print(e)
                            pass        
            return gradx, grady 

        if dim==3:                                                                  # 3D Gradient
            grady = gradx.copy()
            gradz = gradx.copy()
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    for k in range(array.shape[2]):
                        if np.isnan(array[i,j,k])==False:
                            try:
                                if i != (array.shape[0]-1) and j != (array.shape[1]-1) and k != (array.shape[2]-1):
                                    gradx[i,j,k] = (array[i+step[0], j,k] - array[i-step[0], j,k])/(step[0]*2)
                                    grady[i,j,k] = (array[i, j+step[1],k] - array[i, j-step[1]],k)/(step[1]*2)
                                    grady[i,j,k] = (array[i, j, k+step[2]] - array[i, j, k+step[2]])/(step[2]*2)
                            except Exception as e:
                                print(e)
                                pass        
            return gradx, grady, gradz 

    def SmoothKernel(self, data, size, type='Gaussian'):
        # Choose Kernel type
        if type=='Gaussian':
            kernel = Gaussian2DKernel(size)
        
        elif type=='TopHat':
            kernel = Tophat2DKernel(size)

        elif type=='Box':
            kernel = Box2DKernel(size)

        elif type=='AiryDisk':
            kernel = AiryDisk2DKernel(size)

        elif type=='TrapezoidDisk':
            kernel = TrapezoidDisk2DKernel(size)

        # Convolve kernel with data
        smoothed = convolve(data, kernel)
        # Mask out nan data
        smoothed[np.isnan(data)] = np.nan
        return smoothed

    def getSmoothingSize(self, old, new):
        FWHM_gauss = np.sqrt(new**2 - old**2)
        return FWHM_gauss

    # def KernelLength(self, Map, ref, set_pix=0):
    #     degp1 = np.abs(Map.header['CDELT1'])
    #     degp2 = np.abs(ref.header['CDELT1'])
    #     if set_pix==0:
    #         if degp1 > degp2:
    #             pix = degp1 / degp2
    #         else:
    #             pix = degp2 / degp1
    #     else:
    #         pix = set_pix
    #     pix = int(pix)
    #     arcsec_pix = ((pix * degp2)*u.deg).to(u.arcsec).value
    #     return pix, arcsec_pix

def header_rescale(pre_hdu, rebin):
    """Scale Header according to the rebin size"""
    hdu = pre_hdu.copy()
    naxis1, naxis2 = hdu.header['NAXIS1'], hdu.header['NAXIS2']
    hdu.header['NAXIS1'] = int(rebin * hdu.header['NAXIS1'])
    hdu.header['NAXIS2'] = int(rebin * hdu.header['NAXIS2'])
    hdu.header['CDELT1'] /= rebin
    hdu.header['CDELT2'] /= rebin
    hdu.header['CRPIX1'] = (hdu.header['CRPIX1'] / naxis1) * hdu.header['NAXIS1']
    hdu.header['CRPIX2'] = (hdu.header['CRPIX2'] / naxis2) * hdu.header['NAXIS2']
    hdu.data = np.ones((hdu.header['NAXIS1'], hdu.header['NAXIS2']))
    return hdu                

def projectMap(mapOrigin, ref):
    '''This function projects a given map 'mapOrigin' onto the same WCS coordinates as a reference map and returns the map in the same shape'''
    New = ref.copy()
    #proj, footprint = reproject_exact(mapOrigin, ref.header)
    proj, footprint = reproject_interp(mapOrigin, ref.header)
    #proj[np.isnan(ref.data)] = np.nan
    proj[0].data = proj
    New.data = proj
    return New





        
