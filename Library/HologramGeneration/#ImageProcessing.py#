import math
import numpy as np
import scipy.ndimage as nd
import sys
from PIL import Image as pil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def smoothing_fun(gratingphase,alpha,omega):
	return ( np.tanh(alpha*(gratingphase+omega))-np.tanh(alpha*(gratingphase-omega)))/2

'''
images object
increase the resolution of measured phasemap by desired factors
return numpy array of size being (xsize*factor,ysize*factor)
where xsize,ysize = np.shape(Amp)
'''
class Images(object):
        def __init__(self, **kwargs):
                self.method = kwargs.pop('method', None)
                self.Amp = kwargs.pop('Amp', np.zeros((9,9)))
                self.Phase = kwargs.pop('Phase', np.zeros((9,9)))
                self.Phase_res = kwargs.pop('Phase residual', np.zeros((9,9)))
                self.factor = kwargs.pop('factor', 30)
                self.images_large = np.zeros((3,self.factor*np.shape(self.Amp)[0],self.factor*np.shape(self.Amp)[1] ))

	def interpolate(self):
                """                                                                  
                returns the interpolated image      
                """
		#upsample images
		fa=self.factor
		self.images_large[0,fa/2::fa,fa/2::fa]=self.Amp
		self.images_large[1,fa/2::fa,fa/2::fa]=self.Phase
		self.images_large[2,fa/2::fa,fa/2::fa]=self.Phase_res
                self.images_large = self.method(self.factor,self.images_large)

	def interpolate_fft(self):
		ftAmp = np.fft.fft2(Amp)
		ftPhase = np.fft.fft2(Phase)
		ftphres = np.fft.fft2(Phase_res)
		self.images_large = self.method(self.factor, ftAmp,ftPhase,ftphres)
'''
Hologram object
returns full DMD image (1920*1080) 
'''

class Hologram(object):
        def __init__(self, **kwargs):
                self.method = kwargs.pop('method', 'hologramize')
                self.factor = kwargs.pop('factor', 30)
                self.images_large = kwargs.pop('images',np.zeros((3,540,540)))
		self.resolution = kwargs.pop('resolution', (1920,1080))
		self.alpha = kwargs.pop('alpha', 4)
		self.hologram = np.zeros(self.resolution)
		self.im_shape = np.shape(self.images_large)[1]
		self.X, self.Y = np.meshgrid(np.arange(self.im_shape)-self.im_shape/2,np.arange(self.im_shape)-self.im_shape/2)

	def compute_hologram(self):
		phasesum = np.round(np.mod(self.images_large[1],2*math.pi)*12)
		self.hologram = self.method(self.X,self.Y,self.alpha,self.images_large[0],phasesum,self.resolution)


############functions

##############################
#interpolation functions
##############################
#test data. gaussian phase map and intensity map
def gaus(factor,images):
	width=0.2*np.shape(images)[1]
	nx = np.arange(0,np.shape(images)[1]) - np.shape(images)[1]/2
	xx,yy = np.meshgrid(nx,nx)
	images[0] = 1*np.exp(-(xx**2+yy**2)/(2.*width**2))
	images[1] = 1*np.exp(-(xx**2+yy**2)/(2.*width**2))
	images[2] = 0
	return images

##using ndi convolve function
def linear_i(factor,images):
	rect = np.zeros((factor*2,factor*2))
	rect[factor/2:3/2.*factor,factor/2:3/2.*factor] = 1
	kernel=nd.convolve(rect,rect)
	for i in np.arange(3):
		images[i] = nd.convolve(images[i],kernel)
	return images

def sinc_i(factor,images):
	'''
	under construction 1 nov 2016 @sunami
	'''
	rect = np.zeros((factor*2,factor*2))
	rect[factor/2:3/2.*factor,factor/2:3/2.*factor]=1
	kernel=nd.convolve(rect,rect,mode='wrap')
	for i in np.arange(3):
		images[i] = nd.convolve(images[i],kernel,mode='wrap')
	return images


##using fourier plane convolution(simply multiply in fourier plane) for speed up
def zeropad(factor,ftAmp,ftPhase,ftphres):
	'''
	**under construction as of nov 1 2016 @sunami
	'''
	zp_image = np.zeros((3,factor*np.shape(ftAmp)[0],factor*np.shape(ftAmp)[1] ))
	zp_image[0] = np.fft.ifftw(Amp)	


##############################
#hologram functions
##############################
def hologramize(X,Y,alpha,amp,phasesum,resolution):
	dmdpattern = np.zeros(resolution)
	gratingphase = np.mod(Y+X+phasesum,12)
	omega = np.amin(amp)/amp
	hologram=1*( smoothing_fun(gratingphase,alpha,omega) < omega )
	Xsize,Ysize = np.shape(amp)
	dmdpattern[540-Xsize/2:540+Xsize/2,960-Ysize/2:960+Ysize/2]=hologram
	return dmdpattern
	
if __name__ == '__main__':
	print "test"






