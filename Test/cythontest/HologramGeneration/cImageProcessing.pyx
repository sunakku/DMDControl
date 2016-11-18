import math
import numpy as np
cimport numpy as cnp
#cimport numpy as np
import scipy.ndimage as nd
import scipy.interpolate as interp
import scipy.special as special
import skimage as ski#from skimage import filters as 
from skimage.filters.rank import median
from skimage.morphology import disk
import sys
from PIL import Image as pil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rand_smoothing(gratingphase,alpha,omega,xsize,ysize):
    value = (np.tanh(alpha*(gratingphase+omega/2))-np.tanh(alpha*(gratingphase-omega/2)))/2  #probability
    return np.divide(np.random.rand(xsize,ysize),value)
     #x <1 at a given probability for each array elements


'''
TargetImage object
load target image, apply fft2 and return fourier images of desired resolution
'''
class TargetImage(object):
    def __init__(self, **kwargs):
        self.image = kwargs.pop('image', None)
        self.targetshape = kwargs.pop('shape', (540,540))
        self.path =  kwargs.pop('path', None)
        self.fimage = np.zeros((3,self.targetshape[0],self.targetshape[1])) #to use in sinc_i function, first dim has #3
        self.width = kwargs.pop('width', 1.0)        ##gaussian width within [0,1] from zero to imagesize*2
        self.FWHM = self.width*self.targetshape[0]*2.354
        self.X, self.Y = np.meshgrid(np.arange(self.targetshape[0])-self.targetshape[0]/2,np.arange(self.targetshape[0])-self.targetshape[0]/2)


    def shift(self,x,y):
        pass

    def imagefft(self):
        if (self.image == "gaus"):
            self.fimage[0] = gaus(self.X,self.Y,self.width) #amplitude only
        elif (self.image == "tem"):
            self.fimage[0] = np.absolute(tem(self.X,self.Y,self.width))
            self.fimage[1] = math.pi*(tem(self.X,self.Y,self.width)<0)
        elif (self.image =="sinc"):
            self.fimage[0] = np.absolute(sinc(self.X,self.Y,self.width))
            self.fimage[1] = math.pi*(sinc(self.X,self.Y,self.width)<0)
        elif (self.image =="linsinc"):
            self.fimage[0] = np.absolute(linsinc(self.X,self.Y,self.width))
            self.fimage[1] = math.pi*(linsinc(self.X,self.Y,self.width)<0)
        elif (self.image =="lgaus"):
            self.fimage[0] = np.abs(lgaus(self.X,self.Y,self.width))
            self.fimage[1] = np.angle(lgaus(self.X,self.Y,self.width))#math.pi*(lgaussian(self.targetshape,self.width)<0)
        elif (self.image == "circles"):
            self.fimage[0] = 1*(self.X**2+self.Y**2 <100)+((self.X-100)**2+self.Y**2 <100)
            #image = 1*(self.X**2+self.Y**2 <10000)+((self.X-300)**2+self.Y**2 <10000)
            #fourier = np.fft.fftshift(np.fft.fft2(image))
            #self.fimage[0] = np.abs(fourier)
            #self.fimage[1] = np.angle(fourier)
        elif (self.image == "lines"):
            self.fimage[0] = 1*(np.mod(self.X+self.Y,500) <50)
        else:
            #fourier = imload(self.path)
            #fa = self.targetshape[0]/fourier.shape[0]
            #self.fimage[0,fa/2::fa,fa/2::fa] = fourier
            im = imload(self.path)
            im-=np.amin(im)
            im = np.rint(im)*1.
            plt.figure(4)
            plt.imshow(im)
            plt.colorbar()
            fourier = np.fft.fftshift(np.fft.fft2(im,norm="ortho"))
            imx, imy = np.meshgrid(np.arange(im.shape[0])-im.shape[0]/2,np.arange(im.shape[0])-im.shape[0]/2)
            fourier *= (1-gaus(imx,imy,3) )##filter out center peak
            plt.figure(3)
            plt.imshow(np.absolute(fourier))
            plt.colorbar()
            plt.figure(2)
            plt.imshow(median(np.angle(fourier)/(math.pi),disk(2)),interpolation="None")
            plt.show()



            ######make sure shape of image is same as phasemap_large.
            ######interpolation can be applied only if the targat shape is multiple of imshape
            if ((self.targetshape[0]>fourier.shape[0])&(self.targetshape[0]%fourier.shape[0] ==0)):
                fa = self.targetshape[0]/fourier.shape[0]
                self.fimage[0,fa/2::fa,fa/2::fa]=np.absolute(np.abs(fourier))
                self.fimage[1,fa/2::fa,fa/2::fa]=math.pi*(np.abs(fourier)<0)
                #ski.filters.rank.median(np.angle(fourier)/(math.pi),ski..morphology.disk(4))
                self.fimage=nearest_i(fa,self.fimage)
                self.fimage[2]=0

                #fim = np.fft.fftshift(np.fft.fft2(sinc_i(fa,self.fimage)[0],norm="ortho"))
                #self.fimage[0] = np.absolute(fim)
                #self.fimage[1] = np.angle(fim)
            elif (self.targetshape[0]==fourier.shape[0]):
                #self.fimage[0] = np.abs(fourier)
                self.fimage[1] = np.angle(fourier)
                self.fimage[0] = np.absolute(fourier)
                #self.fimage[1] = math.pi*(np.abs(fourier)<0)
            else:
                print "cannot interpolate fourier plane"
                print "interpolation can be applied only if the targatshape is multiple of phasemap.shape"
                sys.exit(1)
        #apply gaussian filter to amplitude. (sometimes hologram amp around the edge saturate)
        #self.fimage[0] *= gaus(self.X,self.Y,0.3)	
'''
phasemap object
increase the resolution of measured phasemap by desired factors
return numpy array of size being (xsize*factor,ysize*factor)
where xsize,ysize = np.shape(Amp)
'''
class Phasemap(object):
    def __init__(self, **kwargs):
        self.method = kwargs.pop('method', None)
        self.Amp = kwargs.pop('Amp', np.zeros((9,9)))
        self.Phase = kwargs.pop('Phase', np.zeros((9,9)))
        self.Phase_res = kwargs.pop('Phase residual', np.zeros((9,9)))
        self.factor = kwargs.pop('factor', 30)
        self.phasemap_large = np.zeros((3,self.factor*np.shape(self.Amp)[0],self.factor*np.shape(self.Amp)[1] ))

    def interpolate(self):
        """
        returns the interpolated image
        """
        #upsample images
        fa=self.factor
        self.phasemap_large[0,fa/2::fa,fa/2::fa]=self.Amp
        self.phasemap_large[1,fa/2::fa,fa/2::fa]=self.Phase
        self.phasemap_large[2,fa/2::fa,fa/2::fa]=self.Phase_res
        self.phasemap_large = self.method(self.factor,self.phasemap_large)

    def interpolate_scipy(self):
        self.phasemap_large = self.method(self.factor,self.Amp,self.Phase,self.Phase_res,self.phasemap_large)

    def interpolate_fft(self):
        ftAmp = np.fft.fft2(Amp)
        ftPhase = np.fft.fft2(Phase)
        ftphres = np.fft.fft2(Phase_res)
        self.phasemap_large = self.method(self.factor, ftAmp,ftPhase,ftphres)

'''
Hologram object
returns full DMD image (1920*1080)
'''

class Hologram(object):
    def __init__(self, **kwargs):
        self.method = kwargs.pop('method', 'hologramize')
        self.fimage =  kwargs.pop('fimage', np.zeros((3,540,540)))
        self.factor = kwargs.pop('factor', 30)
        self.phasemap_large = kwargs.pop('phasemap',np.zeros((3,540,540)))
        self.resolution = kwargs.pop('resolution', (1080,1920))
        self.alpha = kwargs.pop('alpha', 4)
        self.hologram = np.zeros(self.resolution)
        self.im_shape = np.shape(self.phasemap_large)[1]
        self.X, self.Y = np.meshgrid(np.arange(self.im_shape)-self.im_shape/2,np.arange(self.im_shape)-self.im_shape/2)

    def compute_hologram(self):
        phasesum = np.mod(self.fimage[1],2.*math.pi)*12./(2.*math.pi)
        #phasesum = np.mod(self.phasemap_large[1]+self.fimage[1],2.*math.pi)*12./(2.*math.pi)
        self.hologram = self.method(self.X,self.Y,self.alpha,self.phasemap_large[0],phasesum,self.fimage[0],self.resolution)


############functions

##############################
#image loading and fourier transf.
##############################
def imload(path):
    return np.array(pil.open(path))[:,:,1]/255.
def imreshape(image):
    '''
    under construction
    '''
    return image

def gaus(xx,yy,width):
    return 1*np.exp(-(xx**2+yy**2)/(2.*(width*xx.shape[0]*2)**2))

def tem(xx,yy,width):
    order = np.zeros((4,4))
    order[3,3] =1
    width *= xx.shape[0]
    #return 1*np.polynomial.hermite.hermval(np.sqrt(2)*xx/width,order)*np.polynomial.hermite.hermval(np.sqrt(2)*yy/width,order2)*np.exp(-(xx**2+yy**2)/width)
    return 1*np.polynomial.hermite.hermval2d(np.sqrt(2)*xx/width,np.sqrt(2)*yy/width,order)*np.exp(-(xx**2+yy**2)/width**2)

def lgaus(xx,yy,width):
    order=4
    lorder = 0
    width *= xx.shape[0]*100
    r=np.sqrt(xx**2+yy**2)
    laguerre = special.genlaguerre(order,lorder)
    return 1*(r**lorder)*laguerre(2*r**2/width**2)*np.exp(-r**2/width**2)*np.exp(1j*lorder*math.pi) #replace 1 by phi


def sinc(xx,yy,width):
    width *= xx.shape[0]/1.7
    return np.sinc(np.sqrt(xx**2+yy**2)/width)

def linsinc(xx,yy,width):
    width *= xx.shape[0]/1.7
    return np.sinc((xx-yy)/width)



##############################
#interpolation functions
##############################

#test data. gaussian phase map and intensity map
def gaussian(factor,phasemap):
    width=0.2*np.shape(phasemap)[1]
    nx = np.arange(0,np.shape(phasemap)[1]) - np.shape(phasemap)[1]/2
    xx,yy = np.meshgrid(nx,nx)
    phasemap[0] = 1*np.exp(-(xx**2+yy**2)/(2.*width**2))
    phasemap[1] = 1*np.exp(-(xx**2+yy**2)/(2.*width**2))
    phasemap[2] = 0
    return phasemap

def nearest_i(factor,phasemap):
    kernel = np.ones((factor,factor))
    maxval = phasemap.max(axis=2).max(axis=1)
    minval = phasemap.min(axis=2).min(axis=1)
    for j in np.arange(3):
        convolved = nd.convolve(phasemap[j],kernel)
        convolved -= np.amin(convolved)#set minimum to zero
        phasemap[j] = (maxval[j]-minval[j])*convolved/(np.amax(convolved))#rescale
    phasemap[0] /= np.amax(phasemap[0]) #normalize intensity
    return phasemap


##using ndi convolve function
def linear_i(factor,phasemap):
    rect = np.zeros((factor*2,factor*2))
    rect[factor/2:3/2.*factor,factor/2:3/2.*factor] = 1
    kernel=nd.convolve(rect,rect)
    maxval = phasemap.max(axis=2).max(axis=1)
    minval = phasemap.min(axis=2).min(axis=1)
    for j in np.arange(3):
        convolved = nd.convolve(phasemap[j],kernel)
        convolved -= np.amin(convolved)#set minimum to zero
        phasemap[j] = (maxval[j]-minval[j])*convolved/(np.amax(convolved))#rescale
    phasemap[0] /= np.amax(phasemap[0]) #normalize intensity
    return phasemap


def sinc_i(factor,phasemap):
    size = np.shape(phasemap)[1]
    kernel = np.zeros((size,size))
    width = size/2/factor
    kernel[size/2-width:size/2+width,size/2-width:size/2+width]=1
    kernel = np.fft.fftshift(kernel)
    maxval = phasemap.max(axis=2).max(axis=1)
    minval = phasemap.min(axis=2).min(axis=1)
    for i in np.arange(3):
        convolved = np.fft.ifft2(np.fft.fft2(phasemap[i])*kernel).real
        convolved -= np.amin(convolved)#set minimum to zero
        phasemap[i] = (maxval[i]-minval[i])*convolved/(np.amax(convolved))#rescale
    phasemap[0] /= np.amax(phasemap[0]) #normalize intensity
    return phasemap

##using scipy.interpolate.interp2d
def cubic_i(factor,amp,phase,phres,phasemap):
    size = np.shape(phase)[1]
    x = np.arange(size)
    y = np.arange(size)
    x_new = np.arange(factor*size)
    y_new = np.arange(factor*size)
    phasemap[0] = interp.interp2d(x,y,np.double(amp),kind='cubic')(x_new,y_new)
    phasemap[1] = interp.interp2d(x,y,phase,kind='cubic')(x_new,y_new)
    phasemap[2] = interp.interp2d(x,y,phres,kind='cubic')(x_new,y_new)
    return phasemap

##using fourier plane convolution(simply multiply in fourier plane) for speed up
def zeropad(factor,ftAmp,ftPhase,ftphres):
    '''
    **under construction as of nov 1 2016 @sunami
    '''
    zp_phasemap = np.zeros((3,factor*np.shape(ftAmp)[0],factor*np.shape(ftAmp)[1] ))
    zp_phasemap[0] = np.fft.ifft(Amp)




##############################
#hologram functions
##############################
def hologramize(X,Y,alpha,np.ndarray[cnp.double_t,ndim=2] amp,np.ndarray[cnp.double_t,ndim=2] phasesum,np.ndarray[cnp.double_t,ndim=2] imageamp,resolution):
    cdef double dmdpattern = np.zeros(resolution)
    cdef double gratingphase = np.absolute(np.mod(Y-X+phasesum,12.)-6)/6.#np.mod(Y-X,12.)/12.#
    #####set minimum value for amp.
    #####to avoid inf in omega
    #####effect to intensity:1%
    #####should find way around this problem
    amp += 0.01
    cdef double omega = np.divide(imageamp,amp)/np.amax(np.divide(imageamp,amp))#imageamp#
    cdef int Xsize,Ysize = np.shape(amp)
    cdef double hologram=1*(rand_smoothing(gratingphase,alpha,omega,Xsize,Ysize) <1)
    dmdpattern[540-Xsize/2:540+Xsize/2,960-Ysize/2:960+Ysize/2]=hologram
    return dmdpattern

if __name__ == '__main__':
    print "test"
