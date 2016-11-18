import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)


import math
import numpy as np
import scipy.ndimage as nd
import scipy.misc
import sys
#sys.path.append('../../Library/')
from HologramGeneration import cImageProcessing as IP
#from UsbControl import LightCrafter6500Device as lc
#import UsbControl.PatternCreation as pc
from PIL import Image as pil
import matplotlib.pyplot as plt

'''
hologramtest

take measured phase and amplitude data and generate DMDimage (hologram)
that compensate abberation and create images at the focal point.
using Modules/HologramGeneration/ImageProcessing.py
'''
path = "../../Data/PhaseMeasurement_test/"

def initialize():
    ##########################
    #fetch data
    ##########################

    Amp = np.genfromtxt(path + 'ampfitted.csv', delimiter=',').T
    Phase = np.genfromtxt(path +'fithigherorder.csv', delimiter=',').T
    Phaseresidual = np.genfromtxt(path +'Phase_residual.csv', delimiter=',').T
    xsize,ysize = np.shape(Amp)

    phasemap = np.zeros((3,xsize,ysize))
    phasemap[0] = Amp
    phasemap[1] = Phase


    ###########################
    #interpolation
    ###########################
    #print "start interpolation"
    #interpol_setting = {'method':IP.sinc_i,
    #                'factor':factor,
    #                'Amp':Amp,'Phase':Phase,'Phase residual':Phaseresidual}
    #interpolated = IP.Phasemap(**interpol_setting)
    #interpolated.interpolate()
    #phasemap = interpolated.phasemap_large
    return phasemap


def fourierplane(phasemap, imagesetting):
    ###########################
    #create gaussian or fourier plane
    ###########################
    target_setting =  {'image':imagesetting, "path":"../Data/Hologram_test/fourier130.png",
                        'shape':np.shape(phasemap[0]), #only applicable for
                        'width':0.1}#gaussian width. 0~1, from zero to image size*2 tem width
    Target = IP.TargetImage(**target_setting)
    Target.imagefft()
    target = Target.fimage

    ###########################
    #generate hologram
    ###########################
    hologram_setting = {'phasemap':phasemap,
                        'fimage':target,
                        'alpha':14,
                        'method':IP.hologramize}
    pattern = IP.Hologram(**hologram_setting)
    pattern.compute_hologram()

    return pattern, Target

def upload_hologram(pattern):

    ###########################
    #upload to DMD
    ###########################
    lc_dmd = lc.LC6500Device()
    settings = {'compression':'rle','exposure_time':500000}
    dmd_pattern = pc.DMDPattern(**settings)
    dmd_pattern.pattern = pattern.hologram
    lc_dmd.upload_image(dmd_pattern)



def showimages(pattern,Target, phasemap):
    ###########################
    #show images (save)
    ###########################
    scipy.misc.toimage(pattern.hologram, cmin=0.0, cmax=1).save(path+'hologram2.bmp')
    target = Target.fimage

    fig1=plt.figure(1,(25,10))
    xsize,ysize = np.shape(phasemap[0])
    plt.subplot(2,5,1)
    plt.gray()
    if Target.image != "image":
        plt.imshow(target[0],interpolation="none")
        plt.title('target: '+Target.image)
    else:
        plt.imshow(np.fft.ifft2(target[0]*(np.cos(target[1])+1j*np.sin(target[1]))).real,interpolation="none")
        plt.title("target: fft, interpolation and then ifft")

    #plt.subplot(2,5,4)
    #plt.gray()
    #plt.imshow(phasemap[0],interpolation="none")
    #plt.colorbar()
    #plt.title('amplitude interpolated')

    #plt.subplot(2,5,5)
    #plt.gray()
    #plt.imshow(phasemap[1],interpolation="none")
    #plt.colorbar()
    #plt.title('phase interpolated')

    plt.subplot(2,5,2)
    plt.gray()
    plt.imshow(pattern.hologram)
    plt.title('dmdimage')


    plt.subplot(2,5,6)
    plt.gray()
    plt.imshow(target[1],interpolation="none")
    plt.title('target phase')

#    plt.subplot(2,5,9)
#    plt.gray()
#    plt.imshow(phasemap[0],interpolation="none")
#    plt.title('amp original')


#    plt.subplot(2,5,10)
#    plt.gray()
#    plt.imshow(phasemap[1],interpolation="none")
#    plt.title('phase original')
    plt.subplot(2,5,7)
    plt.gray()
    plt.imshow(pattern.hologram[540-xsize/2:540+xsize/2,960-ysize/2:960+ysize/2])
    plt.title('dmdimage')

    if Target.image == "image":
        plt.subplot(2,5,3)
        #plt.gray()
        #plt.imshow(np.log(target[0]))
        plt.imshow(target[0])
        plt.colorbar()
        plt.title('amp) of fourier transformed image ')

        plt.subplot(2,5,8)
        #plt.gray()
        plt.imshow(target[1])
        plt.colorbar()
        plt.title('phase of fourier transformed image')

    plt.show()



if __name__ == '__main__':
    phasemap = initialize()
    pattern, Target = fourierplane(phasemap,"gaus")
    upload_hologram(pattern)
    showimages(pattern, Target, phasemap)


