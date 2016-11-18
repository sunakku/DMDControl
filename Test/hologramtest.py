import math
import numpy as np
import scipy.ndimage as nd
import scipy.misc
import sys
sys.path.append('../Library/')
from HologramGeneration import ImageProcessing as IP
from UsbControl import LightCrafter6500Device as lc
import UsbControl.PatternCreation as pc
from PIL import Image as pil
import matplotlib.pyplot as plt

'''
hologramtest

take measured phase and amplitude data and generate DMDimage (hologram)
that compensate abberation and create images at the focal point.
using Modules/HologramGeneration/ImageProcessing.py
'''
path = "../Data/PhaseMeasurement/"

def initialize():
    ##########################
    #fetch data
    ##########################

#    Amp = np.genfromtxt(path + 'ampfitted.csv', delimiter=',').T
#    Phase = np.genfromtxt(path +'fithigherorder.csv', delimiter=',').T

    Amp = np.genfromtxt(path + 'amp_full_dmd.csv', delimiter=',').T
    Phase = np.genfromtxt(path +'phase_full_dmd.csv', delimiter=',').T
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


def fourierplane(phasemap, imagesetting, shiftx=0,shifty=0):
    ###########################
    #create gaussian or fourier plane
    ###########################
    target_setting =  {'image':imagesetting, "path":"../Data/Hologram_test/yb.png",
                        'shape':np.shape(phasemap[0]), #only applicable for
                        'width':0.9}#gaussian width. 0~1, from zero to image size*2 tem width
    Target = IP.TargetImage(**target_setting)
    Target.imagefft()
    target = Target.fimage

    ###########################
    #generate hologram
    ###########################
    hologram_setting = {'phasemap':phasemap,
                        'fimage':target,
                        'alpha':8,
                        'method':IP.hologramize,
                        'xshift':shiftx, 'yshift':shifty}
    pattern = IP.Hologram(**hologram_setting)
    pattern.compute_hologram()

    return pattern, Target

def upload_hologram2(phasemap):

    settings = {'compression':'rle','exposure_time':100000}
    dmd_pattern = pc.DMDPattern(**settings)

    xshift = 0
    yshift = 0
    xshift2 = 0.05
    yshift2 = 0.05

    target_setting =  {'image':"sinc", "path":"../Data/Hologram_test/fourier130.png",
                        'shape':np.shape(phasemap[0]), #only applicable for
                        'width':0.3}#gaussian width. 0~1, from zero to image size*2 tem width
    Target = IP.TargetImage(**target_setting)
    Target.imagefft()
    tgt = Target.fimage
    target_setting2 =  {'image':"sinc", "path":"../Data/Hologram_test/fourier130.png",
                        'shape':np.shape(phasemap[0]), #only applicable for
                        'width':0.3}#gaussian width. 0~1, from zero to image size*2 tem width
    Target2 = IP.TargetImage(**target_setting2)
    Target2.imagefft()
    tgt2 = Target2.fimage



    im_shape = phasemap.shape[1]
    X, Y = np.meshgrid(np.arange(im_shape)-im_shape/2,np.arange(im_shape)-im_shape/2)
    phasesum = np.mod(-phasemap[1]+tgt[1]+X*xshift+Y*yshift,2.*math.pi)*12./(2.*math.pi)
    phasesum2 = np.mod(-phasemap[1]+tgt2[1]+X*xshift2+Y*yshift2,2.*math.pi)*12./(2.*math.pi)

    pattern = IP.hologramize2(X,Y,8,phasemap[0],phasesum,tgt[0],phasemap[0],phasesum2,tgt2[0],(1080,1920))
    dmd_pattern.pattern = pattern
    lc_dmd = lc.LC6500Device()
    lc_dmd.upload_image(dmd_pattern)
    showimages(pattern,Target,phasemap)

def upload_hologram_multi(phasemap):

    iteration = 17
    settings = {'compression':'rle','exposure_time':100000}
    dmd_pattern = pc.DMDPattern(**settings)

    phasesum = np.zeros((iteration,phasemap.shape[1],phasemap.shape[1]))
    imageamp = np.zeros((iteration,phasemap.shape[1],phasemap.shape[1]))
    amp = np.zeros((iteration,phasemap.shape[1],phasemap.shape[1]))
    im_shape = phasemap.shape[1]
    X, Y = np.meshgrid(np.arange(im_shape)-im_shape/2,np.arange(im_shape)-im_shape/2)

    target_setting =  {'image':"tem", "path":"../Data/Hologram_test/fourier130.png",
                        'shape':np.shape(phasemap[0]), #only applicable for
                        'width':0.7}#gaussian width. 0~1, from zero to image size*2 tem width
    Target = IP.TargetImage(**target_setting)
###########################################
###################################Yb
###########################################
    desiredshape = ["sinc","sinc","sinc","sinc","sinc","sinc","sinc","sinc","sinc","sinc","sinc","sinc","sinc","sinc","sinc","sinc","sinc"]
    xshift = [-0.04,-0.02,0.02 ,0.04 , 0    ,0    , 0 ,-0.06 ,-0.06 ,-0.06 ,-0.06  ,-0.06 ,-0.08 ,-0.08 ,-0.1  ,-0.09 ,-0.09  ]
    yshift = [0.04 ,0.02 ,0.02 ,0.04 ,-0.02 ,-0.04, 0 ,0     ,0.02  ,0.04  ,-0.025 ,-0.05 ,-0.00 ,-0.05 ,-0.027 ,-0.02 ,-0.04]



    for i in xrange(iteration):
        Target.image = "sinc"#desiredshape[i]
        Target.imagefft()
        phasesum[i] = np.mod(-phasemap[1]+Target.fimage[1]+(Y-X)*xshift[i]+(Y+X)*yshift[i],2.*math.pi)*12./(2.*math.pi)
        imageamp[i] = Target.fimage[0]
        amp[i] = phasemap[0]

    pattern = IP.holograms(X,Y,7,amp,phasesum,imageamp,(1080,1920))


    dmd_pattern.pattern = pattern
    lc_dmd = lc.LC6500Device()
    lc_dmd.upload_image(dmd_pattern)
    showimages(pattern,Target,phasemap)

def upload_hologram(pattern,target,phasemap):

    ###########################
    #upload to DMD
    ###########################
    lc_dmd = lc.LC6500Device()
    settings = {'compression':'rle','exposure_time':500000}
    dmd_pattern = pc.DMDPattern(**settings)
    dmd_pattern.pattern = pattern.hologram
    lc_dmd.upload_image(dmd_pattern)

    showimages(pattern.hologram, Target, phasemap)

def upload_hologram_sequence(phasemap):

    ###########################
    #upload to DMD
    ###########################
    settings = {'compression':'rle','exposure_time':100000}
    dmd_pattern = pc.DMDPattern(**settings)

    dmd_pattern_list = []
    pattern, Target = fourierplane(phasemap,"sinc",0.2,0)
    dmd_pattern.pattern = pattern.hologram
    dmd_pattern_list.append(dmd_pattern)

    dmd_pattern2 = pc.DMDPattern(**settings)
    pattern2, Target = fourierplane(phasemap,"sinc",0,0)
    dmd_pattern2.pattern = pattern2.hologram
    dmd_pattern_list.append(dmd_pattern2)

    dmd_pattern3 = pc.DMDPattern(**settings)
    pattern3, Target = fourierplane(phasemap,"sinc",-0.2,0)
    dmd_pattern3.pattern = pattern3.hologram
    dmd_pattern_list.append(dmd_pattern3)

    lc_dmd = lc.LC6500Device()
    dmd_patterns = {'patterns' : dmd_pattern_list}
    lc_dmd.upload_image_sequence(dmd_patterns)


def showimages(pattern,Target, phasemap):
    ###########################
    #show images (save)
    ###########################
    #scipy.misc.toimage(pattern, cmin=0.0, cmax=1).save(path+'hologram2.bmp')
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
    #plt.imshow(pattern.hologram)
    plt.imshow(pattern)
    plt.title('dmdimage')


    plt.subplot(2,5,6)
    plt.gray()
    plt.imshow(target[1],interpolation="none")
    plt.title('target phase')

    plt.subplot(2,5,9)
    plt.gray()
    plt.imshow(phasemap[0],interpolation="none")
    plt.title('amp')

    plt.subplot(2,5,10)
    plt.gray()
    plt.imshow(phasemap[1],interpolation="none")
    plt.title('phase')

    plt.subplot(2,5,7)
    plt.gray()
    plt.imshow(pattern[540-xsize/2:540+xsize/2,960-ysize/2:960+ysize/2])
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
    pattern, Target = fourierplane(phasemap,"gaus",-0.,0.)
    #upload_hologram(pattern,Target,phasemap)
    #upload_hologram2(phasemap)
    upload_hologram_multi(phasemap)

    #upload_hologram_sequence(phasemap)
    #showimages(pattern.hologram, Target, phasemap)


