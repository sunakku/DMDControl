#from __future__ import division
import time, threading
import pygame, sys
from pygame.locals import *
import numpy as np
import math
import cProfile
cimport numpy as np
ctypedef np.int_t DTYPEi_t
ctypedef np.float64_t DTYPE_t
#cimport cython
#np.import_array()


def rand_smoothing(np.ndarray[DTYPE_t,ndim=2] gratingphase,DTYPE_t alpha,np.ndarray[DTYPE_t,ndim=2] omega,DTYPEi_t xsize,DTYPEi_t ysize):
    value = (np.tanh(alpha*(gratingphase+omega/2))-np.tanh(alpha*(gratingphase-omega/2)))/2  #probability
    return np.divide(np.random.rand(xsize,ysize),value)
     #x <1 at a given probability for each array elements


def imagefft(X,Y,width,image):
    cdef np.ndarray[DTYPE_t,ndim=3] fimage = np.zeros((3,X.shape[0],X.shape[1]),dtype=np.float64)
    if (image == "gaus"):
        fimage[0] = gaus(X,Y,width) #amplitude only
    elif (image == "tem"):
        fimage[0] = np.absolute(tem(X,Y,width))
        fimage[1] = math.pi*(tem(X,Y,width)<0)
    elif (image =="sinc"):
        fimage[0] = np.absolute(sinc(X,Y,width))
        fimage[1] = math.pi*(sinc(X,Y,width)<0)
    elif (image =="linsinc"):
        fimage[0] = np.absolute(linsinc(X,Y,width))
        fimage[1] = math.pi*(linsinc(X,Y,width)<0)
    elif (image =="lgaus"):
        fimage[0] = np.abs(lgaus(X,Y,width))
        fimage[1] = np.angle(lgaus(X,Y,width))#math.pi*(lgaussian(targetshape,width)<0)
    elif (image == "circles"):
        fimage[0] = 1*(X**2+Y**2 <100)+((X-100)**2+Y**2 <100)
    else:
        pass
    return fimage



class imagshow():
    def __init__(self):
        path = "../../Data/PhaseMeasurement_test/"       
        # set up pygame
        pygame.init()
        self.windowSurface = pygame.display.set_mode((1920, 1080), 0, 8)
        self.image = pygame.image.load(path+"hologram.jpg").convert() 
        self.image2 = pygame.image.load(path+"hologram2.jpg").convert() 
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        self.stop_event = threading.Event() #flag stop or not 
        self.swap_event = threading.Event()  #flag increment or not

        self.Xsize, self.Ysize = np.genfromtxt(path + 'ampfitted.csv', delimiter=',').T.shape
        #cdef np.ndarray[DTYPE_t,ndim=3] 
        self.phasemap = np.zeros((3,self.Xsize,self.Ysize))
        self.phasemap[0] = np.genfromtxt(path + 'ampfitted.csv', delimiter=',').T + 0.01
        self.phasemap[1] = np.genfromtxt(path +'fithigherorder.csv', delimiter=',').T
        self.X, self.Y = np.meshgrid(np.arange(self.Xsize)-self.Xsize/2,np.arange(self.Ysize)-self.Ysize/2)
        #create thread and start
        self.thread = threading.Thread(target = self.showing)
        self.thread.start()
        self.windowSurface.blit(self.image,(0,0))
        pygame.display.flip()
        self.count = 0

        #pygame.surfarray.blit_array()
        '''you can blit array directly and its fast. values must be [0,255] and 8 depth'''
    def GetInput():
        key = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE: quit()

    def stop(self):
        """stop thread"""
        pygame.quit()
        self.stop_event.set()
        self.thread.join() 
        print "finished" + str(self.count)


    def profile(self):
        cProfile.run('self.imgswap_single()',sort='time')

    def imgswap(self):
        for i in xrange(100):
            image = "linsinc"        
            self.image2 = pygame.surfarray.make_surface(self.hologramize(image).T*255)
            self.swap_event.set() 
            image = "gaus"        
            self.image2 = pygame.surfarray.make_surface(self.hologramize(image).T*255)
            self.swap_event.set() 

    def imgswap_single(self):
            image = "linsinc"        
            self.image2 = pygame.surfarray.make_surface(self.hologramize(image).T*255)
            self.swap_event.set() 


    def showing(self):
    # run the game loop
        while not self.stop_event.is_set():
            #time.sleep(0.1)
            self.count +=1
            pygame.event.pump()
            if self.swap_event.is_set():
                #self.image = self.image2
                self.swap_event.clear()            
                self.windowSurface.blit(self.image2,(0,0))
                pygame.display.flip()
    '''
    def hologramize(X,Y,alpha,amp,phasesum,imageamp,resolution):
        dmdpattern = np.zeros(resolution)
        gratingphase = np.absolute(np.mod(Y-X+phasesum,12.)-6)/6.#np.mod(Y-X,12.)/12.#
        #####set minimum value for amp.
        #####to avoid inf in omega
        #####effect to intensity:1%
        #####should find way around this problem
        amp += 0.01
        omega = np.divide(imageamp,amp)/np.amax(np.divide(imageamp,amp))#imageamp#
        Xsize, Ysize = amp.shape
        hologram=1*(rand_smoothing(gratingphase,alpha,omega,Xsize,Ysize) <1)
        dmdpattern[540-Xsize/2:540+Xsize/2,960-Ysize/2:960+Ysize/2]=hologram
        return dmdpattern
    '''
    def hologramize(self,imagesetting):
        width = 0.5
        alpha = 14

        fimage = imagefft(self.X,self.Y,width,imagesetting)
        phasesum = np.mod(fimage[1],2.*float(math.pi))*12./(2.*float(math.pi))
        cdef np.ndarray[DTYPE_t,ndim=2] dmdpattern = np.zeros((1080,1920))
        cdef np.ndarray[DTYPE_t,ndim=2] gratingphase = np.absolute(np.mod(self.Y-self.X+phasesum,12.)-6)/6.
        cdef np.ndarray[DTYPE_t,ndim=2] omega = np.divide(fimage[0],self.phasemap[0])/np.amax(np.divide(fimage[0],self.phasemap[0]))
        dmdpattern[540-self.Xsize/2:540+self.Xsize/2,960-self.Ysize/2:960+self.Ysize/2]=1.*(rand_smoothing(gratingphase,alpha,omega,self.Xsize,self.Ysize) <1)
        return dmdpattern


def imload(path):
    return np.array(pil.open(path))[:,:,1]/255.
def imreshape(image):
    '''
    under construction
    '''
    return image

def gaus(xx,yy,width):
    return 1*np.exp(-(xx**2+yy**2)/(2.*(width*2)**2))

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





if __name__ == '__main__':




    h = imagshow()      #start thread
    #h.phasemeasure(-200,200,0,40,24,0,0)
    #for i in xrange(100):
    #    j=i%10
    #    k=i//10
    #    h.phasemeasure(j*100-200,k*100-200,0,40,24,0,0)
    #    time.sleep(2)
    time.sleep(2)
    h.imgswap()
    #for i in xrange(4):
    #    h.imgswap("linsinc")         #change image
    #    time.sleep(2)
    #    h.imgswap("gaus")
    #    time.sleep(2)
    #    h.imgswap("sinc")
    #    time.sleep(2)
    #    h.imgswap("tem")    
    #    time.sleep(2)
    time.sleep(5)
    h.stop()        
    