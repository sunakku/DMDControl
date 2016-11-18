import time, threading
import pygame, sys
sys.path.append('../Scripts/PhaseMeasurement')
from pygame.locals import *
import hologramtest as ht
import PhaseMeasure as pm

class imagshow():
    def __init__(self):
        path = "../Data/PhaseMeasurement_test/"
        # set up pygame
        pygame.init()
        # set up the window
        self.windowSurface = pygame.display.set_mode((1920, 1080), 0, 8)
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        self.image = pygame.image.load('./../Scripts/PhaseMeasurement/patches.bmp').convert() 
        self.image2 = pygame.image.load(path+"hologram2.jpg").convert() 


        self.stop_event = threading.Event() #flag stop or not 
        self.swap_event = threading.Event()  #flag increment or not
        #create thread and start
        self.thread = threading.Thread(target = self.showing)
        self.thread.start()

        self.phasemap=ht.initialize()  

        """
        def GetInput():
            global usrcode
            key = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE: quit()
        """

        self.windowSurface.blit(self.image,(0,0))
        pygame.display.flip()


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
        print "finished"

    def phasemeasure(self,x,y,Phase,Size,GratingSize,x1,y1):
        self.image2 = pygame.surfarray.make_surface(pm.pattern(x,y,Phase,Size,GratingSize,x1,y1).T*255)
        self.swap_event.set() 

    def imgswap(self,image):
        pattern, Target = ht.fourierplane(self.phasemap,image)
        self.image2 = pygame.surfarray.make_surface(pattern.hologram.T*255)
        self.swap_event.set() 

    def showing(self):
    # run the game loop
        while not self.stop_event.is_set():
            #time.sleep(0.1)
            if self.swap_event.is_set():
                self.image = self.image2
                self.swap_event.clear()            
                self.windowSurface.blit(self.image,(0,0))
                pygame.display.flip()

if __name__ == '__main__':
    h = imagshow()      #start thread
    #h.phasemeasure(-200,200,0,40,24,0,0)
    #for i in xrange(100):
    #    j=i%10
    #    k=i//10
    #    h.phasemeasure(j*100-200,k*100-200,0,40,24,0,0)
    #    time.sleep(2)
    #h.imgswap("tem")
    #for i in xrange(4):
    #    h.imgswap("linsinc")         #change image
    #    time.sleep(2)
    #    h.imgswap("gaus")
    #    time.sleep(2)
    #    h.imgswap("sinc")
    #    time.sleep(2)
    #    h.imgswap("tem")    
    #    time.sleep(2)
    time.sleep(100)
    h.stop()        
    