import time, threading
import pygame, sys
from pygame.locals import *
import hologramtest as ht


class imagshow():
    def __init__(self):
        path = "../Data/PhaseMeasurement_test/"
        # set up pygame
        pygame.init()
        # set up the window
        self.windowSurface = pygame.display.set_mode((1920, 1080), 0, 8)
        pygame.event.set_grab(True)
        self.image = pygame.image.load(path+"hologram.jpg").convert() 
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

    def imgswap(self):
        pattern, Target = ht.fourierplane(self.phasemap,"linsinc")
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
    time.sleep(2)   
    h.imgswap()         #change image
    time.sleep(2)
    h.stop()        
    time.sleep(2)   
    print "finish"