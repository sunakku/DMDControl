import pygame, sys
from pygame.locals import *

path = "../Data/PhaseMeasurement_test/"

# set up pygame
pygame.init()

# set up the window
windowSurface = pygame.display.set_mode((1680, 1050), 0, 8)
pygame.event.set_grab(True)
#Screen = max(pygame.display.list_modes())
#icon = pygame.Surface((1,1)); icon.set_alpha(0); pygame.display.set_icon(icon)
#pygame.display.set_caption("[Program] - [Author] - [Version] - [Date]")
#Surface = pygame.display.set_mode(Screen,FULLSCREEN)

# set up the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

backImg = pygame.image.load(path+"hologram.jpg").convert() 
windowSurface.blit(backImg,(0,0))

# draw the window onto the screen
pygame.display.update()

def GetInput():
    global usrcode
    key = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE: pygame.QUIT()


# run the game loop
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            GetInput()
            sys.exit()
