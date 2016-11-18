import scipy.optimize as opt
import math
import numpy as np
import sys
#from UsbControl import LightCrafter6500Device as lc
#import UsbControl.PatternCreation as pc
#import PotetialGeneration.ImagePlanePotentialGenerator as ippg
from PIL import Image as pil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

date = "./20161025/"
Amp = np.genfromtxt(date + 'Ampl.csv', delimiter=',')
Phase = np.genfromtxt(date +'Phaseyx.csv', delimiter=',')


# Create x and y indices
x = np.linspace(0, 20, 21)
y = np.linspace(0, 20, 21)
x, y = np.meshgrid(x, y)

#create data
data = twoD_Gaussian((x, y), 3, 100, 100, 20, 40, 0, 10)

# plot twoD_Gaussian data generated above
plt.figure()
plt.imshow(data.reshape(21, 21))
plt.colorbar()

initial_guess = (3,100,100,20,40,0,10)
data_noisy = data + 0.2*np.random.normal(size=data.shape)
popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data_noisy, p0=initial_guess)

data_fitted = twoD_Gaussian((x, y), *popt)

fig, ax = plt.subplots(1, 1)
ax.hold(True)
ax.imshow(data_noisy.reshape(21, 21), cmap=plt.cm.jet, origin='bottom',
    extent=(x.min(), x.max(), y.min(), y.max()))
ax.contour(x, y, data_fitted.reshape(21, 21), 8, colors='w')
plt.show()


