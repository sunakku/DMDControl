import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

def gauss(x, *p):
    A, mu, sigma = p
    return A*numpy.exp(-(x-mu)**2/(2.*sigma**2))

path = "../Data/PhaseMeasurement/"
p0 = np.genfromtxt(path + 'dat.csv', delimiter=',')
p2 = np.genfromtxt(path + 'dat2.csv', delimiter=',')
p4 = np.genfromtxt(path + 'dat3.csv', delimiter=',')

#offset negative values to zero
p0 = p0*1*(p0>0)
p2 = p2*1*(p2>0)
p4 = p4*1*(p4>0)


#Phaser p
Preal = -(p2+p4-2*p0)/6
Pimag = -(1/(2*math.sqrt(3)))*(p2-p4)
Phaser = Preal + 1j*Pimag
Phase = np.angle(Phaser)
Amp = np.absolute(Phaser)
Phase_data = Phase
xsize = np.shape(Phase)[0]
ysize = np.shape(Phase)[1]

#replace referenced patch with average
#pathup0 = np.unwrap(np.append(xaxis_unwrap[xsize/2-1],Phase[xsize/2-1,ysize/2+1:ysize]))
#pathup1 = np.unwrap(np.append(xaxis_unwrap[xsize/2+2],Phase[xsize/2-1,ysize/2+1:ysize]))
#pathup2 = np.unwrap(np.append(xaxis_unwrap[xsize/2],Phase[xsize/2-1,ysize/2+1:ysize]))
#pathup1 = np.unwrap(Phase[xsize/2+2,ysize/2+1:ysize])
#pathup2 = np.unwrap(Phase[xsize/2,ysize/2+1:ysize])
#Phase[xsize/2+1,ysize/2-1]=(Phase[xsize/2+1,ysize/2-2]+Phase[xsize/2+1,ysize/2-1])/2
Phase[xsize/2+1,ysize/2]=(Phase[xsize/2+1,ysize/2-1]+Phase[xsize/2+1,ysize/2+2])/2
Phase[xsize/2+1,ysize/2+1]=(Phase[xsize/2+1,ysize/2-1]+Phase[xsize/2+1,ysize/2+2])/2
Phase[xsize/2,ysize/2+2]=(Phase[xsize/2+1,ysize/2+2]+Phase[xsize/2-1,ysize/2+2])/2
Phase[xsize/2,ysize/2+3]=(Phase[xsize/2+1,ysize/2+3]+Phase[xsize/2-1,ysize/2+3])/2
#Phase[6,0]=0.8#(Phase[0,5]+Phase[0,7])/2
#Phase[10,4]=0.8
#Phase[1,2]=0.8
#Phase[xsize/2,ysize/2]=(Phase[xsize/2-1,ysize/2+1]+Phase[xsize/2+1,ysize/2-1])/2
#Phase[xsize/2,ysize/2]=(Phase[xsize/2-1,ysize/2+2]+Phase[xsize/2+2,ysize/2+2])/2




#Unwrap
Unwrappedx = np.unwrap(Phase,math.pi,1)
Unwrappedy = np.unwrap(Phase,math.pi,0)
Unwrappedxy = np.unwrap(Unwrappedx,math.pi,0)
Unwrappedyx = np.unwrap(Unwrappedy,math.pi,1)
Unwrappedxyx = np.unwrap(Unwrappedxy,math.pi,1)
Unwrappedyxy = np.unwrap(Unwrappedyx,math.pi,0)


#unwrap from center, like this
# ^ ^ ^ ^ ^
# ^ ^ ^ ^ ^
# < < o > >
# v v v v v
# v v v v v

#extract paths

#pathsUpper = np.zeros(np.shape(Phase)[0]/2+1,
#    for y in xrange(0,np.shape(Phase)[0]):


#unwrap y=0 axis from center
#[::-1] is to take inverse 
xaxis_unwrap_up = np.unwrap(Phase[xsize/2:xsize,ysize/2]) 
xaxis_unwrap_down = np.unwrap(Phase[0:xsize/2+1,ysize/2][::-1])[::-1] 
xaxis_unwrap = np.append(xaxis_unwrap_down,np.delete(xaxis_unwrap_up,0,0))


#prepare unwrapped phase array
Unwrap_center = np.zeros(np.shape(Phase))

for x in xrange(0,xsize):

    pathdown = Phase[x,0:ysize/2][::-1]
    pathup = Phase[x,ysize/2+1:ysize]
    #append wrapped x axis value and unwrap
    pathdown = np.unwrap(np.append(xaxis_unwrap[x],pathdown))[::-1]
    pathup = np.unwrap(np.append(xaxis_unwrap[x],pathup))
    #add to Unwrapped phase array
    Unwrap_center[x,:] = np.append(pathdown,np.delete(pathup,0,0))
Unwrapcx = np.unwrap(Unwrap_center,math.pi,0)

#unwrap x=0 axis from center
#[::-1] is to take inverse 
yaxis_unwrap_up = np.unwrap(Phase[xsize/2,ysize/2:ysize]) 
yaxis_unwrap_down = np.unwrap(Phase[xsize/2,0:ysize/2+1][::-1])[::-1] 
yaxis_unwrap = np.append(yaxis_unwrap_down,np.delete(yaxis_unwrap_up,0,0))

#prepare unwrapped phase array
Unwrap_center2 = np.zeros(np.shape(Phase))

for y in xrange(0,ysize):

    pathdown = Phase[0:xsize/2,y][::-1]
    pathup = Phase[xsize/2+1:xsize,y]
    #append wrapped y axis value and unwrap
    pathdown = np.unwrap(np.append(yaxis_unwrap[x],pathdown))[::-1]
    pathup = np.unwrap(np.append(yaxis_unwrap[x],pathup))
    #add to Unwrapped phase array
    Unwrap_center2[:,y] = np.append(pathdown,np.delete(pathup,0,0))


##interpolate
#Unwrap_center[xsize/2+1,ysize/2]=(Unwrap_center[xsize/2,ysize/2]+Unwrap_center[xsize/2+2,ysize/2])/2
#Unwrap_center[xsize/2+1,ysize/2+1]=(Unwrap_center[xsize/2+1,ysize/2]+Unwrap_center[xsize/2+2,ysize/2+1])/2


#gaus fit
#p0 = [1., 0., 1.]
#coeff, var_matrix = curve_fit(gauss, bin_centres, Unwrap_center, p0=p0)
# Get the fitted curve
#hist_fit = gauss(bin_centres, *coeff)
#plt.plot(bin_centres, hist, label='Test data')
#plt.plot(bin_centres, hist_fit, label='Fitted data')

fourier = np.fft.fftshift(np.fft.fft2(Unwrappedxy))
fourier[:,8] = 0
fourier[8,:] = 0
Unwrappedxy_low = np.real(np.fft.ifft2(np.fft.fftshift(fourier)))



np.savetxt(path + "Phasec1.csv", Unwrap_center, delimiter=",")
np.savetxt(path + "Phasec2.csv", Unwrap_center2, delimiter=",")
np.savetxt(path + "Phasexy.csv", Unwrappedxy, delimiter=",")
np.savetxt(path + "Phaseyx.csv", Unwrappedyx, delimiter=",")
#np.savetxt("Phase_amp.csv", Unwrap_center, delimiter=",")
#np.savetxt("Phase2_amp.csv", Unwrap_center2, delimiter=",")


fig1=plt.figure(1,(20,10))
#plt.imshow(Phase,interpolation='nearest')
plt.subplot(2,4,1)
plt.imshow(Phase)
#plt.title('wrap')
plt.title('phase')
plt.colorbar()

plt.subplot(2,4,2)
plt.imshow(Unwrap_center)
plt.title('unwrap center x')
plt.colorbar()

plt.subplot(2,4,3)
plt.imshow(Unwrap_center2)
plt.title('unwrap center y')
plt.colorbar()

plt.subplot(2,4,4)
plt.imshow(Unwrappedxy)
plt.title('unwrap xy')
plt.colorbar()

plt.subplot(2,4,5)
plt.imshow(Unwrappedyx)
plt.title('unwrap yx')
plt.colorbar()

plt.subplot(2,4,6)
plt.imshow(Unwrapcx)
plt.title('phase center + x')
plt.colorbar()

plt.subplot(2,4,7)
plt.imshow(Unwrappedxy_low)
plt.title('low pass xy')
plt.colorbar()

plt.subplot(2,4,8)
plt.imshow(Phase_data)
plt.title('phase original')
plt.colorbar()
#plot x=0
#plt.subplot(3,3,4)
#plt.plot(Phase[0,:])
#plt.plot(Phase[1,:])
#plt.plot(Phase[2,:])
#plt.plot(Phase[3,:])
#plt.plot(Phase[4,:])
#plt.plot(Phase[5,:])
#plt.plot(Phase[6,:])
#plt.plot(Phase[7,:])
#plt.plot(Phase[8,:])
#plt.title('wrap, x=0')

#plt.subplot(3,3,5)
#plt.plot(Unwrap_center2[0,:])
#plt.plot(Unwrap_center2[1,:])
#plt.plot(Unwrap_center2[2,:])
#plt.plot(Unwrap_center2[3,:])
#plt.plot(Unwrap_center2[4,:])
#plt.plot(Unwrap_center2[5,:])
#plt.plot(Unwrap_center2[6,:])
#plt.plot(Unwrap_center2[7,:])
#plt.plot(Unwrap_center2[8,:])
#plt.title('unwrap from center x x=0')

#plt.subplot(3,3,6)
#plt.plot(Unwrap_center[0,:])
#plt.plot(Unwrap_center[1,:])
#plt.plot(Unwrap_center[2,:])
#plt.plot(Unwrap_center[3,:])
#plt.plot(Unwrap_center[4,:])
#plt.plot(Unwrap_center[5,:])
#plt.plot(Unwrap_center[6,:])
#plt.plot(Unwrap_center[7,:])
#plt.plot(Unwrap_center[8,:])
#plt.title('unwrap from center y  x=0')

#plot y=0
#plt.subplot(3,3,7)
#plt.plot(Phase[:,3])
#plt.title('wrap, y=0')

#plt.subplot(3,3,8)
#plt.plot(Unwrap_center2[:,3])
#plt.title('unwrap from center x  y=0')

#plt.subplot(3,3,9)
#plt.plot(Unwrap_center[:,3])
#plt.title('unwrap from center y y=0')
plt.show()

fig = plt.figure(2)
ax = Axes3D(fig)
x = np.arange(-(xsize/2)*50, (xsize/2+1)*50,50)
y = np.arange(-(ysize/2)*50, (ysize/2+1)*50,50)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, Unwrap_center, rstride=1, cstride=1)
plt.show()
