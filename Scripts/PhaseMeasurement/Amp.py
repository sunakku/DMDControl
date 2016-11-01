import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##import data
#p0 = np.genfromtxt('dat_oversample.csv', delimiter=',')
#p2 = np.genfromtxt('dat2_oversample.csv', delimiter=',')
#p4 = np.genfromtxt('dat3_oversample.csv', delimiter=',')
path = "../Data/PhaseMeasurement/"
p0 = np.genfromtxt(path + 'ampdat.csv', delimiter=',')
p2 = np.genfromtxt(path + 'ampdat2.csv', delimiter=',')
p4 = np.genfromtxt(path + 'ampdat3.csv', delimiter=',')

#delete 1st axes
#p0=np.delete(p0,0,0)
#p0=np.delete(p0,0,1)
#p2=np.delete(p2,0,0)
#p2=np.delete(p2,0,1)
#p4=np.delete(p4,0,0)
#p4=np.delete(p4,0,1)
# np.delete(p0,np.s_[:1],1)

#PD offset
#p0 = p0 + offset
#p2 = p2 + offset
#p4 = p4 + offset

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
print Preal
print Pimag
xsize = np.shape(Phase)[0]
ysize = np.shape(Phase)[1]

#replace referenced patch with average
#pathup0 = np.unwrap(np.append(xaxis_unwrap[xsize/2-1],Phase[xsize/2-1,ysize/2+1:ysize]))
#pathup1 = np.unwrap(np.append(xaxis_unwrap[xsize/2+2],Phase[xsize/2-1,ysize/2+1:ysize]))
#pathup2 = np.unwrap(np.append(xaxis_unwrap[xsize/2],Phase[xsize/2-1,ysize/2+1:ysize]))
#pathup1 = np.unwrap(Phase[xsize/2+2,ysize/2+1:ysize])
#pathup2 = np.unwrap(Phase[xsize/2,ysize/2+1:ysize])
#Phase[xsize/2-1,ysize/2+2]=(Phase[xsize/2-2,ysize/2+2]+Phase[xsize/2-1,ysize/2+1])/2
#Phase[xsize/2,ysize/2+2]=(Phase[xsize/2-1,ysize/2+2]+Phase[xsize/2+2,ysize/2+2])/2
#Phase[xsize/2+1,ysize/2+2]=(Phase[xsize/2-1,ysize/2+2]+Phase[xsize/2+2,ysize/2+2])/2
#Phase[xsize/2,ysize/2]=(Phase[xsize/2-1,ysize/2+2]+Phase[xsize/2+2,ysize/2+2])/2
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

Amp[4,5]=0.009
Amp[3,5]=0.01
Amp[4,6]=0.007
Phase[4,6]=0.6


np.savetxt(path + "Ampl.csv", Amp, delimiter=",")
np.savetxt(path + "phase_residual.csv", Phase, delimiter=",")
#np.savetxt("Phasexy.csv", Unwrappedxy, delimiter=",")
#np.savetxt("Phaseyx.csv", Unwrappedyx, delimiter=",")
#np.savetxt("Phase_amp.csv", Unwrap_center, delimiter=",")
#np.savetxt("Phase2_amp.csv", Unwrap_center2, delimiter=",")


fig1=plt.figure(1,(15,10))
#plt.imshow(Phase,interpolation='nearest')
plt.subplot(2,3,1)
plt.imshow(Phase)
#plt.title('wrap')
plt.title('residual phase')
plt.colorbar()

plt.subplot(2,3,2)
plt.imshow(Unwrap_center)
plt.title('unwrap center x')
plt.colorbar()

plt.subplot(2,3,3)
plt.imshow(Amp)
plt.title('Amp')
plt.colorbar()

plt.subplot(2,3,4)
plt.imshow(Unwrappedxy)
plt.title('unwrap xy')
plt.colorbar()

plt.subplot(2,3,5)
plt.imshow(Unwrappedyx)
plt.title('unwrap yx')
plt.colorbar()

plt.subplot(2,3,6)
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
ax.plot_surface(X, Y, Amp, rstride=1, cstride=1)
plt.show()
print Phase
