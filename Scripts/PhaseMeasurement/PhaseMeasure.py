import sys
sys.path.append('../../Library/')
import time
from UsbControl import LightCrafter6500Device as lc
import UsbControl.PatternCreation as pc
import PotetialGeneration.ImagePlanePotentialGenerator as ippg
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imsave, imread
import scipy.ndimage as ndi
import numpy as np

#usage: >> python phase_Measure.py x1 y1 Phase Size GratingSize


#check if arguments are four integers
def tryinteger(str):
    try:
        int(str)
        return 0
    except ValueError:
        return 1

if len(sys.argv)!=6:
    print "arguments error: must be five integers x1 y1 Phase Size GratingSize"
elif tryinteger(sys.argv[1]):
    print "arguments error: 1st arg not integer"
elif tryinteger(sys.argv[2]):
    print "arguments error: 2st arg not integer"
elif tryinteger(sys.argv[3]):
    print "arguments error: 3st arg not integer"
elif tryinteger(sys.argv[4]):
    print "arguments error: 4st arg not integer"
elif tryinteger(sys.argv[5]):
    print "arguments error: 5st arg not integer"
else:
    lc_dmd = lc.LC6500Device()

#prepare patterns       
    settings = {'function':pc.tilt_phase_measure_fun,'compression':'rle','exposure_time':500000,'X1':int(sys.argv[1]),'Y1':int(sys.argv[2]),'Phase' : int(sys.argv[3]), 'Size':int(sys.argv[4]), 'GratingSize':int(sys.argv[5])}


    dmd_pattern = pc.DMDPattern(**settings)
    dmd_pattern.compute_pattern()

    lc_dmd.upload_image(dmd_pattern)
#    dmd_pattern.show_pattern()
