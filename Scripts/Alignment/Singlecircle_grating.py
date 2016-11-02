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

lc_dmd = lc.LC6500Device()

#prepare patterns       

settings = {'function':pc.tilt_phase_align_fun,'compression':'rle','exposure_time':500000}


dmd_pattern = pc.DMDPattern(**settings)
#    dmd_pattern.compute_pattern(
X=dmd_pattern.X
Y=dmd_pattern.Y
dmd_pattern.pattern = 1*((X<270)&(X>-270)&(Y<270)&(Y>-270)&((X+Y)%12 > 10))

lc_dmd.upload_image(dmd_pattern)
dmd_pattern.show_pattern()






