import pmi2array
import numpy as np
import cv2
from skimage.morphology import disk
from skimage.morphology import opening
from skimage.filters import threshold_otsu, threshold_adaptive
import matplotlib.pyplot as plt





def showthres(iteration):
	data = pmi2array.readdatas(int(np.floor(iteration/3)))
	##adaptive threasholding
	for i in xrange(iteration):
		thres = threshold_adaptive(data[i], 33, offset=-100)
		op = opening(thres,disk(1))
		plt.figure(i)
		pmi2array.showfound(data[i],op) 


def thresholding(datanum,imgnum): ##datanum should be more than 10
	data = pmi2array.readdata(datanum,imgnum)
	thres = threshold_adaptive(data[i], 33, offset=-100)
	op = opening(thres,disk(1))
	return op


if __name__ == '__main__':
	showthres(10)
	plt.show()







