import numpy as np
from struct import *
import matplotlib.pyplot as plt


def imread(path):
	f=open(path,'rb')
	header = []
	header.append(f.read(4)) #fileid
	header.append(unpack('H',f.read(2))) #heddersize
	header.append(unpack('H',f.read(2)))
	header.append(unpack('H',f.read(2)))
	header.append(unpack('H',f.read(2)))
	header.append(unpack('H',f.read(2)))
	header.append(unpack('H',f.read(2)))
	header.append(unpack('H',f.read(2)))
	header.append(unpack('H',f.read(2)))
	header.append(f.read(40)) #name
	header.append(f.read(100))
	header.append(unpack('i',f.read(4)))
	header.append(unpack('i',f.read(4)))
	header.append(unpack('h',f.read(2)))
	header.append(unpack('h',f.read(2)))
	header.append(unpack('h',f.read(2)))
	header.append(unpack('h',f.read(2)))
	header.append(unpack('i',f.read(4)))

	data = np.zeros((1,header[5][0]-header[3][0]+1,header[6][0]-header[4][0]+1))
	formatting = ''
	for j in xrange(header[5][0]-header[3][0]+1):
		formatting += 'H'
	for i in xrange(header[6][0]-header[4][0]+1):
		data[0,i,:]=np.array(unpack(formatting,f.read(2*(header[5][0]-header[3][0]+1))))-header[16][0]
	return  header, data


def readdatas(datanum):

	filenum = 5
	header = []

	for i in xrange(datanum):
		for j in xrange(filenum):
			if np.logical_and(i==0,j==0):
				head,data = imread("./160208/data0"+str(i+10)+"/update00"+str(j+1)+".pmi")	
				header.append(head)
			else:
				head,dat = imread("./160208/data0"+str(i+10)+"/update00"+str(j+1)+".pmi")
				header.append(head)
				data = np.append(data,dat,axis=0)
	return data

def readdata(datanum,imgnum):
	head,data = imread("./160208/data0"+str(datanum)+"/update00"+str(imgnum)+".pmi")
	return data


def showfound(data,op):
	image = np.zeros((512,512,3),dtype=np.float32)
	color = 1.*(data-1*np.amin(data))/(np.amax(data)-np.amin(data))
	image[:,:,0] = color
	image[:,:,1] = 1.*np.logical_not(op)*color
	image[:,:,2] = 1.*np.logical_not(op)*color
	plt.imshow(image,interpolation="None")

