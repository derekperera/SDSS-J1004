import sys # system commands
import string as string # string functions4
import math
import numpy as np # numerical tools
from scipy import *
from pylab import *
import os
import itertools
import math as maths
from scipy import integrate
from scipy.stats import distributions,pearsonr,chisquare,norm
from scipy.optimize import curve_fit, minimize,fmin, fmin_powell, root
from scipy import interpolate
from scipy.signal import argrelextrema,argrelmax,argrelmin,find_peaks
from scipy.interpolate import interp1d
from scipy.misc import derivative
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statistics
import random
from sklearn import linear_model, datasets
import uncertainties.unumpy as unumpy 
from uncertainties import ufloat
from tqdm import tqdm
plt.ion()
plt.rcParams['legend.numpoints']=1
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.minor.size'] = 5
rc('font', weight='bold')
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

class sourcenum:
	def __init__(self, Ds, Dds, name):
		self.D_s = Ds # in pc
		self.D_ds = Dds # in pc
		self.name = name

	def get_imagepositions(self):
		if self.name == 'QSO':
			x = np.array([0.000,-1.317,11.039,8.399,7.197])
			y = np.array([0.000,3.532,-4.492,9.707,4.603])
			imagnum = ['A','B','C','D','E']
		if self.name == 'A1':
			x = np.array([3.93,1.22,19.23,18.83,6.83])
			y = np.array([-2.78,19.37,14.67,15.87,3.22])
			imagnum = ["A11", "A12", "A13", "A14", "A15"]
		if self.name == 'A2':
			x = np.array([4.13,1.93,19.43,18.33,6.83])
			y = np.array([-2.68,19.87,14.02,15.72,3.12])
			imagnum = ["A21", "A22", "A23", "A24", "A25"]	
		if self.name == 'A3':	
			x = np.array([4.33,2.73,19.95,18.03,6.83])
			y = np.array([-1.98,20.37,13.04,15.87,3.12])
			imagnum = ["A31", "A32", "A33", "A34", "A35"]	
		if self.name == 'B1':
			x = np.array([8.88,-5.45,8.33])
			y = np.array([-2.16,15.84,2.57])
			imagnum = ["B11", "B12", "B13"]
		if self.name == 'B2':
			x = np.array([8.45,-5.07,8.33])
			y = np.array([-2.26,16.04,2.57])
			imagnum = ["B21", "B22", "B23"]		
		if self.name == 'C1':
			x = np.array([10.25,-7.55,8.49])
			y = np.array([-3.06,15.39,2.72])
			imagnum = ["C11", "C12", "C13"]
		if self.name == 'C2':
			x = np.array([9.95,-7.30,8.49])
			y = np.array([-3.36,15.44,2.72])	
			imagnum = ["C21", "C22", "C23"]	
		return x,y,imagnum

	def get_pathname(self):
		if self.name == 'QSO':
			pathname = 'alphafield'
		else:
			pathname = 'srcfield'
		return pathname

QSO = sourcenum(5.443495877252635E+25*3.24078E-17,2.682759007567776E+25*3.24078E-17,'QSO')
A1 = sourcenum(4.825862211651242E+25*3.24078E-17,3.082708723956462E+25*3.24078E-17,'A1')
A2 = sourcenum(4.825862211651242E+25*3.24078E-17,3.082708723956462E+25*3.24078E-17,'A2')
A3 = sourcenum(4.825862211651242E+25*3.24078E-17,3.082708723956462E+25*3.24078E-17,'A3')
B1 = sourcenum(5.1074447619963185E+25*3.24078E-17,3.089301820360382E+25*3.24078E-17,'B1')
B2 = sourcenum(5.1074447619963185E+25*3.24078E-17,3.089301820360382E+25*3.24078E-17,'B2')
C1 = sourcenum(4.84998617433819E+25*3.24078E-17,3.086468744030152E+25*3.24078E-17,'C1')
C2 = sourcenum(4.84998617433819E+25*3.24078E-17,3.086468744030152E+25*3.24078E-17,'C2')

sources = [QSO,A1,A2,A3,B1,B2,C1,C2]
for s in sources:
	print(s.name)
srcchoice = input('Which Source: ')
for s in sources:
	if srcchoice == s.name:
		src = s

xref,yref,imagnum = src.get_imagepositions()
pathname = src.get_pathname()

def fims(x,y,tdsurf):
	# Function takes in a Time delay surface and finds all the image positions
	# Basic idea is to compute the gradients along x and y and find where they are 0, then find the intersection points

	contourx=ax1.contour(x,y,np.gradient(tdsurf.T)[0],levels=[0],origin='lower',colors='pink')
	contoury=ax1.contour(x,y,np.gradient(tdsurf.T)[1],levels=[0],origin='lower')

	tgradxx=[] # x values for the gradx of tdsurf
	tgradxy=[] # y values for the gradx of tdsurf
	for item in contourx.collections:
   		for i in item.get_paths():
   		 	v = i.vertices
   		 	xcrit = v[:, 0]
   		 	ycrit = v[:, 1]
   		 	tgradxx.append(xcrit)
   		 	tgradxy.append(ycrit)

	tgradyx=[] # x values for the grady of tdsurf
	tgradyy=[] # y values for the grady of tdsurf
	for item in contoury.collections:
   		for i in item.get_paths():
   			v = i.vertices
   			xcrit = v[:, 0]
   			ycrit = v[:, 1]
   			tgradyx.append(xcrit)
   			tgradyy.append(ycrit)

    # The following loop first checks to see if the contours of x and y intersect
    # If they do, then we calculate the intersection points as the minimum distance between
    # the two contours. 
    # A threshold is set at 0.05 arcsec in separation. We will filter this to get 1 solution later
	xint=[]
	yint=[]
	for ix,contxx in tqdm(enumerate(tgradxx)):
		contxy = tgradxy[ix]
	
		min1x = min(contxx)
		max1x = max(contxx)
		min1y = min(contxy)
		max1y = max(contxy)

		for iy,contyy in enumerate(tgradyy):
			contyx = tgradyx[iy]

			min2y = min(contyy)
			max2y = max(contyy)
			min2x = min(contyx)
			max2x = max(contyx)

			if ((min1x > max2x or min2x > max1x) or 
				(min1y > max2y or min2y > max1y)):

				# This means the contours do not overlap!
				continue

			# Finds the smallest distance between contours at each point
			dist=[]
			for i in range(len(contxx)):
				dist.append(min(np.array([np.sqrt((contxx[i]-contyx[j])**2 + (contxy[i] - contyy[j])**2) for j in range(len(contyy))])))

			# Consider the intersection points as the distances less than 0.05 arcsec
			if min(dist) <= 0.5:
				# xint.append(contxx[np.argmin(dist)])
				# yint.append(contxy[np.argmin(dist)])
				intind = [i for i,item in enumerate(dist) if item<=0.05]
				if len(intind)!=0:
					xint.append(contxx[intind])
					yint.append(contxy[intind])

	xint=np.concatenate(xint)
	yint=np.concatenate(yint)

	# Now identify clusters of distances
	# Take the average of the cluster to be the intersection point
	xrec=[]
	yrec=[]
	ptsep = np.sqrt(np.diff(xint)**2 + np.diff(yint)**2)
	ptsep = np.insert(ptsep,len(ptsep) , 10) # Just to make sure the last distance ends the final cluster
	gate=1 # gate=1 means the cluster has ended
	thresh=0.2 # This is the separation between clusters, 0.2 seems to be fine. Change if necessary (i.e. too frequent finding of phantom images)
	for i,sep in enumerate(ptsep):
		if gate==1:
			if sep>thresh: # Cluster has 1 point, the intersection!
				xrec.append(xint[i])
				yrec.append(yint[i])
				gate=1
				continue
			if sep<=thresh: # New cluster starts, flip gate
				xclust=[]
				yclust=[]
				gate=0
		if gate==0: # We are in a cluster
			if sep<=thresh: # Still in a cluster
				xclust.append(xint[i])
				yclust.append(yint[i])
				continue
			if sep>thresh: # Cluster over!
				xclust.append(xint[i])
				yclust.append(yint[i])

				xrec.append(np.mean(xclust))
				yrec.append(np.mean(yclust))
				gate=1

	return xrec,yrec

############ PATHS ##########################################
path = '/Users/derek/Desktop/UMN/Research/SDSSJ1004/run3/'
c_light = 299792.458 # speed of light in km/s
matter = 0.27
darkenergy=0.73
c_light=299792.458#in km/s
H0=70 #km/s/Mpc
zlens = 0.68
#Dd = angdistz0(zlens,matter,darkenergy) # Still need units for this. ASK LILIYA!!!!
Dd = 4.492770596260956e+25*3.24078e-17 # pc
G = 4.3009172706e-3 # pc Msun^-1 (km/s)^2

# SDSSJ1004 Image Position
xcent = 7.197	
ycent = 4.603
xim = np.loadtxt('SDSSJ1004points.txt',usecols=0)
yim = np.loadtxt('SDSSJ1004points.txt',usecols=1)

# Bounds of Window
xlow = -10
xhigh = 21
ylow = -10
yhigh = 21

# Initialize the plot, but we don't care about this output, we just want the contours
fig1,ax1=subplots(1,figsize=(17,17),sharex=False,sharey=False,facecolor='w', edgecolor='k')

xforproj=[]
yforproj=[]
flag=[]

for num in tqdm(range(1,41)):

	xforim,yforim = [],[]

	for imag in imagnum:

		alphax = np.loadtxt('%s/%s_inv%s_%s.txt'%(path,pathname,num,imag),usecols=2)
		alphay = np.loadtxt('%s/%s_inv%s_%s.txt'%(path,pathname,num,imag),usecols=3)
		lpot = np.loadtxt('%s/%s_inv%s_%s.txt'%(path,pathname,num,imag),usecols=4)
		tdsurf = np.loadtxt('%s/%s_inv%s_%s.txt'%(path,pathname,num,imag),usecols=5)

		alphaxfield = np.array(alphax).reshape(int(np.sqrt(len(alphax))),int(np.sqrt(len(alphax))))
		alphayfield = np.array(alphay).reshape(int(np.sqrt(len(alphay))),int(np.sqrt(len(alphay))))
		lpot = np.array(lpot).reshape(int(np.sqrt(len(lpot))),int(np.sqrt(len(tdsurf))))
		tdsurf = np.array(tdsurf).reshape(int(np.sqrt(len(tdsurf))),int(np.sqrt(len(tdsurf))))
		alphafield = np.sqrt(alphaxfield**2 + alphayfield**2)

		stepx=(xhigh-xlow)/alphafield.shape[0]
		stepy=(yhigh-ylow)/alphafield.shape[1]
		x,y = np.meshgrid(np.arange(xlow,xhigh,stepx),np.arange(ylow,yhigh,stepy))

		xrecon,yrecon = fims(x,y,tdsurf)

		if len(xrecon)!=5:
			flag.append([num,imag,len(xrecon)])

		xrec=[]
		yrec=[]
		for i in range(len(xref)):
			distim = np.sqrt((np.array(xrecon)-xref[i])**2 + (np.array(yrecon)-yref[i])**2)
			xrec.append(xrecon[np.argmin(distim)])
			yrec.append(yrecon[np.argmin(distim)])

		xforim.append(xrec)
		yforim.append(yrec)

	xforproj.append(xforim)
	yforproj.append(yforim)

# xforproj[inversion number][qso image within inversion used as source][recon. qso image]
# flag[[inversion number,qso image within inversion,how many reconstructed images instead were found]]

np.save('%s_xforproj'%(src.name),xforproj) # use np.load('run2_xforproj.npy',allow_pickle=True) to reload
np.save('%s_yforproj'%(src.name),yforproj) # use np.load('run2_yforproj.npy',allow_pickle=True) to reload
np.save('%s_flags'%(src.name),flag) # use np.load('run2_flags.npy',allow_pickle=True) to reload