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

QSO = sourcenum(5.52126010407053e+25*3.24078E-17,2.7210841362473157e+25*3.24078E-17,'QSO')
A1 = sourcenum(4.894803100389117e+25*3.24078E-17,3.126747420012983e+25*3.24078E-17,'A1')
A2 = sourcenum(4.894803100389117e+25*3.24078E-17,3.126747420012983e+25*3.24078E-17,'A2')
A3 = sourcenum(4.894803100389117e+25*3.24078E-17,3.126747420012983e+25*3.24078E-17,'A3')
B1 = sourcenum(5.180408258596266e+25*3.24078E-17,3.1334347035083875e+25*3.24078E-17,'B1')
B2 = sourcenum(5.180408258596266e+25*3.24078E-17,3.1334347035083875e+25*3.24078E-17,'B2')
C1 = sourcenum(4.91927169111445e+25*3.24078E-17,3.1305611546591543e+25*3.24078E-17,'C1')
C2 = sourcenum(4.91927169111445e+25*3.24078E-17,3.1305611546591543e+25*3.24078E-17,'C2')

path = '/Users/derek/Desktop/UMN/Research/SDSSJ1004/wsersic_revim'
c_light = 299792.458 # speed of light in km/s
matter = 0.27
darkenergy=0.73
c_light=299792.458#in km/s
H0=70 #km/s/Mpc
zlens = 0.68
#Dd = angdistz0(zlens,matter,darkenergy) # Still need units for this. ASK LILIYA!!!!
Dd = 4.556953033350398e+25*3.24078e-17 # pc
G = 4.3009172706e-3 # pc Msun^-1 (km/s)^2

# SDSSJ1004 Image Position
xcent = 7.197	
ycent = 4.603
xim = np.loadtxt('SDSSJ1004points.txt',usecols=0)
yim = np.loadtxt('SDSSJ1004points.txt',usecols=1)

sources = [QSO,A1,A2,A3,B1,B2,C1,C2]
srcname = ['QSO','A2','B1','B2','C1','C2']
for s in sources:
	print(s.name)
srcchoice = input('Which Source: ')
for s in sources:
	if srcchoice == s.name:
		src = s
for i,item in enumerate(srcname):
	if src.name == item:
		jsrc = i

xref,yref,imagnum = src.get_imagepositions()
pathname = src.get_pathname()

##########################################################
# xforproj = np.load('%s_xforproj.npy'%(src.name),allow_pickle=True)
# yforproj = np.load('%s_yforproj.npy'%(src.name),allow_pickle=True)
# flag = np.load('%s_flags.npy'%(src.name),allow_pickle=True) 

xforproj = np.load('%s/xforwproj.npy'%(path),allow_pickle=True)
yforproj = np.load('%s/yforwproj.npy'%(path),allow_pickle=True)

#########################################################
################ CREATE PLOT #################################
fig1,ax1=subplots(1,sharex=False,sharey=False,facecolor='w', edgecolor='k')
colours=['blue','green','red','black','sienna']
for i in range(len(imagnum)):
	ax1.scatter(xref[i],yref[i],color=colours[i],label=imagnum[i],marker='^')

# Run 3 version
# xforproj[inversion number][qso image within inversion used as source][recon. qso image]
# flag[[inversion number,qso image within inversion,how many reconstructed images instead were found]]

# Sersic Version
# xforproj[inversion number][source number][reconstructed image number][reconstructed image]

for num in tqdm(range(len(xforproj))): # INVERSION NUMBER

	for i in range(len(imagnum)):

		ax1.scatter(xforproj[num][jsrc].T[i],yforproj[num][jsrc].T[i],color=colours[i],alpha=0.5)
		# ax1.scatter(xforproj[num][2][i],yforproj[num][2][i],color=colours[i],alpha=0.5) # run 3

ax1.set_aspect('equal')
ax1.set_anchor('C')
#plt.tight_layout()

ax1.grid(False)

ax1.tick_params(axis='x',labelsize='10')
ax1.tick_params(axis='y',labelsize='10')

ax1.legend(loc=3,title='',ncol=2,prop={'size':10})

plt.show()
sys.exit()

avglpot = np.zeros((1000,1000))

for num in tqdm(range(1,41)):
	lpot = np.loadtxt('%s/%s_inv%s_%s.txt'%(path,pathname,num,imagnum[0]),usecols=4) # Uses 1 image reconstructions, doesn't matter since lpot is same for each inversion
	lpot = np.array(lpot).reshape(int(np.sqrt(len(lpot))),int(np.sqrt(len(lpot))))
	avglpot += lpot

avglpot = avglpot/40
np.savetxt('%s/AVGlpotplummers_%s.txt'%(path,src.name),avglpot) # Use np.genfromtxt to get array back!
