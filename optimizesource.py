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
from scipy.signal import lfilter
from scipy.interpolate import interp1d,interp2d,RectBivariateSpline
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
############ PATHS ##########################################
srs = input('Sersic? (Yes (w) or No (no)):')
allrev = input('all or rev :')
path = '/Users/derek/Desktop/UMN/Research/SDSSJ1004/%ssersic_%sim/'%(srs,allrev)

clight = 299792.458 # speed of light in km/s
matter = 0.27
darkenergy=0.73
H0=70 #km/s/Mpc
zlens = 0.68
#Dd = angdistz0(zlens,matter,darkenergy) # Still need units for this. ASK LILIYA!!!!
Ddpc = 4.556953033350398e+25*3.24078e-17 # pc
arcsec_pc = Ddpc*(1/206264.80624709636) # pc per 1 arcsec
arcsec_kpc = arcsec_pc*(1/1000) # kpc per 1 arcsec
pc_per_m = 3.24078e-17 # pc in a m
G = 4.3009172706e-3 # pc Msun^-1 (km/s)^2
pcconv = 30856775812800 # Number of km in a pc

# DEFINE ALL SOURCES
class sourcenum:
	def __init__(self, Ds, Dds, name, zs):
		self.D_s = Ds # in pc
		self.D_ds = Dds # in pc
		self.name = name
		self.zs = zs

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

QSO = sourcenum(5.52126010407053e+25*3.24078E-17,2.7210841362473157e+25*3.24078E-17,'QSO',1.734)
A1 = sourcenum(4.894803100389117e+25*3.24078E-17,3.126747420012983e+25*3.24078E-17,'A1',3.33)
A2 = sourcenum(4.894803100389117e+25*3.24078E-17,3.126747420012983e+25*3.24078E-17,'A2',3.33)
A3 = sourcenum(4.894803100389117e+25*3.24078E-17,3.126747420012983e+25*3.24078E-17,'A3',3.33)
B1 = sourcenum(5.180408258596266e+25*3.24078E-17,3.1334347035083875e+25*3.24078E-17,'B1',2.74)
B2 = sourcenum(5.180408258596266e+25*3.24078E-17,3.1334347035083875e+25*3.24078E-17,'B2',2.74)
C1 = sourcenum(4.91927169111445e+25*3.24078E-17,3.1305611546591543e+25*3.24078E-17,'C1',3.28)
C2 = sourcenum(4.91927169111445e+25*3.24078E-17,3.1305611546591543e+25*3.24078E-17,'C2',3.28)

if allrev == 'rev':
	srcchoice = input('Source? (QSO, A2, B1, B2 , C1, C2): ')
	srcname = ['QSO','A2','B1','B2','C1','C2']
	sources = [QSO,A2,B1,B2,C1,C2]
	xim = np.loadtxt('J1004points_revised.txt',usecols=0)
	yim = np.loadtxt('J1004points_revised.txt',usecols=1)
if allrev == 'all':
	srcchoice = input('Source? (QSO, A1, A2, A3, B1, B2 , C1, C2): ')
	srcname = ['QSO','A1','A2','A3','B1','B2','C1','C2']
	sources = [QSO,A1,A2,A3,B1,B2,C1,C2]
	xim = np.loadtxt('SDSSJ1004points.txt',usecols=0)
	yim = np.loadtxt('SDSSJ1004points.txt',usecols=1)

for s in sources:
	if srcchoice == s.name:
		src = s
		xsrc,ysrc,imagnum = s.get_imagepositions()
		Dspc,Ddspc = s.D_s,s.D_ds
		Ds,Dds = Dspc/pc_per_m , Ddspc/pc_per_m # in m 
		zs = s.zs

tdobs = np.array([825.99,781.92,0.00,2456.99,-1.00])
dtobs = np.array([2.10,2.20,1.0,5.55,1.0])
xqso = np.array([0.000,-1.317,11.039,8.399,7.197])
yqso = np.array([0.000,3.532,-4.492,9.707,4.603])

###### LENSING POTENTIAL ######
avglpot = np.genfromtxt('%s/AVGlenspotNULL_%s.txt'%(path,src.name))
# Bounds of Window
xlow = -10
xhigh = 21
ylow = -10
yhigh = 21
stepx=(xhigh-xlow)/avglpot.shape[0]
stepy=(yhigh-ylow)/avglpot.shape[1]
xrang,yrang =np.arange(xlow,xhigh,stepx),np.arange(ylow,yhigh,stepy)
xfield,yfield = np.meshgrid(np.arange(xlow,xhigh,stepx),np.arange(ylow,yhigh,stepy))


##### DEFINE METROPOLIS HASTINGS COMPONENTS ######
fig2,ax2=subplots(1,figsize=(17,17),sharex=False,sharey=False,facecolor='w', edgecolor='k')
def fims(x,y,tdsurf):
	# Function takes in a Time delay surface and finds all the image positions
	# Basic idea is to compute the gradients along x and y and find where they are 0, then find the intersection points
	# x,y are the meshgrid that forms the map for tdsurf

	contourx=ax2.contour(x,y,np.gradient(tdsurf)[0],levels=[0],origin='lower',colors='pink')
	contoury=ax2.contour(x,y,np.gradient(tdsurf)[1],levels=[0],origin='lower')

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

	return np.array(xrec),np.array(yrec)

def prior(params,window): # Priors are flat
    logprior = 0
    xs,ys = params
    xlow,xhigh,ylow,yhigh = window
    
    if ((xs<xlow) or (xs>xhigh)):
        return -np.inf

    if ((ys<ylow) or (ys>yhigh)):
        return -np.inf
    
    return logprior
    
def posterior(params, y, dy, window):
   
	# y: xqso,yqso
	# dy: Uncertainty on observed images positions (0.04 from HST)
	xs,ys = params
	xref,yref = y

    # Prior
	priors = prior(params,window)

    # Calculate model
	thetabeta = ((xfield-xs)/206264.80624709636)*((xfield-xs)/206264.80624709636) + ((yfield-ys)/206264.80624709636)*((yfield-ys)/206264.80624709636)
	avgtdsurf = ((1+zlens)/clight)*((Ddpc*src.D_s)/src.D_ds)*pcconv*(0.5*thetabeta - avglpot.T)
	xrecon,yrecon = fims(xfield,yfield,avgtdsurf)
	# Sort images to make sure reconstructed images are in same order as xqso!
	xrec=[]
	yrec=[]
	for i in range(len(xref)):
		distim = np.sqrt((np.array(xrecon)-xref[i])**2 + (np.array(yrecon)-yref[i])**2)
		xrec.append(xrecon[np.argmin(distim)])
		yrec.append(yrecon[np.argmin(distim)])
	xrec,yrec = np.array(xrec),np.array(yrec)

	if src.name == 'QSO':
		# 2D Interpolation
		tdsurfinterp = RectBivariateSpline(xrang,yrang,avgtdsurf.T)
		tdbase = tdsurfinterp.ev(xrec[2],yrec[2])

		tdelays = []
		for i in range(len(xrec)):
			tdelays.append(tdsurfinterp.ev(xrec[i],yrec[i]) - tdbase)
			#print(tdsurfinterp.ev(xrecqso[i],yrecqso[i]))
		tdelays = np.array(tdelays)/86400 # in days
		td_obs = np.array([tdobs[0],tdobs[1],0.0,tdobs[3],0.0])
		td_pred = np.array([tdelays[0],tdelays[1],0.0,tdelays[3],0.0])

    # Add likelihood to prior to get posterior
	distref = np.sqrt(xref**2 + yref**2)
	distrec = np.sqrt(xrec**2 + yrec**2)
	if src.name == 'QSO':
		#likelihood = -0.5*sum((((distref - distrec)**2)/(dy**2)) + (((td_obs - td_pred)**2)/(dtobs**2)))   # Likelihood is a Gaussian
		likelihood = -0.5*sum((((xref - xrec)**2)/(dy**2)) + (((yref - yrec)**2)/(dy**2)) + (((td_obs - td_pred)**2)/(dtobs**2)))
	else:
		#likelihood = -0.5*sum((((distref - distrec)**2)/(dy**2)))   # Likelihood is a Gaussian
		#likelihood = -0.5*sum((((xref - xrec)**2)/(dy**2)) + (((yref - yrec)**2)/(dy**2)))
		likelihood = -0.5*sum((((xref - xrec)**2)/(dy**2)) + (((yref - yrec)**2)/(dy**2)) + (len(xrecon)-len(xref))**2)

	logpost = likelihood + priors
    
	return logpost

def metropolis(sampsize, initial_state, sigma,y,dy,window):

	n_params = len(initial_state)
    
	#trace = np.empty((sampsize+1,n_params))
	trace = [[] for x in range(sampsize+1)]

	trace[0] = initial_state[0] # Set parameters to the initial state
	logprob = posterior(trace[0],y,dy,window) # Compute p(x)

	accepted = [0]*n_params

	for i in tqdm(range(sampsize)): # while we want more samples
		iterparams = trace[i] # Current parameters in the trace
		print(iterparams)
        
		for j in range(n_params):
			# draw x' from the proposal distribution (a gaussian)
			iterparams_j = trace[i].copy()
			#print(iterparams_j)
			xprime = norm.rvs(loc=iterparams[0],scale=sigma[j][0],size=1)[0]
			yprime = norm.rvs(loc=iterparams[1],scale=sigma[j][1],size=1)[0]
			iterparams_j = [xprime,yprime]
			#print(iterparams_j)
			
            
			# compute p(x')
			logprobprime = posterior(iterparams_j,y,dy,window)

			alpha = logprobprime - logprob
            # Draw uniform from uniform distribution
			u = np.random.uniform(0,1,1)[0]
            
            # Test sampled value
			if np.log(u) < alpha:
				#trace[i+1,j] = xprime
				trace[i+1] = iterparams_j
				logprob = logprobprime
				accepted[j] += 1
			else:
				#trace[i+1,j] = trace[i,j]
				trace[i+1] = trace[i]
                
	return np.array(trace),np.array(accepted)

##########################################################
##########################################################
#################### DO METROPOLIS HASTINGS ##############
##########################################################
##########################################################
n_iter = 100
xs,ys = 8.194598007056307 , 7.095721482558924 # See plottdsurf.py for mean source positions
initstate = [[xs,ys]]
#sigma = [[(priorwindow[1]-priorwindow[0]),(priorwindow[3]-priorwindow[2])]]
# priorwindow = [min(np.concatenate(xs)),max(np.concatenate(xs)),min(np.concatenate(ys)),max(np.concatenate(ys))] # Run fullbackproj.py to get priorwindow (use grale)
priorwindow = [7.7527272658845 , 8.37342623231681 , 4.643658054290711 , 9.958260479074779]
sigma = [[0.04,0.04]]
dyobs=np.zeros(len(xsrc))+0.04
trace,accept = metropolis(n_iter,initstate,sigma,[xsrc,ysrc],dyobs,priorwindow)
print(path)

# Source Positions from Optimization
#### QSO Metro-Hastings Results ####
# xs,ys = 5.62159173, 4.5743219 # wsersic revim (improvement)
# xs,ys = 5.74578679, 4.19416386 # no sersic revim (not an improvement, but already within HST)
# xs,ys = 5.97831416, 4.40308503 # no sersic allim (improved images, basically same time delays)
# xs,ys = 5.81322872, 4.60682808 # wsersic allim (improved images, slightly worse time delays)

#### A1 Results ####
# xs,ys = 7.25697756, 8.4309567 # no sersic allim (improve but flag)
# xs,ys = 7.2025612 , 9.43195203 # wsersic allim (improve)

#### A2 Results ####
# xs,ys = 6.94558686, 9.49335285 # wsersic revim (not an improvement, idk why)
# xs,ys = 6.72875861, 8.05040377 # no sersic revim (not an improvement, but already within HST)
# xs,ys = 7.21901549, 8.48207073 # no sersic allim (improve but flag)
# xs,ys = 7.15417576, 9.47228136 # wsersic allim (improve)

#### A3 Results ####
# xs,ys = 7.21744043, 8.52807292 # no sersic allim (improve)
# xs,ys = 7.15095389, 9.47420873 # wsersic allim (improve)

#### B1 Results ####
# xs,ys = 3.72665182, 8.45679172 # wsersic revim (improve)
# xs,ys = 3.9218521 , 8.00040714 # no sersic revim (improve)
# xs,ys = 4.01989588, 8.21218535 # no sersic allim (improve)
# xs,ys = 3.86813248, 8.44355717 # wsersic allim (improve)

#### B2 Results ####
# xs,ys = 3.87188621, 8.50857302 # wsersic revim (improve)
# xs,ys = 4.07951843, 8.01867869 # no sersic revim (improve)
# xs,ys = 4.15000556, 8.25220558 # no sersic allim (improve)
# xs,ys = 4.00761459, 8.47037831 # wsersic allim (improve)

#### C1 Results ####
# xs,ys = 3.11245139, 8.54323987 # wsersic revim (improve)
# xs,ys = 3.31405609, 7.99671827 # no sersic revim (improve)
# xs,ys = 3.44612895, 8.28093611 # no sersic allim (improve)
# xs,ys = 3.24195532, 8.57364596 # wsersic allim (improve)

#### C2 Results ####
# xs,ys = 3.21141385, 8.55618059 # wsersic revim (improve)
# xs,ys = 3.43932842, 7.97548338 # no sersic revim (improve)
# xs,ys = 3.53095603, 8.31528038 # no sersic allim (improve)
# xs,ys = 3.34321407, 8.58766979 # wsersic allim (improve)