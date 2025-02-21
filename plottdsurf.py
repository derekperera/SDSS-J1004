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
from matplotlib import ticker,cm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statistics
import random
from sklearn import linear_model, datasets
import uncertainties.unumpy as unumpy 
from uncertainties import ufloat
from tqdm import tqdm
from shapely import geometry
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
sigcrit = 117122551738.13449 # Solar Mass per arsec^2

def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian

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
srcchoice = input('Source? (QSO, A2, B1, B2 , C1, C2): ')

QSO = sourcenum(5.52126010407053e+25*3.24078E-17,2.7210841362473157e+25*3.24078E-17,'QSO',1.734)
A1 = sourcenum(4.894803100389117e+25*3.24078E-17,3.126747420012983e+25*3.24078E-17,'A1',3.33)
A2 = sourcenum(4.894803100389117e+25*3.24078E-17,3.126747420012983e+25*3.24078E-17,'A2',3.33)
A3 = sourcenum(4.894803100389117e+25*3.24078E-17,3.126747420012983e+25*3.24078E-17,'A3',3.33)
B1 = sourcenum(5.180408258596266e+25*3.24078E-17,3.1334347035083875e+25*3.24078E-17,'B1',2.74)
B2 = sourcenum(5.180408258596266e+25*3.24078E-17,3.1334347035083875e+25*3.24078E-17,'B2',2.74)
C1 = sourcenum(4.91927169111445e+25*3.24078E-17,3.1305611546591543e+25*3.24078E-17,'C1',3.28)
C2 = sourcenum(4.91927169111445e+25*3.24078E-17,3.1305611546591543e+25*3.24078E-17,'C2',3.28)
srcname = ['QSO','A2','B1','B2','C1','C2']
sources = [QSO,A1,A2,A3,B1,B2,C1,C2]
for s in sources:
	if srcchoice == s.name:
		src = s
		xref,yref,imagnum = s.get_imagepositions()
		Dspc,Ddspc = s.D_s,s.D_ds
		Ds,Dds = Dspc/pc_per_m , Ddspc/pc_per_m # in m 
		zs = s.zs
for i,item in enumerate(srcname):
	if src.name == item:
		jsrc = i

# xim = np.loadtxt('J1004points_revised.txt',usecols=0)
# yim = np.loadtxt('J1004points_revised.txt',usecols=1)
xim = np.loadtxt('SDSSJ1004points.txt',usecols=0)
yim = np.loadtxt('SDSSJ1004points.txt',usecols=1)

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
	thresh=0.07 # This is the separation between clusters, 0.2 seems to be fine. Change if necessary (i.e. too frequent finding of phantom images)
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

def fims2(x,y,tdsurf):
	# Function takes in a Time delay surface and finds all the image positions
	# Basic idea is to compute the gradients along x and y and find where they are 0, then find the intersection points
	# x,y are the meshgrid that forms the map for tdsurf

	contourx=ax2.contour(x,y,np.gradient(tdsurf)[0],levels=[0],origin='lower',colors='pink')
	contoury=ax2.contour(x,y,np.gradient(tdsurf)[1],levels=[0],origin='lower')

	tgradxx=[] # x values for the gradx of tdsurf
	tgradxy=[] # y values for the gradx of tdsurf
	verticesx=[]
	for item in contourx.collections:
		for i in item.get_paths():
			v = i.vertices
			verticesx.append(v)
			xcrit = v[:, 0]
			ycrit = v[:, 1]
			tgradxx.append(xcrit)
			tgradxy.append(ycrit)

	tgradyx=[] # x values for the grady of tdsurf
	tgradyy=[] # y values for the grady of tdsurf
	verticesy=[]
	for item in contoury.collections:
		for i in item.get_paths():
			v = i.vertices
			verticesy.append(v)
			xcrit = v[:, 0]
			ycrit = v[:, 1]
			tgradyx.append(xcrit)
			tgradyy.append(ycrit)

	xrec,yrec = [],[]
	for vx in verticesx:
		for i,vy in enumerate(verticesy):
			polyx = geometry.LineString(vx)
			polyy = geometry.LineString(vy)
			try:
				intersection = polyx.intersection(polyy)
				xrec.append(np.array([float(intersection.geoms[i].x) for i in range(len(intersection.geoms))]))
				yrec.append(np.array([float(intersection.geoms[i].y) for i in range(len(intersection.geoms))]))
			except AttributeError: # Lines don't intersect!
				continue

			
	return np.concatenate(np.array(xrec)),np.concatenate(np.array(yrec))

tdobs = np.array([825.99,781.92,0.00,2456.99,-1.00])
xqso = np.array([0.000,-1.317,11.039,8.399,7.197])
yqso = np.array([0.000,3.532,-4.492,9.707,4.603])

avglpot = np.genfromtxt('%s/AVGlenspot_%s.txt'%(path,src.name))
# avgdeta = np.genfromtxt('%s/AVGdetA_%s.txt'%(path,src.name))
# avgdeta = avgdeta.T
# avglpot = np.genfromtxt('%s/AVGlenspot_A.txt'%(path))
# avglpot = np.genfromtxt('%s/AVGlenspot_QSO1000.txt'%(path))

# Bounds of Window
xlow = -10
xhigh = 21
ylow = -10
yhigh = 21
stepx=(xhigh-xlow)/avglpot.shape[0]
stepy=(yhigh-ylow)/avglpot.shape[1]
xrang,yrang =np.arange(xlow,xhigh,stepx),np.arange(ylow,yhigh,stepy)
x,y = np.meshgrid(np.arange(xlow,xhigh,stepx),np.arange(ylow,yhigh,stepy))

xs,ys = 5.97831416, 4.40308503
# Mean Source Positions from BackProjecting
#### QSO Metro-Hastings Results ####
# xs,ys = 5.622198033024151,4.5739307409491445 # wsersic revim
# xs,ys = 5.749015855715751, 4.195245540977685 # no sersic revim
# xs,ys = 5.9800581470757646, 4.404827165829795 # no sersic allim 
# xs,ys = 5.8147364853828485,4.607978040213 # wsersic allim
# xs,ys = 5.926607662911028 , 4.686413152702048 # wsersic SECim
# Best reproduced time delays: no sersic and all images
# Best reproduced images: with sersic and revised images

#### A1 ##### 
# xs,ys = 7.25415783044848,8.425702719367916 # no sersic allim
# xs,ys = 7.218735585617262,9.419197276440189 # wsersic allim
# xs,ys = 7.161333041318158 , 9.330755493288367 # wsersic SECim

#### A2 ####
# xs,ys = 6.958680222237737,9.501191143668656 # wsersic revim (2 extra)
# xs,ys = 6.723361141639646,8.059062749150801 # no sersic revim (4 extra)
# xs,ys = 7.205186973750094,8.49630072959274 # no sersic allim (6 extra)
# xs,ys = 7.148544517051755,9.475903990259173 # wsersic allim
# xs,ys = 7.091261674989174 , 9.358703457909495 # wsersic SECim

#### A3 ##### 
# xs,ys = 7.230013504072972,8.542037986473103 # no sersic allim
# xs,ys = 7.152742826524689,9.474674169065539 # wsersic allim  
# xs,ys = 7.073555657117504 , 9.327865247855453 # wsersic SECim

#### B1 ####
# xs,ys = 3.75893645251819,8.48012706890018 # wsersic revim
# xs,ys = 3.949215454954421, 7.999332696154761 # no sersic revim
# xs,ys = 4.025935648923096,8.23234019568104 # no sersic allim
# xs,ys = 3.882773015877312,8.468883286946497 # wsersic allim (2 extra)
# xs,ys = 4.014614882714272 , 8.60659873633885 # wsersic SECim

#### B2 ####
# xs,ys = 3.8514484906323174,8.497250367148846 # wsersic revim
# xs,ys = 4.053458178734305, 8.018180284450214 # no sersic revim
# xs,ys = 4.133318868806904,8.24653754492616 # no sersic allim
# xs,ys = 3.988569990600779,8.481673947185866 # wsersic allim (2 extra)

#### C1 ####
# xs,ys = 3.1014647178992285,8.568088557271771 # wsersic revim (2 extra)
# xs,ys = 3.324762801802125, 7.9883365899485455 # no sersic revim
# xs,ys = 3.4405280858991647,8.311419131392702 # no sersic allim
# xs,ys = 3.2316947714507487,8.598254434353171 # wsersic allim (2 extra)

#### C2 ####
# xs,ys = 3.200233711761816,8.538359496126294 # wsersic revim (2 extra)
# xs,ys = 3.4200077709069556, 7.959280933719373 # no sersic revim
# xs,ys = 3.5311933980122707,8.289947367802016 # no sersic allim
# xs,ys = 3.3270774925060076,8.573420840957537 # wsersic allim (2 extra)

thetabeta = ((x-xs)/206265.0)*((x-xs)/206265.0) + ((y-ys)/206265.0)*((y-ys)/206265.0)
avgtdsurf = ((1+zlens)/clight)*((Ddpc*Dspc)/Ddspc)*pcconv*(0.5*thetabeta - avglpot.T)

xrecon,yrecon = fims2(x,y,avgtdsurf)
plt.scatter(xrecon,yrecon)
# Sort images to make sure reconstructed images are in same order as xqso!
xrec=[]
yrec=[]
for i in range(len(xref)):
	distim = np.sqrt((np.array(xrecon)-xref[i])**2 + (np.array(yrecon)-yref[i])**2)
	xrec.append(xrecon[np.argmin(distim)])
	yrec.append(yrecon[np.argmin(distim)])
xrec,yrec = np.array(xrec),np.array(yrec)

distrecon = np.sqrt((np.array(xrec)-xref)**2 + (np.array(yrec)-yref)**2)
print('Mean Image Separation:',np.mean(distrecon))

if src.name == 'QSO':
	# 2D Interpolation
	tdsurfinterp = RectBivariateSpline(xrang,yrang,avgtdsurf.T)
	tdbase = tdsurfinterp.ev(xrec[2],yrec[2])

	tdelays = []
	for i in range(len(xrec)):
		tdelays.append(tdsurfinterp.ev(xrec[i],yrec[i]) - tdbase)
		#print(tdsurfinterp.ev(xrecqso[i],yrecqso[i]))
	tdelays = np.array(tdelays)/86400 # in days

	tdsep = np.array([abs(tdelays[0]-tdobs[0]),abs(tdelays[1]-tdobs[1]), abs(tdelays[3]-tdobs[3])])
	print('Mean Time Delay Separation: ',np.mean(tdsep))

# detA = np.genfromtxt('%s/AVGdetA.txt'%(path))
# detA = detA.T
# detcontour=ax2.contour(x,y,detA,levels=[0],origin='lower',colors='green')
#xs,ys = np.meshgrid(xs,ys)
#contour2 = ax1.contour(xs,ys,SISell,levels=[0],origin='lower',colors='purple',linestyles='dashed')

# xcrit,ycrit=[],[]
# for item in detcontour.collections:
#    for i in item.get_paths():
#       v = i.vertices
#       xcr = v[:, 0]
#       ycr = v[:, 1]
#       xcrit.append(xcr)
#       ycrit.append(ycr)

magnmatrix = hessian(avgtdsurf)
deta = magnmatrix[0][0]*magnmatrix[1][1] - magnmatrix[1][0]*magnmatrix[0][1]
plt.imshow(deta,extent=(xlow,xhigh,ylow,yhigh),aspect='auto',cmap='seismic',origin='lower',norm=colors.CenteredNorm())
plt.contour(x,y,deta,levels=[0])
plt.scatter(xrec,yrec,color='g')
avgmassdens = np.genfromtxt('%s/AVGmassdens.txt'%(path))
stdmassdens = np.genfromtxt('%s/STDmassdens.txt'%(path))
avgmassdens = avgmassdens/sigcrit
stdmassdens = stdmassdens/sigcrit
avgkap = (Dds/Ds)*np.genfromtxt('%s/AVGkapp_QSO.txt'%(path)) 
avggam1 = (Dds/Ds)*np.genfromtxt('%s/AVGgam1_QSO.txt'%(path)) 
avggam2 = (Dds/Ds)*np.genfromtxt('%s/AVGgam2_QSO.txt'%(path)) 
stdkap = (Dds/Ds)*np.genfromtxt('%s/STDkapp_QSO.txt'%(path)) 
stdgam1 = (Dds/Ds)*np.genfromtxt('%s/STDgam1_QSO.txt'%(path)) 
stdgam2 = (Dds/Ds)*np.genfromtxt('%s/STDgam2_QSO.txt'%(path)) 

convshear = [avgkap.T,avggam1.T,avggam2.T,avgmassdens]
errparams = [stdkap.T,stdgam1.T,stdgam2.T,stdmassdens]
paramlabel = ['kappa','gamma1','gamma2','sig/sigcrit']
print(imagnum)
for i,param in enumerate(convshear):
	surfinterp = RectBivariateSpline(xrang,yrang,param)
	errinterp = RectBivariateSpline(xrang,yrang,errparams[i])
	paramest = surfinterp.ev(xrec,yrec)
	errparamest = errinterp.ev(xrec,yrec)
	print('%s'%paramlabel[i], paramest)
	# print('err', '%s'%paramlabel[i], errparamest)


# CREATE PLOT
fig1,ax1=subplots(1,figsize=(17,17),sharex=False,sharey=False,facecolor='w', edgecolor='k')
# im = ax1.imshow(avgtdsurf,extent=(xlow,xhigh,ylow,yhigh),aspect='auto',cmap='YlOrRd',origin='lower')
# im = ax1.imshow(avgmassdens,extent=(xlow,xhigh,ylow,yhigh),aspect='auto',cmap='YlOrRd',origin='lower',vmin=0.0,vmax=11.0)
im = ax1.imshow(np.log10(avgmassdens),extent=(xlow,xhigh,ylow,yhigh),aspect='auto',cmap='YlOrRd',origin='lower',vmin=-1.0,vmax=1.04)
# levels = np.linspace(-6e9,-5.4e9,50) # w all C1
# levels = np.linspace(-6.5e9,-5.954e9,50) # no all C1
# contour=ax1.contour(x,y,avgtdsurf,levels=500,origin='lower')
contour=ax1.contour(x,y,np.log10(avgmassdens),levels=50,origin='lower',alpha=0.3)

divider = make_axes_locatable(ax1)
ax1.set_ylabel(r'y [arcsec]',fontsize=15,fontweight='bold')
ax1.set_xlabel(r'x [arcsec]',fontsize=15,fontweight='bold')
cax1=divider.append_axes("right", size="5%",pad=0.25)

cax1.yaxis.set_label_position("right")
cax1.yaxis.tick_right()

# fig1.colorbar(im,label=r'Surface Mass Density [M$_{\odot}$ arcsec$^{-2}$]',cax=cax1)
fig1.colorbar(im,label=r'log$_{10}$($\Sigma / \Sigma_{crit}$)',cax=cax1)
ax1.minorticks_on()

def region(regionx,regiony,regionsize):

	# regionx is x coordinate of center of circular region in arcsec
	# regiony is y coordinate of center of circular region in arcsec
	# regionsize is radius of circular region in arcesec

	dist = np.sqrt((x-regionx)**2 + (y-regiony)**2)
	densreginit = np.array([avgmassdens[i][j]*(stepx**2) for i in range(dist.shape[0]) for j in range(dist.shape[1]) if dist[i][j] <= regionsize])
	regpos = np.array([(i,j) for i in range(dist.shape[0]) for j in range(dist.shape[1]) if dist[i][j] <= regionsize])
	for i,item in enumerate(densreginit):
		if item == max(densreginit):
			xregpos,yregpos = regpos[i][0],regpos[i][1]
			break
	xclump = x[xregpos][yregpos] # Peak of Clump x
	yclump = y[xregpos][yregpos] # Peak of Clump y

	# xclump,yclump = regionx,regiony
	phi = np.linspace(0, 2*np.pi, 100)
	x1 = regionsize*np.cos(phi) + xclump
	x2 = regionsize*np.sin(phi) + yclump
	ax1.errorbar(x1,x2,color='g',label='South-East Clump')
	# ax1.errorbar([regionx],[regiony],color='b',marker='*')
	# ax1.errorbar([xclump],[yclump],color='g',marker='*')

	dist = np.sqrt((x-xclump)**2 + (y-yclump)**2)
	densregion = np.array([(avgmassdens[i][j]-1.2e11)*(stepx**2) for i in tqdm(range(dist.shape[0])) for j in range(dist.shape[1]) if dist[i][j] <= regionsize])
	massregion = sum(densregion)

	return xclump,yclump,massregion
xclump,yclump,massclump = region(5.7,-1.6,1.5)

# ax1.scatter(xim,yim,color='r',label='Galaxy Images')
ax1.scatter(xref,yref,color='lime',label='Observed',zorder=2)
ax1.scatter(xrecon,yrecon,color='purple',marker='^',label='Reconstructed',zorder=2)
# ax1.scatter([xs],[ys],color='b',marker='*',label='Source')
ax1.set_aspect('equal')
ax1.set_anchor('C')
ax1.grid(False)
ax1.tick_params(axis='x',labelsize='10')
ax1.tick_params(axis='y',labelsize='10')
# for i in range(len(xcrit)):
# 	ax1.errorbar(xcrit[i],ycrit[i],color='green')
ax1.contour(x,y,deta,levels=[0],colors='b',linestyles='dashed')
ax1.legend(loc=1,title='',ncol=1,prop={'size':8})
ax1.set_xlim(1.1,13.0)
ax1.set_ylim(-3.2,8.0)

plt.show()

psix = np.gradient(avglpot.T)[0]
psiy = np.gradient(avglpot.T)[1]
psixx = np.gradient(psix)[0]
psiyy = np.gradient(psiy)[1]
psixy = np.gradient(psix)[1]
psiyx = np.gradient(psiy)[0]

# Idk where the factor comes from but it is needed
kap = 0.5*(psixx + psiyy)
gam1 = 0.5*(psixx - psiyy)
gam2 = psixy
massinterp = RectBivariateSpline(np.linspace(xlow,xhigh,avgmassdens.shape[0]),np.linspace(ylow,yhigh,avgmassdens.shape[1]),avgmassdens)
kapinterp = RectBivariateSpline(xrang,yrang,kap)
mref=massinterp.ev(np.array(xref),np.array(yref))
kref=kapinterp.ev(np.array(xref),np.array(yref))
factor = np.mean(mref/kref)
kap = kap*factor
gam1 = gam1*factor
gam2 = gam2*factor
detmag = (1.0-kap)*(1.0-kap) - (gam1**2) - (gam2**2)