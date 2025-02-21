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
clight = 299792.458 # speed of light in km/s
matter = 0.27
darkenergy=0.73
H0=70 #km/s/Mpc
zlens = 0.68
#Dd = angdistz0(zlens,matter,darkenergy) # Still need units for this. ASK LILIYA!!!!
Ddpc = 4.556953033350398e+25*3.24078e-17 # pc
Dspc,Ddspc = 5.52126010407053e+25*3.24078E-17,2.7210841362473157e+25*3.24078E-17
arcsec_pc = Ddpc*(1/206264.80624709636) # pc per 1 arcsec
arcsec_kpc = arcsec_pc*(1/1000) # kpc per 1 arcsec
pc_per_m = 3.24078e-17 # pc in a m
G = 4.3009172706e-3 # pc Msun^-1 (km/s)^2
pcconv = 30856775812800 # Number of km in a pc
sigcrit = ((clight**2)/(4*np.pi*G))*(Dspc/(Ddpc*Ddspc))*(arcsec_pc**2) # Solar Mass per arsec^2

hubparam2 = (H0**2)*(matter*((1+zlens)**3) + darkenergy)
critdens = ((3*hubparam2)/(8*np.pi*G))*(1/(10**12)) # solar mass per pc^3

def NFW(params,r):
	# Input array r of pc distances
	# Return Solar Mass per pc^2 if r is given in pc
	r_s,c = params 

	deltac = (200/3)*((c**3)/(np.log(1+c) - (c/(1+c))))
	xdim = r/r_s

	sigmanfw = []
	for i in range(len(xdim)):
		if xdim[i] == 1:
			sigmanfw.append((2*r_s*deltac*critdens)/3)
		if xdim[i] < 1:
			sigmanfw.append(((2*r_s*deltac*critdens)/(xdim[i]**2 - 1))*(1 - (2/np.sqrt(1 - xdim[i]**2))*np.arctanh(np.sqrt((1-xdim[i])/(1+xdim[i])))))
		if xdim[i] > 1:
			sigmanfw.append(((2*r_s*deltac*critdens)/(xdim[i]**2 - 1))*(1 - (2/np.sqrt(xdim[i]**2 - 1))*np.arctan(np.sqrt((xdim[i]-1)/(1+xdim[i])))))

	return np.array(sigmanfw)

def meanNFW(params,r):
	# Input array r of pc distances
	# Return Solar Mass per pc^2 if r is given in pc
	r_s,c = params 

	deltac = (200/3)*((c**3)/(np.log(1+c) - (c/(1+c))))
	xdim = r/r_s

	sigmanfw = []
	for i in range(len(xdim)):
		if xdim[i] == 1:
			sigmanfw.append((4*r_s*deltac*critdens)*(1+np.log(0.5)))
		if xdim[i] < 1:
			sigmanfw.append(((4*r_s*deltac*critdens)/(xdim[i]**2))*((2/np.sqrt(1 - xdim[i]**2))*np.arctanh(np.sqrt((1-xdim[i])/(1+xdim[i]))) + np.log(xdim[i]/2)))
		if xdim[i] > 1:
			sigmanfw.append(((4*r_s*deltac*critdens)/(xdim[i]**2))*((2/np.sqrt(xdim[i]**2 - 1))*np.arctan(np.sqrt((xdim[i]-1)/(1+xdim[i]))) + np.log(xdim[i]/2)))

	return np.array(sigmanfw)

def chi2(params,r,sig,err_sig):
	r_s,c = params 
	return sum(((sig - NFW([r_s,c],r))/(np.array(err_sig)))**2)


fig4,ax4=subplots(1,sharex=False,sharey=False,facecolor='w', edgecolor='k')
fig4.subplots_adjust(hspace=0)

############ PATHS ##########################################
circproftot=[]
for permutation in range(1):
	srs = input('Sersic? (Yes (w) or No (no)):')
	allrev = input('all or rev :')
	path = '/Users/derek/Desktop/UMN/Research/SDSSJ1004/%ssersic_%sim/'%(srs,allrev)

	##### SURFACE MASS DENSITY MAP #####
	avgmassdens = np.genfromtxt('%s/AVGmassdens.txt'%(path))
	# Bounds of Window
	xlow = -10
	xhigh = 21
	ylow = -10
	yhigh = 21
	stepx=(xhigh-xlow)/(avgmassdens.shape[0])
	stepy=(yhigh-ylow)/(avgmassdens.shape[1])
	x,y = np.meshgrid(np.linspace(xlow,xhigh,avgmassdens.shape[0]),np.linspace(ylow,yhigh,avgmassdens.shape[1]))

	##### CIRCULAR AVERAGE MASS DENSITY #####
	### Circular average about the BCG? ####
	nbins = 150
	distbins = [[] for x in range(nbins)] # Bins of circular profile
	xmapcent,ymapcent = np.mean([xlow,xhigh]),np.mean([ylow,yhigh])
	binvals = np.linspace(1e-14,xhigh-xmapcent,nbins) # Values of bins
	xbcg = 7.114 # arcsec
	ybcg = 4.409 # # arcsec
	for i in tqdm(range(len(x))):
		for j in range(len(y)):
			xpix = x[i][j]
			ypix = y[i][j]
			massdenspix = avgmassdens[i][j]
			dist = np.sqrt(((xpix-xbcg)**2 + (ypix-ybcg)**2)) # Distance from pixel to BCG

			if (dist>max(binvals)):
				distbins[nbins-1].append(massdenspix)
				continue

			for k in range(len(binvals)-1):
				if (dist>binvals[k] and dist<=binvals[k+1]):
					distbins[k].append(massdenspix)

	circprof = np.array([np.mean(np.array(bins)) for bins in distbins])
	std_circprof = np.array([np.std(np.array(bins)) for bins in distbins])

	circproftot.append(circprof)
	########## PLOT ##################

	logsigma = np.log10(circprof)
	logr = np.log10(binvals)

	colors = ['b','r','k','y']

	ax4.errorbar(binvals,circprof,color=colors[permutation],linestyle='-',label='%s sersic / %s im.'%(srs,allrev))
	# ax4.errorbar(logr,logsigma,color=colors[permutation],linestyle='-',label='%s sersic / %s im.'%(srs,allrev))
	# y += np.random.normal(0, 0.1, size=y.shape)
	ax4.fill_between(binvals, circprof-std_circprof, circprof+std_circprof,color=colors[permutation],alpha=0.5)

circprof = circprof/(arcsec_pc**2) # Solar mass per pc^2
std_circprof = std_circprof/(arcsec_pc**2)

scalerad,cvals = [],[]
for i in tqdm(range(500)):
	cprof = np.array([np.random.normal(circprof[j],std_circprof[j]) for j in range(len(circprof))])
	if srs == 'no':
		res = minimize(chi2,[39*arcsec_pc,6.1],args=(binvals[1:]*arcsec_pc,cprof[1:],std_circprof[1:]),method='Nelder-Mead',options = {'maxiter':10000})
		scalerad.append(res.x[0])
		cvals.append(res.x[1])
	if srs == 'w':
		res = minimize(chi2,[39*arcsec_pc,6.1],args=(binvals*arcsec_pc,cprof,std_circprof),method='Nelder-Mead',options = {'maxiter':10000})
		scalerad.append(res.x[0])
		cvals.append(res.x[1])
scalerad,cvals = np.array(scalerad),np.array(cvals)
signfw = NFW([np.mean(scalerad),np.mean(cvals)],binvals*arcsec_pc) # Solar mass per pc^2
signfw = signfw*arcsec_pc*arcsec_pc # Solar mass per arcsec^2
ax4.errorbar(binvals,signfw,color='r',linestyle='--',label='Best Fit NFW')
# ax4.errorbar(binvals,NFW([39*arcsec_pc,6.1],binvals*arcsec_pc)*arcsec_pc*arcsec_pc,color='r',linestyle='--')

print('R_s: ',np.mean(scalerad)/arcsec_pc,'+/-',np.std(scalerad)/arcsec_pc)
print('c: ',np.mean(cvals),'+/-',np.std(cvals))
print(np.mean(scalerad),np.std(scalerad))
R200 = ufloat(np.mean(scalerad),np.std(scalerad))*ufloat(np.mean(cvals),np.std(cvals))
print(R200)
M200 = (4/3)*np.pi*200*critdens*(R200**3)

ax4.set_xlabel(r'r [arcsec]',fontsize=15,fontweight='bold')
ax4.set_ylabel(r'$\Sigma$ [M$_{\odot}$ arcsec$^{-2}$]',fontsize=15,fontweight='bold')

# ax4.set_xscale('log')
# ax4.set_yscale('log')

ax4.legend(loc=0,title='',ncol=1,prop={'size':15})
ax4.minorticks_on()

ax4.set_ylim(bottom=0.0, top=max(np.concatenate(circproftot))-100)
# ax4.set_xlim(left=-3.0, right=None)

plt.show()
