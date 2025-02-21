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

############ PATHS ##########################################
c_light = 299792.458 # speed of light in km/s
matter = 0.27
darkenergy=0.73
c_light=299792.458#in km/s
H0=70 #km/s/Mpc
zlens = 0.68
#Dd = angdistz0(zlens,matter,darkenergy) # Still need units for this. ASK LILIYA!!!!
Dd = 4.556953033350398e+25*3.24078e-17 # pc
Ds,Dds = 5.52126010407053e+25*3.24078E-17,2.7210841362473157e+25*3.24078E-17
G = 4.3009172706e-3 # pc Msun^-1 (km/s)^2
# ADD BCG
xcent = 7.114 # arcsec
ycent = 4.409 # # arcsec
arcsec_pc = Dd*(1/206264.80624709636) # pc per 1 arcsec
sigcrit = ((c_light**2)/(4*np.pi*G))*(Ds/(Dd*Dds))*(arcsec_pc**2) # Solar Mass per arsec^2

# Bounds of Window
xlow = -10
xhigh = 21
ylow = -10
yhigh = 21

comparisons = [[] for x in range(2)]
choices = [[] for x in range(2)]
for i in range(2):
	srs = input('Sersic? (Yes (w) or No (no)):')
	allrev = input('all or rev :')
	path = '/Users/derek/Desktop/UMN/Research/SDSSJ1004/%ssersic_%sim/'%(srs,allrev)
	choices[i].append((srs,allrev))
	avgmassdens = np.genfromtxt('%s/AVGmassdens.txt'%(path))
	stdmass = np.genfromtxt('%s/STDmassdens.txt'%(path))
	comparisons[i].append(avgmassdens)
fracdiff = (comparisons[0][0]-comparisons[1][0])/comparisons[0][0]
print(choices)

stepx=(xhigh-xlow)/(avgmassdens.shape[0])
stepy=(yhigh-ylow)/(avgmassdens.shape[1])
# x,y = np.meshgrid(np.arange(xlow,xhigh,stepx),np.arange(ylow,yhigh,stepy))
x,y = np.meshgrid(np.linspace(xlow,xhigh,avgmassdens.shape[0]),np.linspace(ylow,yhigh,avgmassdens.shape[1]))

# CREATE PLOT
fig1,ax1=subplots(1,figsize=(17,17),sharex=False,sharey=False,facecolor='w', edgecolor='k')

# avgmassdens = avgmassdens/sigcrit
im = ax1.imshow(fracdiff,extent=(xlow,xhigh,ylow,yhigh),aspect='auto',cmap='seismic',origin='lower',norm=colors.CenteredNorm())
contourmass=ax1.contour(x,y,fracdiff,levels=20,origin='lower',alpha=1)
#contourmass=ax1.contour(x,y,avgmassdens,levels=30,origin='lower',alpha=1)
contour1=ax1.contour(x,y,fracdiff,levels=[-1,1],origin='lower',colors='k')

divider = make_axes_locatable(ax1)
ax1.set_ylabel(r'y [arcsec]',fontsize=15,fontweight='bold')
ax1.set_xlabel(r'x [arcsec]',fontsize=15,fontweight='bold')
cax1=divider.append_axes("right", size="5%",pad=0.25)

cax1.yaxis.set_label_position("right")
cax1.yaxis.tick_right()

fig1.colorbar(im,label=r'Fractional Difference',cax=cax1)
ax1.minorticks_on()

ax1.set_aspect('equal')
ax1.set_anchor('C')
#plt.tight_layout()

ax1.grid(False)

ax1.tick_params(axis='x',labelsize='10')
ax1.tick_params(axis='y',labelsize='10')
ax1.set_title('%s sersic / %s im'%(choices[1][0][0],choices[1][0][1]),fontsize=15,fontweight='bold')

plt.show()