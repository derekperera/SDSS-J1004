import grale.lenses as lenses
import grale.cosmology as cosmology
import grale.images as images
import grale.util as util
import grale.plotutil as plotutil
from grale.constants import *
import matplotlib.pyplot as plt
import numpy as np
import sys
from pylab import *
from scipy import *
import os
import itertools
import math as maths
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tqdm as tqdm
plt.ion()
plt.rcParams['legend.numpoints']=1
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.minor.size'] = 5
pc_per_m = 3.24078e-17 # pc in a m
c_light = 299792.458 # speed of light in km/s
G = 4.3009172706e-3 # pc Msun^-1 (km/s)^2
path = '/Users/derek/Desktop/UMN/Research/SDSSJ1004'

bestgen=[]
for i in np.arange(10,50,10):
	bestgen.append(np.loadtxt('beststeps%s_20Gen.txt'%(i),usecols=0))
bestgen=np.array([int(i) for i in np.concatenate((bestgen))])

cosm = cosmology.Cosmology(0.7, 0.27, 0, 0.73)
cosmology.setDefaultCosmology(cosm)
zd = 0.68
zs1 = 1.734 # QSO
zsA = 3.33
zsB = 2.74
zsC = 3.28
Dd, Ds, Dds = cosm.getAngularDiameterDistance(zd), cosm.getAngularDiameterDistance(zs1), cosm.getAngularDiameterDistance(zd, zs1) # in meters
Ddpc,Dspc,Ddspc = Dd*pc_per_m,Ds*pc_per_m,Dds*pc_per_m # in pc
arcsec_pc = Ddpc*(1/206264.80624709636) # pc per 1 arcsec
arcsec_kpc = arcsec_pc*(1/1000) # kpc per 1 arcsec
sigcrit = ((c_light**2)/(4*np.pi*G))*(Dspc/(Ddpc*Ddspc))*(arcsec_pc**2) # Solar Mass per arsec^2

gridsize=1000
xlow,ylow,xhigh,yhigh = -10,-10,21,21
nx = (xhigh-xlow)/gridsize
ny = (yhigh-ylow)/gridsize
xrang = np.arange(xlow,xhigh,nx)
yrang = np.arange(ylow,yhigh,ny)
thetas = np.array([[x,y] for x in xrang for y in yrang])*ANGLE_ARCSEC
x,y = np.meshgrid(xrang,yrang)

#### ADD TIME DELAY INFO ####
tdelays_obs = np.array([825.99,781.92,0.000,2456.99,-1.000])
xqso = np.array([0.000,-1.317,11.039,8.399,7.197])
yqso = np.array([0.000,3.532,-4.492,9.707,4.603])

imgList = images.readInputImagesFile("%s/J1004points_revised.txt"%path, True) 

for i in range(len(tdelays_obs)):
    imgList[0]["imgdata"].addTimeDelayInfo(i,0,tdelays_obs[i])
imgX = imgList[0]["imgdata"] 

itest = 6
lensX = lenses.GravitationalLens.load("best_step%s_%s.lensdata"%(itest,bestgen[itest-1]))
lensparams = lensX.getLensParameters()
sersic = lensparams[len(lensparams)-1]
factor = sersic['factor']
sersiclens = sersic['lens']
sersicparams = sersiclens.getLensParameters()
qell = sersicparams['q']
n=sersicparams['index']
scaletheta = sersicparams['scale']
sigcent = sersicparams['centraldensity']
massdens = sersiclens.getSurfaceMassDensityMap((xlow*ANGLE_ARCSEC,ylow*ANGLE_ARCSEC),(xhigh*ANGLE_ARCSEC,yhigh*ANGLE_ARCSEC),gridsize+1,gridsize+1)
massdens = massdens*((DIST_KPC**2)/MASS_SUN)*(arcsec_kpc**2) # Solar Mass per square arcsec

mass = 0
for i in range(gridsize):
	for j in range(gridsize):
		mass += massdens[i][j]*(nx**2)
# lpot = sersiclens.getProjectedPotential(Ds,Dds,thetas)
# lpot = lpot.reshape(gridsize,gridsize)

factors=[]
for itest in range(1,41):
	lensX = lenses.GravitationalLens.load("best_step%s_%s.lensdata"%(itest,bestgen[itest-1]))
	lensparams = lensX.getLensParameters()
	sersic = lensparams[len(lensparams)-1]
	factor = sersic['factor']
	factors.append(factor)

sys.exit()

def kappa(y,x):
	magtheta = np.sqrt(x**2 + (y)**2)
	sigma = sigcent*np.exp(-((magtheta/scaletheta)**(1/n)))

	return sigma/sigcrit

def gravfunc(x,y):
	integrand = lambda yprime,xprime: kappa(yprime,xprime)*np.log(np.sqrt((x - xprime)**2 + (y-yprime)**2))
	return (1/np.pi)*integrate.dblquad(integrand,xlow*ANGLE_ARCSEC,xhigh*ANGLE_ARCSEC,ylow*ANGLE_ARCSEC,yhigh*ANGLE_ARCSEC,epsrel=1.5e10)[0]

print(gravfunc(thetas[0][0],thetas[0][1]))
# lpottest = np.array([gravfunc(thetas[i][0],thetas[i][1]) for i in range(len(thetas))])
lpottest=[]
masstest=[]
for i in range(len(thetas)):
	print(i)
	lpottest.append(gravfunc(thetas[i][0],thetas[i][1]))
	masstest.append(kappa(thetas[i][1],thetas[i][0]))
masstest = np.array(masstest)
lpottest = np.array(lpottest)
lpottest = lpottest.reshape(gridsize,gridsize)
masstest = masstest.reshape(gridsize,gridsize)
