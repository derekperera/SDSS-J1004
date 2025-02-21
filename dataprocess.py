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

path = '/Users/derek/Desktop/UMN/Research/SDSSJ1004/nosersic_allim' #CHANGE PATH NAME FOR DIFFERENT RUNS
# bestgen=[11 ,18 ,14 ,14, 9, 19 ,8 ,16, 17, 20, 15 ,20 ,17 ,20, 9 ,19 ,9 ,10, 10, 10, 16, 12, 9, 9,
#  20, 14 ,20, 19, 18, 17, 20, 10, 10, 12, 19, 20, 15 ,18, 8, 18] ### run3/

bestgen=[]
for i in np.arange(10,50,10):
	bestgen.append(np.loadtxt('%s/beststeps%s_20Gen.txt'%(path,i),usecols=0))
bestgen=np.array([int(i) for i in np.concatenate((bestgen))])
# bestgen = np.loadtxt('%s/beststeps4_NULL20Gen.txt'%(path),usecols=0)

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

gridsize=500
xlow,ylow,xhigh,yhigh = -10,-10,21,21
nx = (xhigh-xlow)/gridsize
ny = (yhigh-ylow)/gridsize
xrang = np.arange(xlow,xhigh,nx)
yrang = np.arange(ylow,yhigh,ny)
thetas = np.array([[x,y] for x in xrang for y in yrang])*ANGLE_ARCSEC
x,y = np.meshgrid(xrang,yrang)
xs,ys = 5.622198033024152, 4.5739307409491445

#### ADD TIME DELAY INFO ####
tdelays_obs = np.array([825.99,781.92,0.000,2456.99,-1.000])
xqso = np.array([0.000,-1.317,11.039,8.399,7.197])
yqso = np.array([0.000,3.532,-4.492,9.707,4.603])

# imgList = images.readInputImagesFile("J1004points_revised.txt", True) 
imgList = images.readInputImagesFile("SDSSJ1004points.txt", True) 


for i in range(len(tdelays_obs)):
    imgList[0]["imgdata"].addTimeDelayInfo(i,0,tdelays_obs[i])
imgX = imgList[0]["imgdata"] 

##### GET ALL PLUMMERS FROM RUN ########
plummermass,plummerwidth,plummerx,plummery = [],[],[],[]
avgmassdens = np.zeros((gridsize,gridsize))
avgtdelay = np.zeros((gridsize,gridsize))
avglpot = np.zeros((gridsize,gridsize))
avgdetA= np.zeros((gridsize,gridsize))
avgkap,avggam1,avggam2 = np.zeros((gridsize,gridsize)),np.zeros((gridsize,gridsize)),np.zeros((gridsize,gridsize))
massobj,tdelayobj,lpotobj,detaobj = [],[],[],[]
kapobj,gam1obj,gam2obj = [],[],[],

for i in range(1,2): # 1,41
	lensX = lenses.GravitationalLens.load("%s/best_step%s_%s.lensdata"%(path,i,bestgen[i-1]))
	# lensX = lenses.GravitationalLens.load("%s/invNULLPRIORITY%s_%s.lensdata"%(path,i,int(bestgen[i-1])))
	# lensX = lenses.GravitationalLens.load("%s/correctedlens_%s.lensdata"%(path,i))
	# lensX = lenses.GravitationalLens.load("%s/eq_model-N_1024-gpw_0.0001-dw_0.1-s_scs.lensdata"%(path))


	# # Average Surface Mass Density (kg/m^2)
	# surfdensplum = lensX.getSurfaceMassDensityMap((xlow*ANGLE_ARCSEC,ylow*ANGLE_ARCSEC),(xhigh*ANGLE_ARCSEC,yhigh*ANGLE_ARCSEC),gridsize+1,gridsize+1)
	# avgmassdens += surfdensplum*((DIST_KPC**2)/MASS_SUN)*(arcsec_kpc**2) # Solar Mass per square arcsec
	# # massobj.append(surfdensplum*((DIST_KPC**2)/MASS_SUN)*(arcsec_kpc**2))
	print('Run: %s'%i)

	# Average Time Delay Surface (s?) For the QSO 
	# tdelayX = lensX.getTimeDelay(zd,Ds,Dds,thetas,np.array([xs,ys])*ANGLE_ARCSEC)
	# tdsurf = tdelayX.reshape(gridsize,gridsize)
	# avgtdelay += tdsurf
	# tdelayobj.append(tdsurf)

	# # Average Lens Potential
	lpotX = lensX.getProjectedPotential(Ds,Dds,thetas)
	lpot = lpotX.reshape(gridsize,gridsize)
	# lpot = 0.5*(((x-xs)*ANGLE_ARCSEC)**2 + ((y-ys)*ANGLE_ARCSEC)**2) - lpot
	avglpot += lpot
	lpotobj.append(lpot)

	# # Average Shear 
	# alphavec=lensX.getAlphaVectorDerivatives(thetas)
	# alphaxx,alphayy,alphaxy = alphavec.T[0].reshape(gridsize,gridsize),alphavec.T[1].reshape(gridsize,gridsize),alphavec.T[2].reshape(gridsize,gridsize)
	# gamma1 = 0.5*(alphaxx-alphayy)
	# gamma2 = alphaxy
	# # gammasquared = gamma1**2 + gamma2**2
	# kappa = 0.5*(alphaxx+alphayy)
	# avgkap += kappa
	# avggam1 += gamma1
	# avggam2 += gamma2
	# # deta = (1-kappa)**2 - gammasquared
	# # avgdetA += deta
	# kapobj.append(kappa)
	# gam1obj.append(gamma1)
	# gam2obj.append(gamma2)


	# # Average Magnification
	# detaX = lensX.getInverseMagnification(Ds, Dds, thetas)
	# deta = detaX.reshape(gridsize,gridsize)
	# avgdetA += deta
	# detaobj.append(deta)
sys.exit()
# 	# Get All Plummers! (Optional I think)
# 	lensparams = lensX.getLensParameters()
# 	massplum,widthplum,xplum,yplum = [],[],[],[]
# 	for j in range(len(lensparams)-1):
# 		plummerparams = lensparams[j]['lens'].getLensParameters()
# 		massplum.append(plummerparams['mass']*lensparams[j]['factor'])
# 		widthplum.append(plummerparams['width'])
# 		xplum.append(lensparams[j]['x'])
# 		yplum.append(lensparams[j]['y'])
# 	plummermass.append(np.array(massplum)/MASS_SUN) # Mass in Solar Mass
# 	plummerwidth.append(np.array(widthplum)/ANGLE_ARCSEC) # Width in Arcsec
# 	plummerx.append(np.array(xplum)/ANGLE_ARCSEC) # x plummer in Arcsec
# 	plummery.append(np.array(yplum)/ANGLE_ARCSEC) # y plummer in Arcsec
# plummermass,plummerwidth,plummerx,plummery = np.array(plummermass),np.array(plummerwidth),np.array(plummerx),np.array(plummery)

# massdens = lensX.getSurfaceMassDensity(np.array([21,21])*ANGLE_ARCSEC)*((DIST_KPC**2)/MASS_SUN)*(arcsec_kpc**2)

# avgmassdens = avgmassdens/4
# avgtdelay = avgtdelay/40
# avglpot = avglpot/4
# avgdetA = avgdetA/40
avgkap = avgkap/40
avggam1 = avggam1/40
avggam2 = avggam2/40

# massobj,tdelayobj,lpotobj = np.array(massobj),np.array(tdelayobj),np.array(lpotobj)
# stdmass,stdtdelay,stdlpot = massobj.std(axis=0),tdelayobj.std(axis=0),lpotobj.std(axis=0) # Standard Deviation at all points in the field
kapobj,gam1obj,gam2obj = np.array(kapobj),np.array(gam1obj),np.array(gam2obj)
stdkap,stdgam1,stdgam2 = kapobj.std(axis=0),gam1obj.std(axis=0),gam2obj.std(axis=0)

# massobj = np.array(massobj)
# stdmass = massobj.std(axis=0)
# lpotobj = np.array(lpotobj)
# stdlpot = lpotobj.std(axis=0)
# detaobj = np.array(detaobj)
# stddeta = detaobj.std(axis=0)

# np.savetxt('%s/AVGmassdensSMOOTH1024.txt'%(path),avgmassdens) # Use np.genfromtxt to get array back!
# np.savetxt('%s/STDmassdens.txt'%(path),stdmass) # Use np.genfromtxt to get array back!
# np.savetxt('%s/AVGtimedelaytest.txt'%(path),avgtdelay) # Use np.genfromtxt to get array back!
# np.savetxt('%s/STDtimedelaytest.txt'%(path),stdtdelay) # Use np.genfromtxt to get array back!
# np.savetxt('%s/AVGlenspotSMOOTH1024_QSO.txt'%(path),avglpot) # Use np.genfromtxt to get array back!
# np.savetxt('%s/STDlenspot_B.txt'%(path),stdlpot) # Use np.genfromtxt to get array back!
# np.savetxt('%s/AVGdetA_QSO.txt'%(path),avgdetA) # Use np.genfromtxt to get array back!
# np.savetxt('%s/STDdetA_B.txt'%(path),stddeta) # Use np.genfromtxt to get array back!

np.savetxt('%s/AVGkapp_QSO.txt'%(path),avgkap) # Use np.genfromtxt to get array back!
np.savetxt('%s/AVGgam1_QSO.txt'%(path),avggam1) # Use np.genfromtxt to get array back!
np.savetxt('%s/AVGgam2_QSO.txt'%(path),avggam2) # Use np.genfromtxt to get array back!
np.savetxt('%s/STDkapp_QSO.txt'%(path),stdkap) # Use np.genfromtxt to get array back!
np.savetxt('%s/STDgam1_QSO.txt'%(path),stdgam1) # Use np.genfromtxt to get array back!
np.savetxt('%s/STDgam2_QSO.txt'%(path),stdgam2) # Use np.genfromtxt to get array back!