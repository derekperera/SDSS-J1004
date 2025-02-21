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
rc('font', weight='bold')
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
############ PATHS ##########################################
srs = input('Sersic? (Yes (w) or No (no)):')
allrev = input('all or rev :')
path = '/Users/derek/Desktop/UMN/Research/SDSSJ1004/%ssersic_%sim/'%(srs,allrev)
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
pcconv = 30856775814914 # km per 1 pc
syear = 365*24*60*60 # seconds in 1 year
hubparam2 = (H0**2)*(matter*((1+zlens)**3) + darkenergy)
critdens = ((3*hubparam2)/(8*np.pi*G))*(1/(10**12)) # solar mass per pc^3
xgal = np.array([30.78,12.14,2.76,25.29,-9.22,14.54,24.61,9.36,2.767,14.789,-1.359,12.00,7.84,-7.21])
ygal = np.array([4.50,3.67,14.13,-9.06,-2.53,24.23,4.72,2.41,-0.171,-5.454,0.482,13.82,9.10,-8.84])
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
			x = np.array([3.93,1.33,19.23,18.83,6.83])
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

# ADD SIS LENS
xcent = 7.114 # arcsec
ycent = 4.409 # # arcsec
# veldisp = 352 # km/s
arcsec_pc = Dd*(1/206264.80624709636) # pc per 1 arcsec
arcsec_kpc = arcsec_pc*(1/1000) # kpc per 1 arcsec
sigcrit = ((c_light**2)/(4*np.pi*G))*(Ds/(Dd*Dds))*(arcsec_pc**2) # Solar Mass per arsec^2

# Bounds of Window
xlow = -10
xhigh = 21
ylow = -10
yhigh = 21

avgmassdens = np.genfromtxt('%s/AVGmassdens.txt'%(path))
# avgtotmassdens = np.genfromtxt('%s/AVGmassdenstotal.txt'%(path))
stdmass = np.genfromtxt('%s/STDmassdens.txt'%(path))
# stdtotmass= np.genfromtxt('%s/STDmassdenstotal.txt'%(path))

stepx=(xhigh-xlow)/(avgmassdens.shape[0])
stepy=(yhigh-ylow)/(avgmassdens.shape[1])
# x,y = np.meshgrid(np.arange(xlow,xhigh,stepx),np.arange(ylow,yhigh,stepy))
x,y = np.meshgrid(np.linspace(xlow,xhigh,avgmassdens.shape[0]),np.linspace(ylow,yhigh,avgmassdens.shape[1]))

# CREATE PLOT
fig1,ax1=subplots(1,figsize=(17,17),sharex=False,sharey=False,facecolor='w', edgecolor='k')
#fig1.subplots_adjust(hspace=0)

avgmassdens = avgmassdens/sigcrit
im = ax1.imshow(np.log10(avgmassdens),extent=(xlow,xhigh,ylow,yhigh),aspect='auto',cmap='YlOrRd',origin='lower',vmin=-1.0,vmax=1.04)
# im = ax1.imshow(avgmassdens,extent=(xlow,xhigh,ylow,yhigh),aspect='auto',cmap='YlOrRd',origin='lower')

# minind = np.unravel_index(np.argmin(avgtotmassdens.T),avgtotmassdens.T.shape)
# totmassmin = avgtotmassdens.T[minind[0]][minind[1]]
# maxind = np.unravel_index(np.argmax(avgtotmassdens.T),avgtotmassdens.T.shape)
# totmassmax = avgtotmassdens.T[maxind[0]][maxind[1]]

#levels = [totmassmin,3e10,5e10,7e10,9e10,2e11,4e11,6e11,8e11,1e12,1e13]
levels = np.concatenate((np.linspace(4,9,10)*1e10,np.linspace(1,2,5)*1e11,np.linspace(3,9,4)*1e11,np.linspace(1,5,3)*1e12))
#contourmass=ax1.contour(x,y,avgtotmassdens.T,levels=levels,origin='lower',alpha=0.5)
#contourmass=ax1.contour(x,y,avgtotmassdens.T,levels=100,origin='lower',norm=colors.LogNorm(vmin=avgtotmassdens.T.min(), vmax=avgtotmassdens.T.max()))
contourmass=ax1.contour(x,y,np.log10(avgmassdens),levels=40,origin='lower',alpha=1,linewidths=0.75)
#contourmass=ax1.contour(x,y,avgmassdens,levels=30,origin='lower',alpha=1)


#contourmass=ax1.contour(x,y,avgmassdens.T,levels=10,origin='lower')
#contoursis=ax1.contour(x,y,totmassdens.T,levels=10,origin='lower')

divider = make_axes_locatable(ax1)
ax1.set_ylabel(r'y [arcsec]',fontsize=15,fontweight='bold')
ax1.set_xlabel(r'x [arcsec]',fontsize=15,fontweight='bold')
cax1=divider.append_axes("right", size="5%",pad=0.25)

cax1.yaxis.set_label_position("right")
cax1.yaxis.tick_right()

# fig1.colorbar(im,label=r'Surface Mass Density [M$_{\odot}$ arcsec$^{-2}$]',cax=cax1)
fig1.colorbar(im,label=r'log$_{10}$($\Sigma / \Sigma_{crit}$)',cax=cax1)
ax1.minorticks_on()

xrecon = np.load('%s/xrecon.npy'%(path),allow_pickle=True)
yrecon = np.load('%s/yrecon.npy'%(path),allow_pickle=True)
if allrev == 'rev':
	srcname = ['QSO','A2','B1','B2','C1','C2']
	sources = [QSO,A2,B1,B2,C1,C2]
	xim = np.loadtxt('J1004points_revised.txt',usecols=0)
	yim = np.loadtxt('J1004points_revised.txt',usecols=1)
if allrev == 'all':
	srcname = ['QSO','A1','A2','A3','B1','B2','C1','C2']
	sources = [QSO,A1,A2,A3,B1,B2,C1,C2]
	xim = np.loadtxt('SDSSJ1004points.txt',usecols=0)
	yim = np.loadtxt('SDSSJ1004points.txt',usecols=1)
zgroups = [1.734,3.33,2.74,3.28]
zcolors=['lime','aqua','plum','gold']
galgroups = ['QSO','Group A','Group B','Group C']
immarkers = ['^','o','s','d']
imgxgroups,imgygroups = [[] for x in range(len(zgroups))],[[] for x in range(len(zgroups))]
obsimx,obsimy = [[] for x in range(len(zgroups))],[[] for x in range(len(zgroups))]
for i in range(len(srcname)):
	for z in range(len(zgroups)):
		if sources[i].zs == zgroups[z]:
			imgxgroups[z].append(xrecon[i])
			imgygroups[z].append(yrecon[i])
			obsimx[z].append(sources[i].get_imagepositions()[0])
			obsimy[z].append(sources[i].get_imagepositions()[1])
for i in range(len(zgroups)):
	ax1.scatter(np.concatenate(imgxgroups[i]),np.concatenate(imgygroups[i]),color=zcolors[i],label=galgroups[i],zorder=2.5,marker=immarkers[i],alpha=0.6,s=70)

print('Extra Images: ',len(np.concatenate(xrecon))-len(xim))
# ax1.scatter(xim,yim,color='r',label='Galaxy Images',zorder=2.5)
xqso = np.array([0.000,-1.317,11.039,8.399,7.197])
yqso = np.array([0.000,3.532,-4.492,9.707,4.603])
ax1.scatter(xqso,yqso,color='lime',label='QSO Images',zorder=2.5)
# ax1.scatter([xsis],[ysis],color='y',marker='*',zorder=2.5) # BCG Center for Sersic Lens

# ax1.scatter(xqso[:4],yqso[:4],color='cyan',marker='o',s=90,facecolors='none', edgecolors='cyan',zorder=2.5)

ax1.set_aspect('equal')
ax1.set_anchor('C')
#plt.tight_layout()

ax1.grid(False)

ax1.tick_params(axis='x',labelsize='10')
ax1.tick_params(axis='y',labelsize='10')

ax1.scatter(xim,yim,color='k',alpha=1,marker='.',label='Observed',zorder=3,s=15)
ax1.legend(loc=3,title='',ncol=3,prop={'size':9})
# ax1.errorbar(xgal,ygal,color='g',marker='d',linestyle='None',label='Cluster Galaxy')
plt.show()
sys.exit()
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
	# ax3.errorbar(x1,x2,color='r',label='region')
	# # ax1.errorbar([regionx],[regiony],color='b',marker='*')
	# ax1.errorbar([xclump],[yclump],color='r',marker='*')

	dist = np.sqrt((x-xclump)**2 + (y-yclump)**2)
	densregion = np.array([(avgmassdens[i][j]-1.2e11)*(stepx**2) for i in tqdm(range(dist.shape[0])) for j in range(dist.shape[1]) if dist[i][j] <= regionsize])
	massregion = sum(densregion)

	return xclump,yclump,massregion
xclump,yclump,massclump = region(5,-1,2.0)
# xclump,yclump,massclump = region(xcent,ycent,100/arcsec_kpc)

peakind = np.unravel_index(np.argmax(avgmassdens),avgmassdens.shape)
xpeakind,ypeakind = peakind[0],peakind[1]
xpeak,ypeak = x[xpeakind][ypeakind],y[xpeakind][ypeakind] # Peak of entire Mass Distribution
n,m,masscenter = region(xpeak,ypeak,3.0)
# ax1.errorbar([xpeak],[ypeak],color='r',marker='*')
centoffset = np.sqrt((xpeak-xcent)**2 + (ypeak-ycent)**2)
centdens = avgmassdens[xpeakind][ypeakind]*(1.989e+33/(arcsec_pc**2))*(1/(3.086e+18**2)) # g per cm^2

Rclump = 2.0*arcsec_pc # in pc
mvir = 200*critdens*((4*np.pi)/3)*(Rclump**3)*200
vclump = np.sqrt((G*massclump)/Rclump) # DM particle velocity in clump in km/s
tcross = (2*Rclump*pcconv)/vclump # in s
tcross = tcross/syear # in year

dmcross = 1 # cm^2 / g
densclump = massclump/((4/3)*np.pi*(Rclump**3)) # clump density in solar mass per pc^3
densclump = densclump*(1.989e+33/(3.086e+18**3)) # clump density in g cm^-3
vclump = vclump*100000 # DM particle velocity in cm/s
tdisp = (1/(densclump*dmcross*vclump))/syear # in years

tclust = (7.594-6.020)*(10**9) # Age of cluster if formed at z=1 in years
sigom = 1/(densclump*vclump*tclust*syear) # DM cross in cm^2 / g for cluster forming at certain z
# ax1.invert_xaxis()

# CREATE PLOT
fig4,ax4=subplots(1,figsize=(17,17),sharex=False,sharey=False,facecolor='w', edgecolor='k')
#fig1.subplots_adjust(hspace=0)

# avgmassdens = avgmassdens/sigcrit
imreg = ax4.imshow(avgmassdens,extent=(xlow,xhigh,ylow,yhigh),aspect='auto',cmap='YlOrRd',origin='lower')
contourmass=ax4.contour(x,y,np.log10(avgmassdens),levels=40,origin='lower',alpha=1)
ax4.errorbar([xclump],[yclump],color='b',marker='*',linestyle='None',label='SE Clump Peak')
divider = make_axes_locatable(ax4)
ax4.set_ylabel(r'y [arcsec]',fontsize=15,fontweight='bold')
ax4.set_xlabel(r'x [arcsec]',fontsize=15,fontweight='bold')
cax4=divider.append_axes("right", size="5%",pad=0.25)
cax4.yaxis.set_label_position("right")
cax4.yaxis.tick_right()
fig4.colorbar(imreg,label=r'Surface Mass Density [M$_{\odot}$ arcsec$^{-2}$]',cax=cax4)
ax4.minorticks_on()
ax4.set_aspect('equal')
ax4.set_anchor('C')
ax4.grid(False)
ax4.tick_params(axis='x',labelsize='10')
ax4.tick_params(axis='y',labelsize='10')
ax4.set_xlim(xclump-4,xclump+4)
ax4.set_ylim(yclump-4,yclump+4)
ax4.errorbar(xgal,ygal,color='g',marker='d',linestyle='None',label='Cluster Galaxy')
ax4.legend(loc=3,title='',ncol=2,prop={'size':10})

distclump = min(np.sqrt((xgal-xclump)**2 + (ygal-yclump)**2)) # Distance from SE Clump peak to nearest cluster galaxy in arcsec
distclump = distclump*arcsec_kpc # in kpc

plt.show()
# sys.exit()

# CREATE PLOT
fig2,ax2=subplots(1,figsize=(17,17),sharex=False,sharey=False,facecolor='w', edgecolor='k')
#fig1.subplots_adjust(hspace=0)

RSD = stdmass/avgmassdens # Coefficient of Variation or Relative Standard Deviation
# RMSmassdens = np.genfromtxt('%s/RMSmassdensplummers.txt'%(path))
# RMStotmassdens = np.genfromtxt('%s/RMSmassdenstotal.txt'%(path))
imstd = ax2.imshow(RSD,extent=(xlow,xhigh,ylow,yhigh),aspect='auto',cmap='PuBu',origin='lower')
contourstd=ax2.contour(x,y,RSD,levels=10,origin='lower',alpha=0.5)

divider = make_axes_locatable(ax2)
ax2.set_ylabel(r'y [arcsec]',fontsize=15,fontweight='bold')
ax2.set_xlabel(r'x [arcsec]',fontsize=15,fontweight='bold')
cax2=divider.append_axes("right", size="5%",pad=0.25)

cax2.yaxis.set_label_position("right")
cax2.yaxis.tick_right()

#fig2.colorbar(imstd,label=r'$\sigma$ [M$_{\odot}$ arcsec$^{-2}$]',cax=cax2)
fig2.colorbar(imstd,label=r'RSD',cax=cax2)
ax2.minorticks_on()

ax2.scatter(xim,yim,color='r',label='Galaxy Images')
ax2.scatter(xqso,yqso,color='lime',label='QSO Images')

ax2.set_aspect('equal')
ax2.set_anchor('C')
#plt.tight_layout()

ax2.grid(False)

ax2.tick_params(axis='x',labelsize='10')
ax2.tick_params(axis='y',labelsize='10')

ax2.legend(loc=3,title='',ncol=2,prop={'size':10})

plt.show()

# CREATE PLOT
fig3,ax3=subplots(1,figsize=(17,17),sharex=False,sharey=False,facecolor='w', edgecolor='k')

rgb = plt.imread("background50x50.jpg")
ax3.imshow(rgb, extent=[-25,25,-25,25])
# plt.gca().invert_xaxis()
imcolors = ['lime','aqua','plum','gold']
qsonames = ['A','B','C','D','E']
qsotextdist = [(0,-30),(-10,10),(10,-25),(10,10),(-10,10)]
Anames = ['AX.1','AX.2','AX.3','AX.4','AX.5']
Atextdist = [(0,-20),(-5,10),(50,-40),(10,10),(0,-20)]
Bnames = ['BX.1','BX.2','BX.3']
Btextdist = [(0,-20),(-5,10),(20,-20)]
Cnames = ['CX.1','CX.2','CX.3']
Ctextdist = [(30,0),(10,-20),(30,10)]

# Image Centered at the BCG location
for i in range(len(zgroups)):
	imgx,imgy = np.concatenate(obsimx[i]) - xcent , np.concatenate(obsimy[i]) - ycent
	ax3.scatter(imgx,imgy,color=imcolors[i],label=galgroups[i],zorder=2.5,alpha=0.5,marker=immarkers[i],s=15)
	if sources[i].name == 'QSO':
		for j in range(len(np.array(obsimx[i]).T)):
			ax3.annotate(qsonames[j], # this is the text
						(np.array(obsimx[i]).T[j]-xcent,np.array(obsimy[i]).T[j]-ycent), # this is the point to label
						textcoords="offset pixels", # how to position the text
						xytext=(qsotextdist[j][0],qsotextdist[j][1]), # distance from text to points (x,y)
						ha='right',color='g') # horizontal alignment can be left, right or center
	if i==1: # Group A
		for j in range(len(np.array(obsimx[i][0]).T)):
			ax3.annotate(Anames[j], # this is the text
						(np.array(obsimx[i][0]).T[j]-xcent,np.array(obsimy[i][0]).T[j]-ycent), # this is the point to label
						textcoords="offset pixels", # how to position the text
						xytext=(Atextdist[j][0],Atextdist[j][1]), # distance from text to points (x,y)
						ha='right',color='aqua') # horizontal alignment can be left, right or center		
	if i==2: # Group B
		for j in range(len(np.array(obsimx[i][0]).T)):
			ax3.annotate(Bnames[j], # this is the text
						(np.array(obsimx[i][0]).T[j]-xcent,np.array(obsimy[i][0]).T[j]-ycent), # this is the point to label
						textcoords="offset pixels", # how to position the text
						xytext=(Btextdist[j][0],Btextdist[j][1]), # distance from text to points (x,y)
						ha='right',color='plum') # horizontal alignment can be left, right or center
	if i==3: # Group C
		for j in range(len(np.array(obsimx[i][0]).T)):
			ax3.annotate(Cnames[j], # this is the text
						(np.array(obsimx[i][0]).T[j]-xcent,np.array(obsimy[i][0]).T[j]-ycent), # this is the point to label
						textcoords="offset pixels", # how to position the text
						xytext=(Ctextdist[j][0],Ctextdist[j][1]), # distance from text to points (x,y)
						ha='right',color='gold') # horizontal alignment can be left, right or center		
ax3.plot([13,18],[17,17],lw=5,color='k')
ax3.annotate('5"', # this is the text
			(15,17.5), # this is the point to label
			color='k') # horizontal alignment can be left, right or center	

plt.xticks(color='w')
plt.yticks(color='w')
ax3.set_aspect('equal')
ax3.set_anchor('C')
ax3.minorticks_on()
ax3.grid(False)
ax3.tick_params(axis='x',labelsize='10')
ax3.tick_params(axis='y',labelsize='10')
# ax3.set_ylabel(r'y [arcsec]',fontsize=15,fontweight='bold')
# ax3.set_xlabel(r'x [arcsec]',fontsize=15,fontweight='bold')
ax3.scatter(xgal-xcent,ygal-ycent,color='r',marker='d',alpha=0.5,s=15)
ax3.legend(loc=4,title='',ncol=2,prop={'size':10})
ax3.set_xlim(-18,19)
ax3.set_ylim(-18,19)
ax3.errorbar([xclump-xcent],[yclump-ycent],color='b',marker='*',linestyle='None',label='SE Clump Peak')

plt.show()