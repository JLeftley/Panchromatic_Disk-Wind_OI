#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
import os
#sys.path.append(os.getcwd())

"""
Created on Thu Jan  5 13:57:30 2017

@author: jleftley
"""
#print("begin")
from multiprocessing import Pool # Python multiprocessing
# from schwimmbad import MPIPool # MPI pool


import numpy as np

import matplotlib.pylab as plt

import emcee

import datetime
#print("begin")
from UVplotter2py3 import UVpoint
#print("begin")
from astropy.time import Time
#print("begin")
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
#print("begin")
import galario
if galario.HAVE_CUDA:
    from galario import double_cuda as double
    #print('GPU used')
else:
    from galario import double as double
    #print('CPU used')

from astropy.io import fits
from copy import deepcopy as dc
from glob import glob

from scipy import special
from scipy.stats import truncnorm, circmean, circstd
import corner
import warnings
warnings.filterwarnings('ignore') # You can remove but many warnings from numpy (inconsiquential warnings)
from os.path import isfile
from os import mkdir

os.environ["OMP_NUM_THREADS"] = "1"

from astropy.convolution import Gaussian2DKernel, convolve
import scipy

import oifits





def loadextAmMgFeolrv1p5():
    fn='/srv/jleftley/AGN/Q_Am_MgFeolivine_Dor_DHS_f1.0_rv0.1.dat'
    it=np.genfromtxt(fn,skip_header=1) 
    wlext, Q = it[:,0],it[:,1]
    a=0.1*1e-4#um
    rho=4. #g/cm^3
    ext= (3/4)* np.array(Q)/(rho*a)
    return(wlext,ext)
def loadextISM():
    ext=np.array([
     39793.801,
     67494.584,
     78152.768,
     116309.28,
     154490.88,
     210697.36,
     200780.75,
     267667.88,
     323695.77,
     356932.44,
     404597.06,
     448591.37,
     507437.96,
     565666.00,
     637002.05,
     931517.27,
     1137652.6,
     919333.49,
     631419.27,
     567468.25,
     503253.84,
     368146.13,
     278863.55,
     199173.93,
     135122.01,
     85026.363,
     66102.880,
     50531.502,
     38448.414,
     28846.463,
     21354.264,
     15798.436,
     11838.844,
     9010.1833,
     6986.4643,
     5493.4506,
     4383.0545,
     3583.7993,
     3206.5404,
     2924.1727,
     2806.9763,
     2639.7689,
     3053.1844,
     4800.5646,
     7260.0788,
     9651.0855,
     12116.474,
     15534.693,
     17369.322,
     16458.639,
     15175.293,
     13640.735,
     11497.354,
     9885.0501,
     8538.0231,
     7301.4810,
     5923.5105,
     4831.8437,
     4063.4057,
     3988.1476,
     4400.6627,
     4952.5537,
     5557.2525,
     6054.0399,
     6276.3328,
     6124.0335,
     5641.0919,
     5136.5000,
     4713.1708,
     4310.8821,
     3989.8932,
     3696.9629,
     3180.8610,
     2760.5042,
     2395.2222,
     2081.1324,
     1789.5720,
     1516.7320,
     1273.3178,
     1061.1560,
     879.22762,
     725.49803,
     491.86208,
     333.69313,
     228.05272,
     189.83898,
     99.169315,
     55.711456,
     31.821471,
     18.266502,
     10.507832,
     6.0476670,
     3.4824221,
     1.9172626,
     1.2454002,
     0.65250673])/10
    #Need to be aligned in the left side
    wlext=[
       0.0010000000,
       0.0013180000,
       0.0017380000,
       0.0022910000,
       0.0030199999,
       0.0039809998,
       0.0052479999,
       0.0069180001,
       0.0091199996,
       0.012020000,
       0.015850000,
       0.020889999,
       0.027540000,
       0.036309998,
       0.047860000,
       0.063100003,
       0.083180003,
       0.10960000,
       0.14450000,
       0.19050001,
       0.25119999,
       0.33109999,
       0.43650001,
       0.57539999,
       0.75860000,
       1.0000000,
       1.1480000,
       1.3180000,
       1.5140001,
       1.7380000,
       1.9950000,
       2.2909999,
       2.6300001,
       3.0200000,
       3.4670000,
       3.9809999,
       4.5710001,
       5.2480001,
       5.7540002,
       6.3099999,
       6.6069999,
       7.2440000,
       7.5860000,
       7.9429998,
       8.2220001,
       8.5109997,
       8.8100004,
       9.1199999,
       9.4410000,
       9.7720003,
       10.120000,
       10.470000,
       10.840000,
       11.220000,
       11.610000,
       12.020000,
       12.590000,
       13.180000,
       13.800000,
       14.450000,
       15.140000,
       15.850000,
       16.600000,
       17.379999,
       18.200001,
       19.049999,
       19.950001,
       20.889999,
       21.879999,
       22.910000,
       23.990000,
       25.120001,
       27.540001,
       30.200001,
       33.110001,
       36.310001,
       39.810001,
       43.650002,
       47.860001,
       52.480000,
       57.540001,
       63.099998,
       75.860001,
       91.199997,
       109.60000,
       120.20000,
       166.00000,
       218.80000,
       288.39999,
       380.20001,
       501.20001,
       660.70001,
       871.00000,
       1171.0000,
       1451.0000,
       2000.0000]
    return(wlext,ext)

def loadextC():
    table=np.genfromtxt('/srv/jleftley/AGN/carbon1to30um.csv',skip_header=1,delimiter=',')
    wlext=table[:,0]
    ext=table[:,1]
    return(wlext,ext)

def mix(wlens,p):#returns in wlextsm (p betweeon 0 and 1 is frac of carbon)
    wlextC, extC = loadextC()
    wlext, extsm = loadextAmMgFeolrv1p5()
    #wlext, extsm = loadextISM()
    intextC=np.interp( wlens, wlextC, extC)#wlext defined outside
    intextsm=np.interp( wlens, wlext, extsm)
    return(p*intextC+(1-p)*intextsm)

def rndm(a, b, g,gen, size=1):
    """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
    r = gen.random(size=size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag)*r)**(1./g)

def Gauss(amp,mx,my,stdx,stdy,theta,varx,vary):
    Ga=(np.cos(theta)**2)/(2*stdx**2)+(np.sin(theta)**2)/(2*stdy**2)
    Gb=(-np.sin(2*theta))/(2*stdx**2)+(np.sin(2*theta))/(2*stdy**2)
    Gc=(np.sin(theta)**2)/(2*stdx**2)+(np.cos(theta)**2)/(2*stdy**2)
    gaussian=amp*np.exp(-Ga*(varx-mx)**2-Gb*(varx-mx)*(vary-my)-Gc*(vary-my)**2)
    return gaussian

def bb(A,wavs,T):
    return (A*3.97289171e-25/(wavs**3))/(np.exp(0.0143877688/(wavs*T))-1)

def pixcirc(rad,oversampling=4):
    pixsmall=1/oversampling
    rrup=int(np.ceil(rad))+1
    yupr,xupr=np.mgrid[-rrup:rrup:pixsmall,-rrup:rrup:pixsmall]-0.5#+pixsmall/2
    imcirc=np.zeros(np.array(xupr.shape))
    imcirc[xupr**2+yupr**2<=(rad)**2]=1
    imgen=imcirc.reshape((rrup*2,oversampling,rrup*2,oversampling)).sum((1,-1))[1:,1:]
    yrs,xrs=np.mgrid[-rrup+1:rrup,-rrup+1:rrup]
    return yrs,xrs,imgen/imgen.sum()

def wavpixcirc(rad,oversampling=4):
    pixsmall=1/oversampling
    rrup=int(np.ceil(max(rad)))+1
    yupr,xupr=np.mgrid[-rrup:rrup:pixsmall,-rrup:rrup:pixsmall]-0.5#+pixsmall/2
    imcirc=np.zeros(np.array((len(rad),*xupr.shape)))
    for num,insi in enumerate(rad):
        imcirc[num,xupr**2+yupr**2<=(insi)**2]=1
    imgen=imcirc.reshape((len(rad),rrup*2,oversampling,rrup*2,oversampling)).sum((2,-1))[:,1:,1:]
    yrs,xrs=np.mgrid[-rrup+1:rrup,-rrup+1:rrup]
    return yrs,xrs,(imgen.T/imgen.sum((1,2))).T


def hyp_cone_cube_fov(alph,alphoff,beta,obslid,rsub,rbreak,Tstar,rstar,smob,inc,ang,grad,nxy,dxy,wlen,coolobsc,numpo,Tin,clumplf,wbase,csize,vloss,vloss2,seed):
    # Removed dbreak, a, zoom, nps, npsoff, frac, windlaw, dbase
    # Added Tstar, Tin

    # Create a blank image and coords, changing image size will not effect the scaling, changing the step will.
    #grad=0
    width=0.05
    pixel_scale_conv=dxy*180*3600*1000/np.pi
    
    
    rng = np.random.default_rng(seed)
    #np.random.seed(seed)
    #npsi=int(nps)
    npsi=int(numpo)
    npsiw=0#int(numpo*10**(frac))
    sublimation=rsub/1
    #ROUT=500*sublimation
    ROUT=4*nxy*pixel_scale_conv[-1]
    betaz=beta/1
    #b=dc(zoom)
    #xymax=np.sqrt((ROUT/a + np.sqrt(1+sublimation**2 / b**2 ) )**2 - 1)*b
    #if xymax<=ROUT:
    #    xymax=dc(ROUT)
    #xhy=np.random.uniform(sublimation,xymax,npsiw)#*zoom
    '''
    xhy=rndm(9999,10000,1,rng,size=npsiw)
    aerr=rng.normal(a,width,size=npsiw)
    '''
    
    if inc==obslid:
        obslid+=1e-12
    inc_rad=inc*np.pi/180
    obslid2=obslid*np.pi/180
    # Create a blank image and coords, changing image size will not effect the scaling, changing the step will.
    thettan=np.tan(obslid2)
    asd, bsd = (rsub) / beta, (ROUT ) / beta
    xdk=truncnorm.rvs(asd,bsd,loc=0,scale=betaz,size=npsi-npsiw,random_state=rng)
    #ydk=np.random.choice([-1,1],size=npsi,replace=True)*(thettan+1e-8)*(xdk-sublimation)
    ydk=rng.normal(0,(xdk-sublimation)*thettan/3)
    tinc=np.tan(inc_rad)
    #yhy=rng.choice([-1,1],size=npsiw,replace=True)*aerr*(np.sqrt(1+xhy**2/b**2)-np.sqrt(1+sublimation**2/b**2))
    
    #yhy=np.append(yhy,ydk)
    #xhy=np.append(xhy,xdk)
    yhy=ydk.copy()
    xhy=xdk.copy()
    #yhy*=zoom
    #xhy*=zoom
    
    theta=rng.uniform(0,2*np.pi,npsi)
    comp_IM=np.zeros((len(dxy),nxy,nxy))
    xrot=np.cos(theta)*xhy
    zrot=np.sin(theta)*xhy
    
    
    yrot=yhy*np.cos(-inc_rad)+zrot*np.sin(-inc_rad)
    zrot2=-yhy*np.sin(-inc_rad)+zrot*np.cos(-inc_rad)
    
    xrot2=xrot*np.cos(-ang*np.pi/180)+yrot*np.sin(-ang*np.pi/180)
    yrot2=-xrot*np.sin(-ang*np.pi/180)+yrot*np.cos(-ang*np.pi/180)
    aso=np.argsort(zrot2)
    zrot2=zrot2[aso]
    xrot2=xrot2[aso]
    yrot2=yrot2[aso]
    yhy=yhy[aso]
    xhy=xhy[aso]
    yrot=yrot[aso]
    xrot=xrot[aso]
    zrot=zrot[aso]
    xrotc=xrot2.copy()
    zrotc=zrot2.copy()
    yrotc=yrot2.copy()
    # wind=np.arange(0,npsi,dtype=int)[aso]
    #wind=np.arange(0,npsi,dtype=int)[aso>=npsiw]
    #wind=np.arange(0,npsi,dtype=int)[aso]>=npsiw
    
    # xrot2/=pixel_scale_conv
    xrot2=np.array([xrot2/i for i in pixel_scale_conv])
    yrot2=np.array([yrot2/i for i in pixel_scale_conv])
    zrot2=np.array([zrot2/i for i in pixel_scale_conv])
    #zrot2=[zrot2/i for i in pixel_scale_conv]
    wlord=np.argsort(wlen)
    
    coolobsc2=mix(wlen[wlord]*1e6,grad)*(10**coolobsc)
    coolobsc2=coolobsc2[np.argsort(wlord)]

    coolobscd=mix(wlen[wlord]*1e6,grad)*(10**smob)
    coolobscd=coolobscd[np.argsort(wlord)]
    
    # zrot2/=pixel_scale_conv
    
    #xp=np.array(xrot2,dtype=int)+nxy//2
    #yp=np.array(yrot2,dtype=int)+nxy//2
    # zp=zrot2.astype(int)+128
    #mask=(xp<nxy)*(yp<nxy)*(yp>0)*(xp>0)#*(np.abs(zp)<nxy)  
    
    ##obscartion from disk with scale hight
    

    
    z1= (-np.sqrt(xrot**2 *tinc**2 * thettan**2 - xrot**2 * thettan**4 + yhy**2 * thettan**2 - 2 * yhy * zrot * tinc * thettan**2 + 2 * yhy * sublimation * thettan**3 + zrot**2 * tinc**2 * thettan**2 - 2 * zrot * tinc * sublimation * thettan**3 + sublimation**2 * thettan**4) - yhy * tinc + zrot * tinc**2 - tinc * sublimation * thettan)/(tinc**2 - thettan**2)
    z2= (np.sqrt(xrot**2 *tinc**2 * thettan**2 - xrot**2 * thettan**4 + yhy**2 * thettan**2 - 2 * yhy * zrot * tinc * thettan**2 + 2 * yhy * sublimation * thettan**3 + zrot**2 * tinc**2 * thettan**2 - 2 * zrot * tinc * sublimation * thettan**3 + sublimation**2 * thettan**4) - yhy * tinc + zrot * tinc**2 - tinc * sublimation * thettan)/(tinc**2 - thettan**2)
    
    
    thettan=np.tan(-obslid2)
    z3= (-np.sqrt(xrot**2 *tinc**2 * thettan**2 - xrot**2 * thettan**4 + yhy**2 * thettan**2 - 2 * yhy * zrot * tinc * thettan**2 + 2 * yhy * sublimation * thettan**3 + zrot**2 * tinc**2 * thettan**2 - 2 * zrot * tinc * sublimation * thettan**3 + sublimation**2 * thettan**4) - yhy * tinc + zrot * tinc**2 - tinc * sublimation * thettan)/(tinc**2 - thettan**2)
    z4= (np.sqrt(xrot**2 *tinc**2 * thettan**2 - xrot**2 * thettan**4 + yhy**2 * thettan**2 - 2 * yhy * zrot * tinc * thettan**2 + 2 * yhy * sublimation * thettan**3 + zrot**2 * tinc**2 * thettan**2 - 2 * zrot * tinc * sublimation * thettan**3 + sublimation**2 * thettan**4) - yhy * tinc + zrot * tinc**2 - tinc * sublimation * thettan)/(tinc**2 - thettan**2)
    
    
    
    
    nanmask=np.isnan(z1)*np.isnan(z2)*np.isnan(z3)*np.isnan(z4)
    if obslid2<inc_rad:
        
        if True in (z1>z2):
            Z1=z1
            Z2=z2
            Z3=z4
            Z4=z3
        else:# True in (z2>z1):
            Z1=z2
            Z2=z1
            Z3=z3
            Z4=z4
        
        
        ### REJECTING ABOVE/BELOW zero removes false intersection
        y1=(Z1-zrot)*tinc + yhy
        Z1[y1<0]=np.nan
        y1[y1<0]=np.nan
        y2=(Z2-zrot)*tinc + yhy
        Z2[y2<0]=np.nan
        y2[y2<0]=np.nan
        y3=(Z3-zrot)*tinc + yhy
        Z3[y3>0]=np.nan
        y3[y3>0]=np.nan
        y4=(Z4-zrot)*tinc + yhy
        Z4[y4>0]=np.nan
        y4[y4>0]=np.nan
        
        
        Z1mask=y1<yhy
        Z2mask=y2<yhy
        Z4[Z2mask]=np.nan
        Z3[Z1mask]=np.nan
        Z1[Z1mask]=np.nan
        Z2[Z2mask]=np.nan
        
        y4[Z4<zrot]=yhy[Z4<zrot]
        Z4[Z4<zrot]=zrot[Z4<zrot]
        y3[Z3<zrot]=yhy[Z3<zrot]
        Z3[Z3<zrot]=zrot[Z3<zrot]
        
        zmax=np.sqrt(ROUT**2 - xrot**2)
        ymax=(ROUT-sublimation)*np.tan(obslid2)
        outmask=Z1>zmax
        Z1[outmask]=zmax[outmask]
        y1[outmask]=ymax
        
        y4[np.isnan(Z4)]=0
        Z4[np.isnan(Z4)]=0
        y3[np.isnan(Z3)]=0
        Z3[np.isnan(Z3)]=0
        y2[np.isnan(Z2)]=0
        Z2[np.isnan(Z2)]=0
        y1[np.isnan(Z1)]=0
        Z1[np.isnan(Z1)]=0
        
        
        
        Z4/=(1.4142135623730951*betaz)
        Z3/=(1.4142135623730951*betaz)
        Z2/=(1.4142135623730951*betaz)
        Z1/=(1.4142135623730951*betaz)
        
        
        # z_ent/=(1.4142135623730951*beta)
        # z_exit/=(1.4142135623730951*beta)
        x_ent=xrot/(1.4142135623730951*betaz)
        #xerf=np.array([math.erf(i) for i in x_ent])
        # zpath=np.array([math.erf(i)+math.erf(j)-math.erf(k)-math.erf(l) for i,j,k,l in zip(Z1,Z2,Z3,Z4)])
        zpath=special.erf(Z1)+special.erf(Z2)-special.erf(Z3)-special.erf(Z4)
        #cool_cont=coolobsc2*np.pi/4
        zpath[zpath<0]=0
        zpath[np.abs(xrot)>ROUT]=0
        tausmo=-np.abs(np.exp(-x_ent**2)*zpath*(np.sqrt(1+tinc**2)))#*(y1+y2-y3-y4))
    else:
        if True in (z1>z2):
            Z1=z1
            Z2=z2
            Z3=z3
            Z4=z4
        else:# True in (z2>z1):
            Z1=z2
            Z2=z1
            Z3=z4
            Z4=z3
        
        ### REJECTING ABOVE/BELOW zero removes false intersection
        y1=(Z1-zrot)*tinc + yhy
        Z1[y1<0]=np.nan
        y1[y1<0]=np.nan
        y2=(Z2-zrot)*tinc + yhy
        Z2[y2<0]=np.nan
        y2[y2<0]=np.nan
        y3=(Z3-zrot)*tinc + yhy
        Z3[y3>0]=np.nan
        y3[y3>0]=np.nan
        y4=(Z4-zrot)*tinc + yhy
        Z4[y4>0]=np.nan
        y4[y4>0]=np.nan
        Z1mask=Z1<zrot
        Z3mask=Z3<zrot
        
        Z2[Z1mask]=np.nan
        Z4[Z1mask]=np.nan
        Z4[Z3mask]=np.nan
        Z1[Z1mask]=zrot[Z1mask]
        y1[Z1mask]=yhy[Z1mask]
        Z3[Z3mask]=zrot[Z3mask]
        y3[Z3mask]=yhy[Z3mask]
        
        Z2mask=Z2<zrot
        Z2[Z2mask]=np.nan
        Z4mask=Z4<zrot
        Z4[Z4mask]=np.nan
        
        Z5mask=np.isfinite(Z2)+np.isfinite(Z4)
        Z5=dc(Z4)*np.nan
        y5=dc(y4)*np.nan
        Z5[Z5mask]=zrot[Z5mask]
        y5[Z5mask]=yhy[Z5mask]
        
        
        y5[np.isnan(Z5)]=0
        Z5[np.isnan(Z5)]=0        
        y4[np.isnan(Z4)]=0
        Z4[np.isnan(Z4)]=0
        y3[np.isnan(Z3)]=0
        Z3[np.isnan(Z3)]=0
        y2[np.isnan(Z2)]=0
        Z2[np.isnan(Z2)]=0
        y1[np.isnan(Z1)]=0
        Z1[np.isnan(Z1)]=0
        
        Z1[nanmask]=zrot[nanmask]
        y1[nanmask]=yhy[nanmask]
        
        zmax=np.sqrt(ROUT**2 - xrot**2)/(1.4142135623730951*betaz)
        ymax=(ROUT-sublimation)*np.tan(obslid2)
        
        zmax[np.isnan(zmax)]=0
        
        Z5/=(1.4142135623730951*betaz)
        Z4/=(1.4142135623730951*betaz)
        Z3/=(1.4142135623730951*betaz)
        Z2/=(1.4142135623730951*betaz)
        Z1/=(1.4142135623730951*betaz)
        
        x_ent=xrot/(1.4142135623730951*betaz)
        #xerf=np.array([math.erf(i) for i in x_ent])
        # zpath=np.array([1.0-math.erf(i)+math.erf(j)+math.erf(l)-math.erf(m)-math.erf(k) for i,j,k,l,m in zip(Z1,Z2,Z3,Z4,Z5)])
        zpath=(special.erf(zmax)-special.erf(Z1)+special.erf(Z2)+special.erf(Z4)-special.erf(Z5)-special.erf(Z3))
        #cool_cont=coolobsc2*np.pi/4
        zpath[zpath<0]=0
        zpath[np.abs(xrot)>ROUT]=0
        tausmo=-np.abs(np.exp(-x_ent**2)*zpath*(np.sqrt(1+tinc**2)))#*(y1+y2-y3-y4))
    cool_cont=coolobsc2
    cool_contd=coolobscd*np.pi/4
    tau=tausmo
    cool_cont[cool_cont<0.0]=0.0
    cool_contd[cool_contd<0.0]=0.0
    poi=np.arctan2(yrotc,-zrotc)
    poi=np.abs(((poi+np.pi/2)%np.pi)-np.pi/2)
    pot=np.arctan2(-zrotc,xrotc)
    posfrac=0.5*np.cos(poi)*np.sin(pot)+0.5
    
    #Set clump temperatures
    rrr=np.sqrt(xhy**2+yhy**2)
    T2=(Tin*(rsub)**alph)/(rrr**alph)
    
    T2[rrr>rbreak*rsub]=(((Tin*(rsub)**alph)/((rsub*rbreak)**alph)))*((rsub*rbreak)**(alph+alphoff))/(rrr[rrr>rbreak*rsub]**(alph+alphoff))
    ## Wind disable
    #T2[wind]=Tin*(rsub)**nps/(np.sqrt(xhy[wind]**2+yhy[wind]**2)**nps)
    #rbw=np.arange(len(T2),dtype=int)[wind][rrr[wind]>dbreak*rsub]
    #T2[wind[rrr[wind]>dbreak*rsub]]=(((1800*(rsub)**nps)/((rsub*dbreak)**nps)))*((rsub*dbreak)**(nps+npsoff))/(rrr[wind][rrr[wind]>dbreak*rsub]**(nps+npsoff))
    #T2[rbw]=(((Tin*(rsub)**nps)/((rsub*dbreak)**nps)))*((rsub*dbreak)**(nps+npsoff))/(rrr[wind][rrr[wind]>dbreak*rsub]**(nps+npsoff))
    T3=np.zeros(T2.shape)+wbase
    #T3[wind]=dbase
    h = 6.63*10**(-34) #J*s=W*s=1*10^7erg=kg*m^2*s^-2
    c = 2.998*10**8 #m/s  to cm
    k = 1.38*10**(-23)  #J/K = kg*m^2*s / K


    posfracfrac=clumplf*(1-np.exp(-1.5*cool_cont*csize))
    posfracfrac[posfracfrac>1]=1
    posfracfrac[posfracfrac<0]=0
    #tcool=((2*h*c) / (wlen**3))/(np.exp(((h*c)/(k*100*wlen)))-1)
    fluxob=np.array([(1-posfracfrac[num]+posfracfrac[num]*posfrac)*np.exp(tau*cool_contd[num])*((2*h*c) / (i**3))/(np.exp(((h*c)/(k*T2*i)))-1) +((2*h*c) / (i**3))/(np.exp(((h*c)/(k*T3*i)))-1) for num,i in enumerate(wlen)])
    # Apply a FoV effect (make automatic, currently have to set UT (8) or AT (1.8) dish size manually)
    Dish=1.8 #AT
    #Dish=8 #UT
    fluxob=fluxob*np.exp(-(((xrotc**2+yrotc**2)*((np.pi/(1000*3600*180))**2))[np.newaxis]/(2*((wlen/(Dish*2.335))**2)[:,np.newaxis])))

    
    xmean=0#nxy//2
    ymean=0#nxy//2
    xp=np.array(xrot2+nxy/2,dtype=int)-int(xmean)
    yp=np.array(yrot2+nxy/2,dtype=int)-int(ymean)
    zp=np.array(zrot2,dtype=int)
    mask=(xp<nxy)*(yp<nxy)*(yp>0)*(xp>0)#*(np.abs(zp)<nxy) 

    wlenmasked=(np.zeros(yp.shape,dtype=int).T+np.arange(0,yp.shape[0])).T
    
    #Remove very faint clumps for computation time saveing (currently set so very little is rejected)
    zeromask=(fluxob.T>1e-2000*np.max(fluxob,axis=1)).T
    totmask=zeromask*mask
    
    #Generate a template clump per wavelength
    ycs,xcs,circles=wavpixcirc(csize/pixel_scale_conv,6)

    
    rc2=xcs**2+ycs**2
    
    tausm=np.exp(-2*np.sqrt(((csize)**2) - (pixel_scale_conv**2)[:,np.newaxis,np.newaxis]*rc2[np.newaxis])*cool_cont[:,np.newaxis,np.newaxis])
    tausm[np.isnan(tausm)]=1
    xp=xp[totmask,np.newaxis,np.newaxis]
    yp=yp[totmask,np.newaxis,np.newaxis]
    zp=zp[totmask,np.newaxis,np.newaxis]
    wlenmasked=wlenmasked[totmask]
    #outflux=
    norm=np.array([np.nansum(fluxob[i,np.invert(totmask[i])]) for i in range(len(wlen))])
    fluxob=fluxob[totmask]
    
    

    wlenm=wlenmasked[:,np.newaxis,np.newaxis]*np.ones(xcs.shape,dtype='bool')[np.newaxis]
    circles*=(1-tausm)
    fluxm=circles[wlenmasked]*fluxob[:,np.newaxis,np.newaxis]
    
    obsc=tausm[wlenmasked]
    #fluxm=(1-obsc)*fluxm
    xpm=xp+xcs[None]
    # print(tausm.shape,tau.shape,xcs.shape)
    ypm=yp+ycs[None]
    zpm=zp+np.zeros(ycs.shape)[None]
    
    # Calculate star flux as BB scaled in amplitude
    bbscale=(((2*h*c) / (wlen**3))/(np.exp(((h*c)/(k*Tstar*wlen)))-1))
    bbscale=(10**(vloss))*bbscale/bbscale[0]
    
    #Generate a template clump per wavelength
    ycsst,xcsst,circlesst=wavpixcirc(rstar/pixel_scale_conv,6)

    
    fstar=circlesst*bbscale[:,np.newaxis,np.newaxis]
    
    # for iii in range(len(wlen)):
    # comp_IM[:,ycsst+nxy//2,xcsst+nxy//2]+=fstar
    
    
    
    rpm=np.sqrt((xpm-nxy//2)**2+(ypm-nxy//2)**2+(zp-nxy//2)**2)*pixel_scale_conv[wlenm]
    mask=(xpm<nxy)*(ypm<nxy)*(ypm>0)*(xpm>0)*(fluxm>0)*(rpm>rsub)
    
    wlenm=wlenm[mask].tolist()
    fluxm=fluxm[mask].tolist()
    xpm=xpm[mask].tolist()
    ypm=ypm[mask].tolist()
    obsc=obsc[mask].tolist()
    zpm=zpm[mask].tolist()
    
    if xpm==[]:
        return comp_IM
    
    idx=np.argmin(np.abs(zpm))
    wlenstar=np.arange(len(wlen),dtype=int)[:,np.newaxis,np.newaxis]*np.ones(fstar.shape,dtype=int)
    wlenm=wlenm[:idx]+wlenstar.flatten().tolist() + wlenm[idx:]
    fluxm=fluxm[:idx]+fstar.flatten().tolist() + fluxm[idx:]
    zeropos=np.zeros(wlen.shape,dtype=int)[:,np.newaxis,np.newaxis]+nxy//2
    xstar=xcsst[None]+zeropos
    ystar=ycsst[None]+zeropos
    ostar=1-circlesst/circlesst.max(axis=(1,2))[:,np.newaxis,np.newaxis]
    
    xpm=xpm[:idx]+xstar.flatten().tolist() + xpm[idx:]
    ypm=ypm[:idx]+ystar.flatten().tolist() + ypm[idx:]
    obsc=obsc[:idx]+ostar.flatten().tolist() + obsc[idx:]
    
    ### Ignore: in progress faster image creation (slowest part of code)
    '''
    pos=list(zip(xpm,ypm,wlenm))
    uni,uni_inds,uni_counts=np.unique(pos,axis=0,return_inverse=True,return_counts=True)
    bbbb=max(uni_counts)
    # fluxm=fluxm.tolist()
    sortind=np.argsort(uni_inds)
    
    fluxm=fluxm[sortind]
    fluxm=fluxm.tolist()
    uni_counts=[0]+uni_counts.tolist()
    
    # fluzl=[fluxm[slice(uni_counts[i],uni_counts[i+1])] for i in tqdm.tqdm(range(len(uni)))]
    uni_countscum=np.cumsum(uni_counts)
    obsc=np.exp(tau*cool_cont[np.newaxis]*np.arange(bbbb)[:,np.newaxis])
    sumarry=np.array([(bbbb-uni_counts[i+1])*[0] + fluxm[slice(uni_countscum[i],uni_countscum[i+1])] for i in range(len(uni))])
    # sumarry=np.zeros((len(uni),max(uni_counts)))
    # t2=time.time()
    # print(t2-t1)
    # print(sumarry[1,-uni_counts[1]:])
    # print(uni)
    # for i in tqdm.tqdm(range(len(uni))): sumarry[i,-uni_counts[i]:]=fluxm[uni_inds==i]
    # print(sumarry.shape)
    obscs=np.array([obsc[:,i] for i in uni[:,-1]])
    sumarry=sumarry*obscs
    sumarry=np.sum(sumarry,axis=1)
    comp_IM[uni[:,-1],uni[:,-2],uni[:,-3]]=sumarry
    '''
    # Add points to image in order and apply obscuraton to points behind (slow)
    comp_IM=comp_IM.tolist()
    for xxc,yyc,ffc,wwc,ooc in zip(xpm,ypm,fluxm,wlenm,obsc):
        comp_IM[wwc][yyc][xxc]=comp_IM[wwc][yyc][xxc]*ooc +ffc

    norml=((np.sum(comp_IM,axis=(1,2))+norm*np.nansum(circles,axis=(1,2)))[0])
    comp_IM=np.array(comp_IM)/norml
    h = 6.63*10**(-34) #J*s=W*s=1*10^7erg=kg*m^2*s^-2
    c = 2.998*10**8 #m/s  to cm
    k = 1.38*10**(-23)  #J/K = kg*m^2*s / K
    bbover=(((2*h*c) / (wlen**3))/(np.exp(((h*c)/(k*20000*wlen)))-1))
    bbover=vloss2*bbover/bbover[0]
    


    ## Apply clumps outside image as normalistion
    comp_IM+=(bbover[:,np.newaxis,np.newaxis]+(norm*np.nansum(circles,axis=(1,2)))[:,np.newaxis,np.newaxis]/norml)/(nxy**2)
    # Add star to central pixel (Warning: no obscuration applied to star currently)
    #comp_IM[:,nxy//2,nxy//2]=bbscale
    
    
    return (np.transpose(comp_IM)/np.sum(comp_IM,axis=(1,2))).T#comp_IM/max(np.sum(comp_IM,axis=(1,2)))#(np.transpose(comp_IM)/np.sum(comp_IM,axis=(1,2))).T
 
 


def uv_model_hyp_chrome(alph,alphoff,beta,obslid,rsub,rbreak,Tstar,rstar,width,inc,ang,grad,coolobsc,numpo,frac,scale,wbase,Tcool,Vloss,Vloss2,u,v,uc,vc, nxy, dxy,wlen,seed):
    '''Produce a model uv data set.'''
    
    
    #Produce Image
    
    Im = hyp_cone_cube_fov(alph,alphoff,beta,obslid,rsub,rbreak,Tstar,rstar,width,inc,ang,grad,nxy,dxy,wlen,coolobsc,numpo,frac,scale,wbase,Tcool,Vloss,Vloss2,seed)


    pixel_scale=dxy
    #uunan=np.invert(np.isnan(u)).reshape((-1,len(wlen)))
    #ucnan=np.invert(np.isnan(uc)).reshape((-1,len(wlen),3)).transpose((0,2,1)).reshape((-1,len(wlen)))
    #vvnan=np.invert(np.isnan(v)).reshape((-1,len(wlen)))
    #vcnan=np.invert(np.isnan(vc)).reshape((-1,len(wlen),3)).transpose((0,2,1)).reshape((-1,len(wlen)))
    u=u.reshape((-1,len(wlen)))
    v=v.reshape((-1,len(wlen)))
    uc=uc.reshape((-1,len(wlen),3)).transpose((0,2,1)).reshape((-1,len(wlen)))
    vc=vc.reshape((-1,len(wlen),3)).transpose((0,2,1)).reshape((-1,len(wlen)))
    
    u_co=np.append(u,uc,axis=0)
    #u_conan=np.append(uunan,ucnan,axis=0)
    v_co=np.append(v,vc,axis=0)
    #v_conan=np.append(vvnan,vcnan,axis=0)
    #print(v_co.shape,v_conan.shape)
    #print(v_co[:,0][v_conan[:,0]].shape)
    
    # make complex vis, can probably be improved
    model_uv = np.array([double.sampleImage(Im[i],pixel_scale[i],u_co[:,i].copy(),v_co[:,i].copy(),origin='lower') for i in range(len(wlen))])

    #covert complex data to Visibility and Phase
    model_vis = (np.absolute(model_uv[:,:len(u)]))**2#

    model_vis=np.hstack(model_vis.T)
    
    model_cdiffphase=model_uv[:,len(u):].reshape((len(wlen),uc.shape[0]))
    model_cdiffphase=model_cdiffphase.T
    model_cdiffphase=model_cdiffphase.reshape((-1,3,len(wlen))).transpose((0,2,1)).reshape((-1,3))
#    model_vis=(np.sum(np.real(model_uv)**2)+np.sum(np.imag(model_uv)**2)-
#               np.sum(np.var(np.real(model_uv)))-np.sum(np.var(np.imag(model_uv))))-scale
#    model_phase = np.angle(model_uv) * 180/np.pi
    
    return model_vis, model_cdiffphase
    



def lnlike_hyp_chrome(theta, u_co, v_co, vis, viserr, u_phase, v_phase, c_phase,c_phaseerr, nxy, dxy,wlen,seed):
    ''' Log likelihood function. u_co, v_co, u_phase, v_phase, and wavelen are in meters; phases and angle should be in degrees.
    u_phase and v_phase should be of shape (n,3) for each set of coords with each set being [T3-T2,T2-T1,T3-T1].'''
    
    alph,alphoff,beta,obslid,rsub,rbreak,Tstar,rstar, width, inc, ang,grad, coolobsc,numpo,frac, lnf,lnfc, scale,wbase, Tcool, Vloss,Vloss2 = theta

    #Create model vis and phase
    #For closure phase handling option 2 (see below) append phase uv's to full list
    
    # utot=np.append(u_co, u_phase.flatten())
    # vtot=np.append(v_co, v_phase.flatten())
    
    
    model_vis, model_cdiffphase = uv_model_hyp_chrome(alph,alphoff,beta,obslid,rsub,rbreak,Tstar,rstar,width,inc,ang,grad,coolobsc,numpo,frac,scale,wbase,Tcool,Vloss,Vloss2,u_co,v_co,u_phase,v_phase, nxy, dxy,wlen,seed)
    
    #convert abs phase into closure phase
    

    trip = model_cdiffphase[:,0]*model_cdiffphase[:,1]*np.conj(model_cdiffphase[:,2])
    model_cphase = -np.angle(trip,deg=True) # Neg is VLTI standard, careful if not using VLTI
    
    
    if np.all(np.isnan(model_vis)) or np.all(np.isnan(model_cphase)):
        model_vis=np.zeros(model_vis.shape)-np.inf
        model_cphase=np.zeros(model_cphase.shape)-np.inf
        return -np.inf
    
# =============================================================================
#       Consider importing a dictionary due to the large number of inputs
#       Could import the fits files but may want custom vis and phase data
    
#       Also consider making robust to phases with no vis counterparts --- Done
# =============================================================================
    '''
    #option 1 for phase selection
    model_cphase=[]
    for i in range(len(u_phase)):
        u1diff=np.abs(u_co-u_phase[i][0])
        u2diff=np.abs(u_co-u_phase[i][1])
        u3diff=np.abs(u_co-u_phase[i][2])
        pos1_ind=np.argmin(u1diff)
        pos2_ind=np.argmin(u2diff)
        pos3_ind=np.argmin(u3diff)
        
        if u1diff[pos1_ind]>1e-3:
            u1diff=np.abs(-u_co-u_phase[i][0])
            pos1_ind=np.argmin(u1diff)
            if u1diff[pos1_ind]>1e-3:
                raise ValueError('No visibility associated with the '+str(i)+' index closure phase')
    '''

    #option 2 is appending the closure phase uv's to the model --more robust but could be slower ^^^^^^

            
    residuals=np.abs(c_phase-model_cphase)
    residuals[residuals>180]=np.abs(360-residuals[residuals>180])
    
    
    # May need a 2nd lnf
    inv_sigma2_vis = 1.0/(viserr**2 + model_vis**2*np.exp(2*lnf))
    inv_sigma2_cphase = 1.0/(c_phaseerr**2 + model_cphase**2*np.exp(2*lnfc))
    return -0.5*(np.nansum((vis-model_vis)**2*inv_sigma2_vis - np.log(inv_sigma2_vis))+
                 np.nansum((residuals)**2*inv_sigma2_cphase - np.log(inv_sigma2_cphase)))



  
def lnprior_hyp_chrome(theta):
    ''' log prior '''
    alph,alphoff,beta,obslid,rsub,rbreak,Tstar,rstar, width, inc, ang,grad, coolobsc,numpo,frac, lnf,lnfc, scale,wbase, Tcool, Vloss,Vloss2 = theta

    # create flat prior
    ####### Must be a good way to clean this up!
    #

    if not 0.00001<alph<5 or not 0<alphoff<2 or not 1e-6<beta<4000 or not 0.0000001<obslid<45 or not 0.01<rsub<400 or not 1<rbreak<50 or not -7<width<1 or  not 0<inc<90 or  not -180<ang<540 or not 0<grad<1 or not -15.0 < lnf < -1 or not -15.0 < lnfc < -1 or not -5.5 < coolobsc < 1 or not 90 < numpo < 20000 or not 500 < frac < 2000 or not 0.1<scale<1.1 or not 0< Tcool < 5 or not -100<=Vloss<0.0 or not 0.0<Vloss2<0.1 or not 0<wbase<400 or not 2000 < Tstar <20000 or not 0.1<rstar<2:
#    if not np.all(perc_resf>0.0) or not (np.sum(perc_resf)>0.999 and np.sum(perc_resf)<1.0001) or not (np.all((x_off-photx)>-10) and np.all((x_off-photx)<10)) or  not (np.all((y_off-photy)>-10) and np.all((y_off-photy)<10)) or not -15.0 < lnf < 15.0 or not 0.00 < scale < 0.3:
        return -np.inf
    return 0.0  



def lnprob_hype_chrome(theta, u_co, v_co, vis, viserr, u_phase, v_phase, c_phase,c_phaseerr, nxy, dxy,wlen,seed):
    ''' Log probability, introduce flattened array of all variables in order '''
    
    #Sort arguments here

    
    
    lp = lnprior_hyp_chrome(theta)-7.33859466
    
    
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_hyp_chrome(theta, u_co, v_co, vis, viserr, u_phase, v_phase, c_phase,c_phaseerr, nxy, dxy,wlen,seed)

def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

def RunGravEmcee(FileList,ESO323,paranal,NGauss,wav_regions,seed,pool,savedir):
#    m=0
    BICc=999999

    resarr=None 
    #pathway = 'clump_size/HD1426661e-0/'
    #resarr=np.load(pathway+'resarrlis.npy')[np.argmax(np.load(pathway+'Bays.npy'))]
    
    #alph,alphoff,beta,obslid,rsub,rbreak,Tstar, width, inc, ang,grad, coolobsc,numpo,frac, lnf,lnfc, scale,wbase, Tcool, Vloss,Vloss2 = theta
   #  resarr=np.array([ 2.66104314e-01,  2.66135853e-01,  1.68653726e+01,  4.18371695e+01,
   #  6.88121484e+00,  1.00689127e+00, 16500,
   #  -5.71221506e+00, 3.43730400e+01,  3.38516907e+02, 0.5, -2.66192805e+00,
   #  1.78266614e+04, 1500, -10,
   # -10,  4.11861440e-01,  2.49014157e+02,
   #  7.48646938e-01,  0.3,  0.05])
    # resarr=np.array([ 4.13514066e-01,  2.36376688e-01,  8.29960532e+02,  2.28392680e+01,
    #         5.03147848e+00,  1.31295051e+01,  1.32242438e+04, -2.42952913e+00,
    #         5.73809067e+01,  8.43802899e+01,  5.97050781e-01, -2.51773377e+00,
    #         1.63822620e+04,  1.80388234e+03, -9.03529850e+00, -9.98424475e+00,
    #         9.74271852e-01,  2.29425710e+02,  7.88313280e-01, -2.24129843e+00,
    #         1.57265096e-02])
    

    for ahsbcsbhcajb in [1]:
        #### Load all data, to be changed with new oifits library
        SciVis=np.array([])
        SciVisErr=np.array([])
        SciCPhase=np.array([])
        SciCPhaseErr=np.array([])
        U=np.array([])
        V=np.array([])
        vflag=np.array([],dtype=bool)
        cflag=np.array([],dtype=bool)
        wavelength=np.array([])

        num=0
        cnum=0
        wlens=[]
        for ll,lh,bins in wav_regions:
            wlens=np.append(wlens,np.linspace(ll,lh,bins+1))
        wavmed=np.zeros((len(FileList),len(wlens)-1))*np.nan
        for tnum,i in enumerate(FileList):
            opfi=oifits.open(i)
 
                
            wav=opfi.vis2[0].wavelength.eff_wave
            mask=np.zeros(len(wav),dtype=bool)
            for LL,LH,bins in wav_regions:
                maskl=wav>LL
                maskh=wav<LH
                mask+=maskl*maskh
            wavarg=np.digitize(wav[mask],wlens)
            for bi in range(len(wlens)-1):
                wavmed[tnum,bi]=np.nanmedian(wav[mask][wavarg==(bi+1)])
            #wavelength=np.append(wavelength,wav[mask])
            for j in range(len(opfi.vis2)):
                num+=1
            for j in range(len(opfi.t3)):
                cnum+=1

        wavmed=np.nanmedian(wavmed,axis=0)
        SciVis=np.zeros((num,len(wlens)-1))*np.nan
        SciVisErr=np.zeros((num,len(wlens)-1))*np.nan
        SciCPhase=np.zeros((cnum,len(wlens)-1))*np.nan
        SciCPhaseErr=np.zeros((cnum,len(wlens)-1))*np.nan
        U=np.zeros((num,len(wlens)-1))*np.nan
        V=np.zeros((num,len(wlens)-1))*np.nan
        Ucphase=np.zeros((cnum,len(wlens)-1,3))*np.nan
        Vcphase=np.zeros((cnum,len(wlens)-1,3))*np.nan
        vflag=np.ones((num,len(wlens)-1),dtype=bool)
        cflag=np.ones((cnum,len(wlens)-1),dtype=bool)
        num=0
        cnum=0
        for tnum,i in enumerate(FileList):
            opfi=oifits.open(i)
            for akjsbdkasd in [1]:
                
                wav=opfi.vis2[0].wavelength.eff_wave
                
                mask=np.zeros(len(wav),dtype=bool)
                for LL,LH,__bins in wav_regions:
                    maskl=wav>LL
                    maskh=wav<LH
                    mask+=maskl*maskh
                #wavarg=[np.argmin(np.abs(wlens-wi)) for wi in wav[mask]]
                wavarg=np.digitize(wav[mask],wlens)
                

                for j in range(len(opfi.vis2)):
        
                    
                    ## use interpolation for differing wavelength arrays - No
                    
                    
                    
                    # mask=maskl*maskh
                    #wavelength=np.append(wavelength,wav[mask])
                    # print(len(wav[mask]))
                    #mask[:MAH]=False
                    for bi in range(len(wlens)-1):
                        vflag=np.invert(opfi.vis2[j].flag[mask][wavarg==(bi+1)])
                        
                        SciVis[num,bi]=np.nanmedian(opfi.vis2[j]._vis2data[mask][wavarg==(bi+1)][vflag])
                        serr=np.sqrt(np.sum((opfi.vis2[j]._vis2err[mask][wavarg==(bi+1)][vflag])**2))/len(opfi.vis2[j]._vis2data[mask][wavarg==(bi+1)][vflag])
                        SciVisErr[num,bi]=np.sqrt(np.std(opfi.vis2[j]._vis2data[mask][wavarg==(bi+1)][vflag])**2+serr**2)
                   
                    #vflag[num,wavarg]=opfi['OI_VIS2',20].data['flag'][j][mask]
                    #SciVis[num,wavarg]=opfi['OI_VIS2',20].data['vis2data'][j][mask]
                    #SciVisErr[num,wavarg]=opfi['OI_VIS2',20].data['vis2err'][j][mask]
                    U[num,:]=opfi.vis2[j].ucoord/wavmed
                    V[num,:]=opfi.vis2[j].vcoord/wavmed
                    num+=1
                for j in range(len(opfi.t3)):
                    for bi in range(len(wlens)-1):
                        cflag=np.invert(opfi.t3[j].flag[mask][wavarg==(bi+1)])
                        if wav[0]<=7e-6:
                            SciCPhase[cnum,bi]=scipy.stats.circmean(opfi.t3[j]._t3phi[mask][wavarg==(bi+1)][cflag],low=-180,high=180,nan_policy='omit')
                        else:
                            SciCPhase[cnum,bi]=-scipy.stats.circmean(opfi.t3[j]._t3phi[mask][wavarg==(bi+1)][cflag],low=-180,high=180,nan_policy='omit')
                        SciCPhaseErr[cnum,bi]=np.sqrt(np.sum(opfi.t3[j]._t3phierr[mask][wavarg==(bi+1)][cflag]**2))
                        serr=np.sqrt(np.sum((opfi.t3[j]._t3phierr[mask][wavarg==(bi+1)][cflag])**2))/len(opfi.t3[j]._t3phi[mask][wavarg==(bi+1)][cflag])
                        SciCPhaseErr[cnum,bi]=np.sqrt(circstd(opfi.t3[j]._t3phi[mask][wavarg==(bi+1)][cflag],low=-180,high=180,nan_policy='omit')**2+serr**2)
                    #cflag[cnum,wavarg]=opfi['OI_T3',20].data['flag'][j][mask]
                    #SciCPhase[cnum,wavarg]=opfi['OI_T3',20].data['t3phi'][j][mask]
                    #SciCPhaseErr[cnum,wavarg]=opfi['OI_T3',20].data['t3phierr'][j][mask]
                    Ucphase[cnum,:,0]=opfi.t3[j].u1coord/wavmed
                    Vcphase[cnum,:,0]=opfi.t3[j].v1coord/wavmed
                    Ucphase[cnum,:,1]=opfi.t3[j].u2coord/wavmed
                    Vcphase[cnum,:,1]=opfi.t3[j].v2coord/wavmed
                    Ucphase[cnum,:,2]=(opfi.t3[j].u1coord+opfi.t3[j].u2coord)/wavmed
                    Vcphase[cnum,:,2]=(opfi.t3[j].v1coord+opfi.t3[j].v2coord)/wavmed
                    #cflag[cnum,wavarg]=opfi['OI_T3'].data['flag'][j][mask]
                    #SciCPhase[cnum,wavarg]=opfi['OI_T3'].data['t3phi'][j][mask]
                    #SciCPhaseErr[cnum,wavarg]=opfi['OI_T3'].data['t3phierr'][j][mask]
                    cnum+=1

            '''
            except:
                wav=opfi['OI_WAVELENGTH'].data['eff_wave']
                
                mask=np.zeros(len(wav),dtype=bool)
                for LL,LH,__bins in wav_regions:
                    maskl=wav>LL
                    maskh=wav<LH
                    mask+=maskl*maskh
                #wavarg=[np.argmin(np.abs(wlens-wi)) for wi in wav[mask]]
                wavarg=np.digitize(wav[mask],wlens)
                if num==4 or num==6:
                    print(i)
                for j in range(opfi['OI_VIS2'].data['vis2data'].shape[0]):
        
                    
                    ## use interpolation for differing wavelength arrays - No
                    
                    for bi in range(len(wlens)-1):
                        vflag=np.invert(opfi['OI_VIS2'].data['flag'][j][mask][wavarg==(bi+1)])
                        SciVis[num,bi]=np.nanmedian(opfi['OI_VIS2'].data['vis2data'][j][mask][wavarg==(bi+1)][vflag])
                        SciVisErr[num,bi]=np.nanmedian(opfi['OI_VIS2'].data['vis2err'][j][mask][wavarg==(bi+1)][vflag])
                    
                    
                    # mask=maskl*maskh
                    #wavelength=np.append(wavelength,wav[mask])
                    # print(len(wav[mask]))
                    #mask[:MAH]=False
                    #vflag[num,wavarg]=opfi['OI_VIS2'].data['flag'][j][mask]
                    #SciVis[num,wavarg]=opfi['OI_VIS2'].data['vis2data'][j][mask]
                    #SciVisErr[num,wavarg]=opfi['OI_VIS2'].data['vis2err'][j][mask]
                    U[num,:]=opfi['OI_VIS2'].data['ucoord'][j]/wavmed
                    V[num,:]=opfi['OI_VIS2'].data['vcoord'][j]/wavmed
                    num+=1
                try:
                    for j in range(opfi['OI_T3'].data['t3phi'].shape[0]):
                        # if opfi[0].header['HIERARCH ESO DET NAME']=='MATISSE-LM':
                        if opfi['OI_WAVELENGTH'].data['EFF_WAVE'][0]>7e-6:
                            for bi in range(len(wlens)-1):
                                cflag=np.invert(opfi['OI_T3'].data['flag'][j][mask][wavarg==(bi+1)])
                                SciCPhase[cnum,bi]=scipy.stats.circmean(-opfi['OI_T3'].data['t3phi'][j][mask][wavarg==(bi+1)][cflag],low=-180,high=180,nan_policy='omit')
                                SciCPhaseErr[cnum,bi]=np.nanmedian(opfi['OI_T3'].data['t3phierr'][j][mask][wavarg==(bi+1)][cflag])
                        else:
                            for bi in range(len(wlens)-1):
                                cflag=np.invert(opfi['OI_T3'].data['flag'][j][mask][wavarg==(bi+1)])
                                SciCPhase[cnum,bi]=scipy.stats.circmean(opfi['OI_T3'].data['t3phi'][j][mask][wavarg==(bi+1)][cflag],low=-180,high=180,nan_policy='omit')
                                SciCPhaseErr[cnum,bi]=np.nanmedian(opfi['OI_T3'].data['t3phierr'][j][mask][wavarg==(bi+1)][cflag])
                        Ucphase[cnum,:,0]=opfi['OI_T3'].data['u1coord'][j]/wavmed
                        Vcphase[cnum,:,0]=opfi['OI_T3'].data['v1coord'][j]/wavmed
                        Ucphase[cnum,:,1]=opfi['OI_T3'].data['u2coord'][j]/wavmed
                        Vcphase[cnum,:,1]=opfi['OI_T3'].data['v2coord'][j]/wavmed
                        Ucphase[cnum,:,2]=(opfi['OI_T3'].data['u1coord'][j]+opfi['OI_T3'].data['u2coord'][j])/wavmed
                        Vcphase[cnum,:,2]=(opfi['OI_T3'].data['v1coord'][j]+opfi['OI_T3'].data['v2coord'][j])/wavmed
                        #cflag[cnum,wavarg]=opfi['OI_T3'].data['flag'][j][mask]
                        #SciCPhase[cnum,wavarg]=opfi['OI_T3'].data['t3phi'][j][mask]
                        #SciCPhaseErr[cnum,wavarg]=opfi['OI_T3'].data['t3phierr'][j][mask]
                        cnum+=1
                        # elif opfi[0].header['HIERARCH ESO DET NAME']=='MATISSE-N':
                        #     for bi in range(len(wlens)-1):
                        #         cflag=np.invert(opfi['OI_T3'].data['flag'][j][mask][wavarg==(bi+1)])
                        #         SciCPhase[cnum,bi]=scipy.stats.circmean(-opfi['OI_T3'].data['t3phi'][j][mask][wavarg==(bi+1)][cflag],low=-180,high=180,nan_policy='omit')
                        #         SciCPhaseErr[cnum,bi]=np.nanmedian(opfi['OI_T3'].data['t3phierr'][j][mask][wavarg==(bi+1)][cflag])
                        #     #cflag[cnum,wavarg]=opfi['OI_T3'].data['flag'][j][mask]
                        #     #SciCPhase[cnum,wavarg]=-opfi['OI_T3'].data['t3phi'][j][mask]
                        #     #SciCPhaseErr[cnum,wavarg]=opfi['OI_T3'].data['t3phierr'][j][mask]
                        #     cnum+=1
                except(KeyError):
                    pass
            '''        

    
        #wlens=np.unique(signif(wavelength,6))
        wlens=dc(wavmed)
        nanws=np.argwhere(np.isnan(wlens))
        SciVis=np.delete(SciVis,nanws,1)
        SciVisErr=np.delete(SciVisErr,nanws,1)
        SciCPhase=np.delete(SciCPhase,nanws,1)
        SciCPhaseErr=np.delete(SciCPhaseErr,nanws,1)
        U=np.delete(U,nanws,1)
        V=np.delete(V,nanws,1)
        Ucphase=np.delete(Ucphase,nanws,1)
        Vcphase=np.delete(Vcphase,nanws,1)
        
        wlens=np.delete(wlens,nanws)
        SciVis[SciVisErr>0.2]=np.nan
        SciVisErr[SciVisErr<1e-3]=1e-3
        SciVis[SciVis<0.0]=np.nan
        SciCPhaseErr[SciCPhaseErr<0.1]=0.1
        #SciVis[vflag]=np.nan
        #SciCPhase[cflag]=np.nan
        print(wlens)
        SciVis=np.hstack(SciVis)
        SciVisErr=np.hstack(SciVisErr)
        nvs=np.argwhere(np.isnan(SciVis))

        U=np.hstack(U)
        V=np.hstack(V)
        SciCPhase=np.hstack(SciCPhase)
        ncf=np.argwhere(np.isnan(SciCPhase))
        Ucphase=np.vstack(Ucphase)
        Vcphase=np.vstack(Vcphase)
        #Ucphase=Ucphase.reshape((-1,len(wlens),3))
        #SciCPhase=SciCPhase.reshape((-1,len(wlens)))
        Ucphase[np.isnan(SciCPhase)]=np.nan
        Vcphase[np.isnan(SciCPhase)]=np.nan
        SciCPhaseErr=np.hstack(SciCPhaseErr)
        '''
        # Recalc uv pos if needed
        for num,i in enumerate(FileList):
            
            hd=fits.open(i)
            if hd[0].header['INSTRUME']=='GRAVITY':
                # for h in wav_index:
                for j in dc(hd['OI_VIS2',20].data['STA_INDEX']):
                    sta_ind.append(j)
                for j in dc(hd['OI_T3',20].data['STA_INDEX']):
                    phase_sta_ind.append(j)
            
                for j in dc(hd['OI_ARRAY'].data['STA_NAME']):
                    sta_array.append(j)
                for j in dc(hd['OI_ARRAY'].data['STA_INDEX']):
                    sta_array_ind.append(j)
                    
                for j in dc(hd['OI_VIS2',20].data['MJD']):
                    timeobs_vis_array.append(Time(j,format='mjd'))
                for j in dc(hd['OI_T3',20].data['MJD']):    
                    timeobs_phase_array.append(Time(j,format='mjd'))
                    
                for j in hd['OI_VIS2',20].data['INT_TIME']:
                    int_time_vis.append(datetime.timedelta(seconds=j/2.0))
                for j in dc(hd['OI_T3',20].data['INT_TIME']):
                    int_time_phase.append(datetime.timedelta(seconds=j/2.0))
            else:
                # for h in wav_index:
                for j in dc(hd['OI_VIS2'].data['STA_INDEX']):
                    sta_ind.append(j)
                for j in dc(hd['OI_T3'].data['STA_INDEX']):
                    phase_sta_ind.append(j)
            
                for j in dc(hd['OI_ARRAY'].data['STA_NAME']):
                    sta_array.append(j)
                for j in dc(hd['OI_ARRAY'].data['STA_INDEX']):
                    sta_array_ind.append(j)
                    
                for j in dc(hd['OI_VIS2'].data['MJD']):
                    timeobs_vis_array.append(Time(j,format='mjd'))
                for j in dc(hd['OI_T3'].data['MJD']):    
                    timeobs_phase_array.append(Time(j,format='mjd'))
                    
                for j in hd['OI_VIS2'].data['INT_TIME']:
                    int_time_vis.append(datetime.timedelta(seconds=j/2.0))
                for j in dc(hd['OI_T3'].data['INT_TIME']):
                    int_time_phase.append(datetime.timedelta(seconds=j/2.0))
            hd.close()
     
        sta_ind=np.array(sta_ind)
        phase_sta_ind=np.array(phase_sta_ind)
        sta_array=np.array(sta_array)
        sta_array_ind=np.array(sta_array_ind)
        
        U=np.zeros((len(SciVis)))
        V=np.zeros((len(SciVis)))
        
        
        
        
        
        
        #New uv coords for vis data
        

        for i in range(len(SciVis)//len(wlens)):
            # for j in wav_index:
                
            #Find station names
            T2i = np.argwhere(sta_array_ind==sta_ind[i][0])[0][0]
            T1i = np.argwhere(sta_array_ind==sta_ind[i][1])[0][0]
            
            #Generate half way times for each observation
            timeobs=timeobs_vis_array[i]
            time=Time(timeobs.datetime + int_time_vis[i])
            
            # Asjust the observational parameters to the new time
            altnight=AltAz(obstime=time,location=paranal)
            ESO323night=ESO323.transform_to(altnight)
            azr=ESO323night.az.radian
            altr=ESO323night.alt.radian
            
            T2 = sta_array[T2i]
            T1 = sta_array[T1i]
            
            #Fill uv arrays
            U[i*len(wlens):(i+1)*len(wlens)] = UVpoint(ESO323, paranal, [T2,T1], azr, altr, time)[1]/wlens
            V[i*len(wlens):(i+1)*len(wlens)] = UVpoint(ESO323, paranal, [T2,T1], azr, altr, time)[2]/wlens
            
                
        if 0 in U:
            raise ValueError('Mismatched UV and Vis length')
        
        
        #New uv coords for cphase data
        
        Ucphase=np.zeros((len(SciCPhase),3))
        Vcphase=np.zeros((len(SciCPhase),3))
            
        for i in range(len(Ucphase)//len(wlens)):
            
            #Find station names
            T3i = np.argwhere(sta_array_ind==phase_sta_ind[i][0])[0][0]
            T2i = np.argwhere(sta_array_ind==phase_sta_ind[i][1])[0][0]
            T1i = np.argwhere(sta_array_ind==phase_sta_ind[i][2])[0][0]
            
            #Generate half way times for each observation
            timeobs=timeobs_phase_array[i]
            time=Time(timeobs.datetime + int_time_phase[i])
            
            # Asjust the observational parameters to the new time
            altnight=AltAz(obstime=time,location=paranal)
            ESO323night=ESO323.transform_to(altnight)
            azr=ESO323night.az.radian
            altr=ESO323night.alt.radian
            
            T3 = sta_array[T3i]
            T2 = sta_array[T2i]
            T1 = sta_array[T1i]
            
            #Fill uv arrays
            Ucphase[i*len(wlens):(i+1)*len(wlens),0] = UVpoint(ESO323, paranal, [T3,T2], azr, altr, time)[1]/wlens
            Vcphase[i*len(wlens):(i+1)*len(wlens),0] = UVpoint(ESO323, paranal, [T3,T2], azr, altr, time)[2]/wlens
            Ucphase[i*len(wlens):(i+1)*len(wlens),1] = UVpoint(ESO323, paranal, [T2,T1], azr, altr, time)[1]/wlens
            Vcphase[i*len(wlens):(i+1)*len(wlens),1] = UVpoint(ESO323, paranal, [T2,T1], azr, altr, time)[2]/wlens
            Ucphase[i*len(wlens):(i+1)*len(wlens),2] = UVpoint(ESO323, paranal, [T3,T1], azr, altr, time)[1]/wlens
            Vcphase[i*len(wlens):(i+1)*len(wlens),2] = UVpoint(ESO323, paranal, [T3,T1], azr, altr, time)[2]/wlens
            
                
    '''
    # =============================================================================
    #     Make the uv arrays symmetric? Probably just a waste of time
    # =============================================================================
        
 
        nparam=22
        ndim, nwalkers = 22, 2000
        #nparam=7
        #ndim, nwalkers = 7, 126
        
        #Declare image parameters
        # nxy,dxy = double.get_image_size(np.append(U, Ucphase.flatten()), np.append(V, Vcphase.flatten()),f_max=10)
        uunan=np.invert(np.isnan(U)).reshape((-1,len(wlens)))
        ucnan=np.invert(np.isnan(Ucphase)).reshape((-1,len(wlens),3)).transpose((0,2,1)).reshape((-1,len(wlens)))
        vvnan=np.invert(np.isnan(V)).reshape((-1,len(wlens)))
        vcnan=np.invert(np.isnan(Vcphase)).reshape((-1,len(wlens),3)).transpose((0,2,1)).reshape((-1,len(wlens)))
        u=U.reshape((-1,len(wlens)))
        v=V.reshape((-1,len(wlens)))
        uc=Ucphase.reshape((-1,len(wlens),3)).transpose((0,2,1)).reshape((-1,len(wlens)))
        vc=Vcphase.reshape((-1,len(wlens),3)).transpose((0,2,1)).reshape((-1,len(wlens)))
    
        u_co=np.append(u,uc,axis=0)
        u_conan=np.append(uunan,ucnan,axis=0)
        v_co=np.append(v,vc,axis=0)
        v_conan=np.append(vvnan,vcnan,axis=0)
        print(u_co.shape,u_conan.shape)
        
        siz=np.array([double.get_image_size(u_co[:,i][u_conan[:,i]].copy(), v_co[:,i][v_conan[:,i]].copy(),f_max=10) for i in range(len(wlens))])
        nxy,dxy=int(min(siz[:,0])),siz[:,1]
        Ucphase[np.isnan(SciCPhase)]=Ucphase[-1]
        Vcphase[np.isnan(SciCPhase)]=Ucphase[-1]
        print(list(zip(wlens,siz[:,0],siz[:,1])))
        print(nxy,dxy)
        #nxy *= 4
        # dxy/=2
#        dxy /= 100
#if not 0.009<alph<25 or not 1e-6<beta<150 or not 0.0000001<obslid<90-(np.abs(((np.arctan2(1,a)+np.pi/2)%np.pi)-np.pi/2))*180/np.pi or not obslid < 30 or not 1<zoom<4 or not 0<rsub<5 or not 0.1 <a<4 or not 0.0001<width<a/2 or not 0.00001<nps<5 or  not 0<inc<90 or  not 0<ang<360 or not -100<grad<100 or not -15.0 < lnf < 15.0 or not 0.00 < coolobsc < 1000 or not 0.7<scale<1:
#alph,beta,obslid,zoom,rsub, a, width, nps, inc, ang,grad, coolobsc, lnf, scale
#alph,beta,obslid,zoom,rsub, a, width, nps, inc, ang,grad, coolobsc,numpo,frac,windlaw, lnf, scale
#alph,alphoff,beta,obslid,zoom,rsub,rbreak,dbreak, a, width, nps,npsoff, inc, ang,grad, coolobsc,numpo,frac,windlaw, lnf,lnfc, scale, Tcool,Vloss
#alph,alphoff,beta,obslid,rsub,rbreak,Tstar, width, inc, ang,grad, coolobsc,numpo,frac,windlaw, lnf,lnfc, scale,wbase, Tcool, Vloss,Vloss2 = theta
        if type(resarr)==type(None):
            pos=[]
            for position in range(nwalkers):
                #Generate the initial params
                res=[]
                res.append(np.random.uniform(0.01,2))
                res.append(np.random.uniform(0.0,1.5))
                res.append(np.random.uniform(0.1,1000.0))
                res.append(np.random.uniform(5,45.0))
                res.append(np.random.uniform(0.1,20))
                res.append(np.random.uniform(1,80))
                res.append(np.random.uniform(8000,18000))
                res.append(np.random.uniform(0.01,1)) # star rad (mas)
                res.append(np.random.uniform(-4,0))
                res.append(np.random.uniform(5,45))
                res.append(np.random.uniform(200,300))
                res.append(np.random.uniform(0.05,0.95))
                #res.append(np.random.uniform(-0.5,1.5))
                res.append(np.random.uniform(-4.5,-2.5))
                res.append(np.random.uniform(100,20000))
                res.append(np.random.uniform(1400,1900))
                res.append(np.random.uniform(-10,-9))
                res.append(np.random.uniform(-10,-9))
                res.append(np.random.uniform(0.5,1.0))
                res.append(np.random.uniform(0.5,300))
                res.append(np.random.uniform(0.05,1.1))
                res.append(np.random.uniform(-15,-8))
                res.append(np.random.uniform(0.0,0.01))
    
    
    #alph,alphoff,beta,obslid,rsub,rbreak,Tstar, width, inc, ang,grad, coolobsc,numpo,frac, lnf,lnfc, scale,wbase, Tcool, Vloss,Vloss2 = theta
    ############## 
                tries=0
                while np.isinf(lnprior_hyp_chrome(res)):
                    res=[]
                    res.append(np.random.uniform(0.01,2)) #alpha or disk temp power
                    res.append(np.random.uniform(0.0,1.5)) #alpha offset after break
                    res.append(np.random.uniform(1,1000.0)) #beta (disk xz stdev) #mas
                    res.append(np.random.uniform(5,45.0)) # disk half opening angle (deg)
                    res.append(np.random.uniform(0.1,20)) #sub rad in mas
                    res.append(np.random.uniform(1,80)) # disk temp power break radius (rsub)
                    res.append(np.random.uniform(8000,18000)) # star temp (K)
                    res.append(np.random.uniform(0.01,1)) # star rad (mas)
                    res.append(np.random.uniform(-4,0)) # smooth obsc scaling (log10)
                    res.append(np.random.uniform(5,45)) # inclination (deg), zero edge-on
                    res.append(np.random.uniform(250,300)) # PA (deg)
                    res.append(np.random.uniform(0.05,0.95)) # dust mix carbon fraction
                    #res.append(np.random.uniform(-0.5,1.5))
                    res.append(np.random.uniform(-4.5,-2.5)) # clump obsc scaling
                    res.append(np.random.uniform(100,20000)) # number of clumps
                    res.append(np.random.uniform(1400,1900)) # Inner dust temp (K)
                    res.append(np.random.uniform(-10,-9)) # vis error underestimation 
                    res.append(np.random.uniform(-10,-9))# cphase error underestimation
                    res.append(np.random.uniform(0.5,1.0)) # moon phase effect scaling
                    res.append(np.random.uniform(0.5,300)) # disk base temperature
                    res.append(np.random.uniform(0.05,1.1)) # clump radius (mas)
                    res.append(np.random.uniform(-15,-8)) # stellar shortest wavelength total flux fraction
                    res.append(np.random.uniform(0,0.01)) # visibility zero scaling
                    tries+=1
                    if tries>=10000:
                        print('failed to get starting pos')
                        exit() 
                #alph,alphoff,beta,obslid,zoom,rsub,rbreak,dbreak, a, width, nps,npsoff, inc, ang,grad, coolobsc,numpo,frac,windlaw, lnf,lnfc, scale, Tcool,Vloss                   
                 
                
                pos.append(res)
                
        else:
            res=dc(resarr)

            resoff=2e-1*res
            pos=np.array([res + resoff*np.random.randn(ndim) for i in range(nwalkers)])

        print('start')
        print(res)
        
        
        nsteps = 5000
        #filename = 'clump_size/BC_Ob1'+str(wav_regions[0][0])[:4]+'/sampler_'+str(seed)+'.h5'
        filename=savedir+'/sampler_'+str(seed)+'.h5'
        #filename=emcee.backends.HDFBackend('/run/user/1001/gvfs/sftp:host=licallo.oca.eu/data/home/jleftley/Allwvs/clump_size/BeC11.4e/sampler_'+str(seed)+'.h5',read_only=True)

        # print(pool.is_master())

        if isfile(filename):
            backend = emcee.backends.HDFBackend(filename)
            print('Predicted steps needed:',np.mean(backend.get_autocorr_time(tol=0))*100)
            print('Steps requested:',nsteps)
            print('Steps completed:',backend.iteration)
            nsteps=nsteps-backend.iteration
                
            pos=None
        else:
            backend = emcee.backends.HDFBackend(filename)
            backend.reset(nwalkers, ndim)
            print('new run')
            
        if nsteps<=0:
            print("done already, skipped")
            return U,V,Ucphase,Vcphase,SciVis,SciCPhase,SciVisErr,SciCPhaseErr,backend,nxy,dxy,wlens,9999999
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_hype_chrome, args=(U, V, SciVis, SciVisErr,Ucphase, Vcphase, SciCPhase,SciCPhaseErr, nxy, dxy,wlens,seed),pool=pool,moves=[(emcee.moves.StretchMove(), 0.1),(emcee.moves.DEMove(), 0.7),(emcee.moves.DESnookerMove(), 0.2)], backend=backend)
        sampler.run_mcmc(pos,nsteps,progress=True,skip_initial_state_check=True)

    return U,V,Ucphase,Vcphase,SciVis,SciCPhase,SciVisErr,SciCPhaseErr,sampler,nxy,dxy,wlens,9999999

BICcurr=99999

### Setup MPI, remove for python multiprocessing
# pool=MPIPool()
# print(pool.is_master())
# if not pool.is_master():
#     pool.wait()
#     sys.exit(0)

print('loading files')

# f=glob('/home/nikmos/Desktop/PhD/Fitting_Modelling/NGC1068_MATISSE-v1.0/VioletaGamez-NGC1068_MATISSE-d60cd61/*.fits')
f=np.array(glob('../../HD45677.Lband.data/data/*.oifits')) ## L Fits files of 1068
#f=glob('./ALLAGN_2022/*VV*20*.fits')
f=np.append(f,glob('../../Reduc_KHH/*.oifits')) ## N Fits files of 1068
f=np.append(f,glob('../../fscmaH/*.fits')) ## N Fits files of 1068
# f=np.append(f,glob('../../NGC_1068_K/AT/*.fits')) ## K Fits files of 1068
# f=glob('../../Downloads/outdirinterpDec3/*.fits')
f=np.array(glob('Beauty_Obj1/*fits'))
f=np.array(glob('/srv/jleftley/RY_Tau/RY*'))
#f=glob('../../Downloads/wetransfer_1068-data_2022-01-20_1507/*.fits')
paranal=EarthLocation.of_site('paranal')
ESO323=SkyCoord.from_name('RY Tau')
print('loaded files')


# All bands together without M band
listw=[[(1.4e-6, 1.8e-6, 2),(2.0e-6,2.5e-6,2),(3.2e-06, 3.9e-06,3),(8.40e-6,12.7e-6,4)]]

# All bands together with M band
listw=[[(1.4e-6, 1.8e-6, 2),(2.0e-6,2.5e-6,2),(3.2e-06, 3.9e-06,3),(4.6e-06, 4.9e-06,1),(8.40e-6,12.7e-6,4)]]
listw=[[(1.4e-6, 1.8e-6, 2),(3.2e-06, 3.9e-06,2),(8.40e-6,12.7e-6,2)]]
listw=[[(3.2e-06, 3.9e-06,2),(4.6e-06, 4.9e-06,1),(8.40e-6,12.7e-6,5)]]

# Each band seperately
#listw=[[(1.4e-6, 1.8e-6, 2)],[(2.0e-6,2.5e-6,2)],[(3.2e-06, 3.9e-06,3)],[(8.40e-6,12.7e-6,4)]]


seeds=np.random.randint(10,1e6,1)
#seeds=np.array([624609])
#seeds=np.array([194305])
#seeds=[361525,491544,252371,748855,523474]
seeds=[186016,461424,491267,559870,637198]
seeds=[47]

#seeds=[]
#seeds=[583295]
for wavcent in listw:#[[(11.8e-6,12.0e-6)]]:#[[(3.2e-6,3.3e-6)],[(3.3e-6,3.4e-6)],[(3.4e-6,3.5e-6)],[(3.5e-6,3.6e-6)],[(3.7e-6,3.8e-6)],[(3.8e-6,3.9e-6)],[(4.6e-6,4.7e-6)],[(4.7e-6,4.8e-6)],[(4.8e-6,4.9e-6)]]:#[[(3.2e-6,3.8e-6),(4.6e-6,4.9e-6)]]:#[3.25,3.50,3.60,3.70,3.80,4.62,4.67,4.77,4.87]:
    #savedirorg='clump_size/ind_edge_low'+str(wavcent[0][0])[:4]
    savedir='clump_size/RY_Tau'+str(wavcent[0][0])[:4] # Output directory
    print(savedir)
    Bays=[]
    BICarray=[]
    allminres=[]
    NGauss=22
    samplerlist=[]
    reslist=[]
    resarrlis=[]
    try:
        mkdir(savedir)
    except:
        pass  
    #seeds=[np.load(savedirorg+'/seeds.npy')[np.argmax(np.load(savedirorg+'/Bays.npy'))]]
    for seed in seeds:
        #wavcent=[(2.0e-06, 2.3e-06)]
        result=RunGravEmcee(f,ESO323,paranal,NGauss,wavcent,seed,Pool(30),savedir)
        U,V,Ucphase,Vcphase,SciVis,SciCPhase,SciVisErr,SciCPhaseErr,sampler,nxy,dxy,wlens,BIC=result
        #samplerlist.append(sampler)
        #samples = sampler.chain[:, 200:, :].reshape((-1, 14))
        #BICs=[np.log(len(SciVis)+len(SciCPhase))*3*(NGauss)-2*lnlike_point(np.reshape(i[:-4],(3,int(len(i[:-4])/3))), i[-1], i[-4],i[-3],i[-2], U, V, SciVis, SciVisErr, Ucphase, Vcphase, SciCPhase,SciCPhaseErr, nxy, dxy) for i in tqdm(samples)]    # BICarray.append(BICs)
        #allminres.append(samples[np.argmin(BICs)])
        # allminres.append(np.median(samples,axis=0))
        # if np.min(BICs)<BICcurr:
        #reslist.append(result)    
        #resultfinal=dc(result)
        resarrlis.append(sampler.get_chain(flat=True)[np.argmax(sampler.get_log_prob(flat=True))])
        Bays.append(np.max(sampler.get_log_prob(flat=True)))
            # array=np.array(samples[np.argmin(BICs)])
            # BICcurr=dc(np.min(BICs))
        # print(NGauss,': ',np.min(BICs))
        
        seedlabel=str(seed)
        U,V,Ucphase,Vcphase,SciVis,SciCPhase,SciVisErr,SciCPhaseErr,sampler,nxy,dxy,wlens,BIC=result#reslist[i]
        resarr=sampler.get_chain(flat=True)[np.argmax(sampler.get_log_prob(flat=True))]
        #resarra=sampler.get_chain()[0]
        #lp=[lnprob_hype_chrome(theta, U, V, SciVis, SciVisErr, Ucphase, Vcphase, SciCPhase,SciCPhaseErr, nxy, dxy,wlens,seed) for theta in tqdm(resarra)]
        #resarr=resarra[np.argmax(lp)]
        #lp=sampler.get_log_prob(flat=True)
        #lp[np.invert(np.isfinite(lp))]=np.nan
        print(resarr)
        print(seed)
        print(np.argmax(sampler.get_log_prob(flat=True)))
        print(np.max(sampler.get_log_prob(flat=True)))
        #print(np.nanargmax(lp))
        #print(np.nanmax(lp))
        print(np.any(np.isfinite(sampler.get_log_prob(flat=True))))
        ndim=21
        fig, axes = plt.subplots(1, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
#alph,alphoff,beta,obslid,rsub,rbreak,Tstar, width, inc, ang,grad, coolobsc,numpo,frac, lnf,lnfc, scale,wbase, Tcool, Vloss,Vloss2 = theta
        labels = [r'$\alpha_\mathrm{disk}$',r'$\alpha_\mathrm{off}$','beta','obslid',r'$r_\mathrm{sub}$',r'$r_\mathrm{break}$',r'$\T_\mathrm{star}$',r'$\eta_\mathrm{sm}$','inc',r'$\theta$','grad','coolobsc','numpo','windfrac','lnf','lnfc','clump grad',r'T$_\mathrm{b}$','clump size','Vloss','Vloss2']
        for i in range(1):
            ax = axes
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_sample_UT_s'+seedlabel+'.pdf')
        #resarr=np.median(flat_samples,axis=0)
        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        labels = [r'$\alpha_\mathrm{disk}$',r'$\alpha_\mathrm{off}$','beta','obslid',r'$r_\mathrm{sub}$',r'$r_\mathrm{break}$',r'$T_\mathrm{star}$',r'$\eta_\mathrm{sm}$','inc',r'$\theta$','grad','coolobsc','numpo','windfrac','lnf','lnfc','clump grad',r'T$_\mathrm{b}$','clump size','Vloss','Vloss2']
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_samples_UT_s'+seedlabel+'.pdf')
        #resarr=np.median(flat_samples,axis=0)
        PAr=np.arctan2(U,V)%np.pi
        plt.figure()
        modvis,modcdiffphase=uv_model_hyp_chrome(*resarr[:15],*resarr[-5:],U,V,Ucphase,Vcphase,nxy,dxy,wlens,seed)
        modvis[np.isnan(SciVis)]=np.nan
        plt.errorbar(np.sqrt(U**2+V**2)*1e-6,SciVis,SciVisErr,c='k',linestyle='none',marker='None',zorder=1)
        plt.scatter(np.sqrt(U**2+V**2)*1e-6,SciVis,c=PAr,cmap='jet',vmin=0,vmax=np.pi,zorder=2)
        #modvis,modcdiffphase=uv_model_hyp_chrome(*resarr[:12],resarr[-1],U,V,Ucphase,Vcphase,nxy,dx
        #y,wlens)
        trip = modcdiffphase[:,0]*modcdiffphase[:,1]*np.conj(modcdiffphase[:,2])
        modcphase = -np.angle(trip,deg=True)
        modcphase[np.isnan(SciCPhase)]=np.nan
        plt.errorbar(np.sqrt(U**2+V**2)*1e-6,modvis,linestyle='none',marker='o')
        plt.xlabel('Baseline (M$\lambda$)')
        plt.ylabel('V$^2$')
        cb=plt.colorbar()
        cb.set_label('PA (rad)')
        plt.title(str(wlens[0])[:4])
        plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_V2T_UT_s'+seedlabel+'.pdf')
        plt.figure()
        plt.errorbar(Ucphase[:,0],SciCPhase,SciCPhaseErr,linestyle='none',marker='o')
        plt.errorbar(Ucphase[:,0],modcphase,linestyle='none',marker='o')
        plt.ylim(-180,180)
        plt.xlabel('U1 (M$\lambda$)')
        plt.ylabel('Closure Phase (deg)')    
        plt.title(str(wlens[0])[:4])
        plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_CPhaseT_UT_s'+seedlabel+'.pdf')
        plt.figure()
        #modvis,modcdiffphase=uv_model_hyp_chrome(*resarr[:12],resarr[-1],U,V,Ucphase,Vcphase,nxy,dxy,wlens,seed)
        plt.errorbar(np.sqrt(U**2+V**2)*1e-6,SciVis-modvis,SciVisErr,c='k',linestyle='none',marker='None',zorder=1)
        plt.scatter(np.sqrt(U**2+V**2)*1e-6,SciVis-modvis,c=PAr,cmap='jet',vmin=0,vmax=np.pi,zorder=2)
        #modvis,modcdiffphase=uv_model_hyp_chrome(*resarr[:12],resarr[-1],U,V,Ucphase,Vcphase,nxy,dx
        #y,wlens)
        #trip = modcdiffphase[:,0]*modcdiffphase[:,1]*np.conj(modcdiffphase[:,2])
        #modcphase = -np.angle(trip,deg=True)
        #plt.errorbar(np.sqrt(U**2+V**2)*1e-6,modvis,linestyle='none',marker='o')
        plt.xlabel('Baseline (M$\lambda$)')
        plt.ylabel('V$^2$')
        cb=plt.colorbar()
        cb.set_label('PA (rad)')
        plt.title(str(wlens[0])[:4])
        plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_V2T_Residuals_UT_s'+seedlabel+'.pdf')
        plt.figure()
        plt.errorbar(Ucphase[:,0],SciCPhase-modcphase,SciCPhaseErr,linestyle='none',marker='o')
        plt.ylim(-180,180)
        #plt.scatter(Ucphase[:,0],modcphase)
        plt.xlabel('U1 (M$\lambda$)')
        plt.ylabel('Closure Phase (deg)')
        plt.title(str(wlens[0])[:4])
        plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_CPhaseT_Residuals_UT_s'+seedlabel+'.pdf')  
        Im=hyp_cone_cube_fov(*resarr[:12],nxy,dxy,wlens,*resarr[12:15],*resarr[-5:],seed)
        plt.figure()
        #plt.imshow(Im[0],origin='lower',extent=(nxy*dxy[0]*3600*1000*180/np.pi/2,-nxy*dxy[0]*3600*1
        #000*180/np.pi/2,-nxy*dxy[0]*3600*1000*180/np.pi/2,nxy*dxy[0]*3600*1000*180/np.pi/2))
        plt.xlabel('$\delta$RA (mas)')
        plt.ylabel('$\delta$Dec (mas)')
        yfwhm=135
        xfwhm=58    
        kern=Gaussian2DKernel(x_stddev=0.5*wlens[0]/(2.335*xfwhm*dxy[0]),y_stddev=0.5*wlens[0]/(2.335*yfwhm*dxy[0]),theta=48*np.pi/180,x_size=nxy+1,y_size=nxy+1)
        #im=convolve(Im[0],kern)
        #plt.imshow(im,origin='lower',extent=(nxy*dxy[0]*3600*1000*180/np.pi/2,-nxy*dxy[0]*3600*1000*180/np.pi/2,-nxy*dxy[0]*3600*1000*180/np.pi/2,nxy*dxy[0]*3600*1000*180/np.pi/2))
        #plt.title(str(wlens[0])[:4])
        #plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_ImConvT_UT_s'+seedlabel+'.pdf')    
        plt.figure()
        plt.imshow(np.log(Im[0]),vmax=np.log(np.sort(Im[0],axis=None)[-2]*1.5),origin='lower',extent=(nxy*dxy[0]*3600*1000*180/np.pi/2,-nxy*dxy[0]*3600*1000*180/np.pi/2,-nxy*dxy[0]*3600*1000*180/np.pi/2,nxy*dxy[0]*3600*1000*180/np.pi/2))

        plt.title(str(wlens[0])[:4])
        plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_ImT_UT_s'+seedlabel+'.pdf')
        #fig =corner.corner(sampler.get_chain(discard=0, flat=True),quantiles=[0.16, 0.5, 0.84],show_titles=True,labels=labels,truths=resarr)
        #plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_corner_s'+seedlabel+'.pdf')
        #imconv=[convolve(Im[i],Gaussian2DKernel(x_stddev=0.5*wlens[i]/(2.335*xfwhm*dxy[i]),y_stddev=0.5*wlens[i]/(2.335*yfwhm*dxy[i]),theta=48*np.pi/180,x_size=nxy+1,y_size=nxy+1)) for i in range(len(wlens))]
        fig,axs=plt.subplots(4,4, figsize=(10, 10*1.414))
        #axs=np.hstack(axs)
        #fig=np.hstack(fig)
        for j in range(axs.shape[0]):
            for k in range(axs.shape[1]):
                try:
                    i=j*axs.shape[0]+k
                    #plt.subplot(3,6,i+1)
                    axs[j,k].imshow(Im[i]**0.6,vmax=(np.sort(Im[0],axis=None)[-2]*1.5)**0.6,origin='lower',extent=(nxy*dxy[i]*3600*1000*180/np.pi/2,-nxy*dxy[i]*3600*1000*180/np.pi/2,-nxy*dxy[i]*3600*1000*180/np.pi/2,nxy*dxy[i]*3600*1000*180/np.pi/2))
                    axs[j,k].legend(title=r'$\lambda =$'+str(wlens[i]*1e6)[:4])
                except:
                    continue
        fig.supylabel(r'$\delta$Dec (mas)')
        fig.supxlabel(r'$\delta$Ra (mas)')
        fig.tight_layout()
        plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_IMall_s'+seedlabel+'.pdf')
        fig,axs=plt.subplots(4,4, figsize=(10, 10*1.414),sharey='all',sharex='all')
        #axs=np.hstack(axs)
        #fig=np.hstack(fig)
        for j in range(axs.shape[0]):
            for k in range(axs.shape[1]):
                try:
                    i=j*axs.shape[0]+k
                    axs[j,k].errorbar(np.sqrt(U.reshape((-1,len(wlens)))[:,i]**2+V.reshape((-1,len(wlens)))[:,i]**2)*wlens[i],SciVis.reshape((-1,len(wlens)))[:,i],SciVisErr.reshape((-1,len(wlens)))[:,i],c='k',linestyle='none',marker='None',zorder=1)
                    axs[j,k].scatter(np.sqrt(U.reshape((-1,len(wlens)))[:,i]**2+V.reshape((-1,len(wlens)))[:,i]**2)*wlens[i],SciVis.reshape((-1,len(wlens)))[:,i],c=PAr.reshape((-1,len(wlens)))[:,i],cmap='jet',vmin=0,vmax=np.pi,zorder=2)
                    axs[j,k].legend(title=r'$\lambda =$'+str(wlens[i]*1e6)[:4])
                    axs[j,k].errorbar(np.sqrt(U.reshape((-1,len(wlens)))[:,i]**2+V.reshape((-1,len(wlens)))[:,i]**2)*wlens[i],modvis.reshape((-1,len(wlens)))[:,i],linestyle='none',marker='x',c='r')
                    axs[j,k].set_yscale('log')
                    axs[j,k].set_ylim(0.0001,1)
                except:
                    continue
        

        fig.supylabel(r'V$^2$')
        fig.supxlabel(r'Baseline (m)')
        fig.tight_layout()
        plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_Vall_s'+seedlabel+'.pdf')
        fig,axs=plt.subplots(4,4, figsize=(10, 10*1.414),sharey='all',sharex='all')
        #axs=np.hstack(axs)
        #fig=np.hstack(fig)
        for j in range(axs.shape[0]):
            for k in range(axs.shape[1]):
                try:
                    i=j*axs.shape[0]+k
                    arrowang=np.arctan2(Ucphase,Vcphase)
                    arrowbs=np.sqrt(Ucphase**2+Vcphase**2)
                    long=np.argsort(arrowbs,axis=-1)
                    arrowperim=np.sum(arrowbs,axis=-1)
                    points=np.array([np.mean(arrowang[k,long[k]][1:]) for k in range(len(long))])
                    axs[j,k].errorbar(arrowperim.reshape((-1,len(wlens)))[:,i]*wlens[i],SciCPhase.reshape((-1,len(wlens)))[:,i],SciCPhaseErr.reshape((-1,len(wlens)))[:,i],linestyle='none',c='k',zorder=0)
                    axs[j,k].scatter(arrowperim.reshape((-1,len(wlens)))[:,i]*wlens[i],SciCPhase.reshape((-1,len(wlens)))[:,i],marker='o',c=points.reshape((-1,len(wlens)))[:,i],cmap='inferno',zorder=1)
                    axs[j,k].errorbar(arrowperim.reshape((-1,len(wlens)))[:,i]*wlens[i],modcphase.reshape((-1,len(wlens)))[:,i],linestyle='none',marker='x',c='r',zorder=2)
                    axs[j,k].set_ylim((-40,40))
                    axs[j,k].legend(title=r'$\lambda =$'+str(wlens[i]*1e6)[:4])
                except:
                    continue
        fig.supylabel(r'$\Phi_c$ (deg)')
        fig.supxlabel(r'Triangle Perimeter (m)')
        fig.tight_layout()
        plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_Call_s'+seedlabel+'.pdf')
        fig,axs=plt.subplots(4,4, figsize=(10, 10*1.414),sharey='all',sharex='all')
        #axs=np.hstack(axs)
        #fig=np.hstack(fig)
        for j in range(axs.shape[0]):
            for k in range(axs.shape[1]):
                try:
                    i=j*axs.shape[0]+k
                    arrowang=np.arctan2(Ucphase,Vcphase)
                    arrowbs=np.sqrt(Ucphase**2+Vcphase**2)
                    long=np.argsort(arrowbs,axis=-1)
                    arrowperim=np.sum(arrowbs,axis=-1)
                    points=np.array([np.mean(arrowang[k,long[k]][1:]) for k in range(len(long))])
                    axs[j,k].errorbar(points.reshape((-1,len(wlens)))[:,i]*180/np.pi,SciCPhase.reshape((-1,len(wlens)))[:,i],SciCPhaseErr.reshape((-1,len(wlens)))[:,i],linestyle='none',c='k',zorder=0)
                    axs[j,k].scatter(points.reshape((-1,len(wlens)))[:,i]*180/np.pi,SciCPhase.reshape((-1,len(wlens)))[:,i],marker='o',c=arrowperim.reshape((-1,len(wlens)))[:,i]*wlens[i],cmap='inferno',zorder=1)
                    axs[j,k].errorbar(points.reshape((-1,len(wlens)))[:,i]*180/np.pi,modcphase.reshape((-1,len(wlens)))[:,i],linestyle='none',marker='x',c='r',zorder=2)
                    axs[j,k].set_ylim((-40,40))
                    axs[j,k].legend(title=r'$\lambda =$'+str(wlens[i]*1e6)[:4])
                except(IndexError):
                    continue
        fig.supylabel(r'$\Phi_c$ (deg)')
        fig.supxlabel(r'Arrow Pointing (rad)')
        fig.tight_layout()
        plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_Call_alt_s'+seedlabel+'.pdf')
        exit()
        plt.close('all')

        resarrm=np.nanmedian(sampler.get_chain(discard=500, flat=True),axis=0)
        plt.figure()
        modvis,modcdiffphase=uv_model_hyp_chrome(*resarrm[:15],*resarrm[-5:],U,V,Ucphase,Vcphase,nxy,dxy,wlens,seed)
        modvis[np.isnan(SciVis)]=np.nan
        plt.errorbar(np.sqrt(U**2+V**2)*1e-6,SciVis,SciVisErr,c='k',linestyle='none',marker='None',zorder=1)
        plt.scatter(np.sqrt(U**2+V**2)*1e-6,SciVis,c=PAr,cmap='jet',vmin=0,vmax=np.pi,zorder=2)
        #modvis,modcdiffphase=uv_model_hyp_chrome(*resarr[:12],resarr[-1],U,V,Ucphase,Vcphase,nxy,dx
        #y,wlens)
        trip = modcdiffphase[:,0]*modcdiffphase[:,1]*np.conj(modcdiffphase[:,2])
        modcphase = -np.angle(trip,deg=True)
        modcphase[np.isnan(SciCPhase)]=np.nan
        plt.errorbar(np.sqrt(U**2+V**2)*1e-6,modvis,linestyle='none',marker='o')
        plt.xlabel('Baseline (M$\lambda$)')
        plt.ylabel('V$^2$')
        cb=plt.colorbar()
        cb.set_label('PA (rad)')
        plt.title(str(wlens[0])[:4])
        plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_V2T_UT_med_s'+seedlabel+'.pdf')
        plt.figure()
        plt.errorbar(Ucphase[:,0],SciCPhase,SciCPhaseErr,linestyle='none',marker='o')
        plt.errorbar(Ucphase[:,0],modcphase,linestyle='none',marker='o')
        plt.ylim(-180,180)
        plt.xlabel('U1 (M$\lambda$)')
        plt.ylabel('Closure Phase (deg)')    
        plt.title(str(wlens[0])[:4])
        plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_CPhaseT_UT_med_s'+seedlabel+'.pdf')
        plt.figure()
        #modvis,modcdiffphase=uv_model_hyp_chrome(*resarr[:12],resarr[-1],U,V,Ucphase,Vcphase,nxy,dxy,wlens,seed)
        plt.errorbar(np.sqrt(U**2+V**2)*1e-6,SciVis-modvis,SciVisErr,c='k',linestyle='none',marker='None',zorder=1)
        plt.scatter(np.sqrt(U**2+V**2)*1e-6,SciVis-modvis,c=PAr,cmap='jet',vmin=0,vmax=np.pi,zorder=2)
        #modvis,modcdiffphase=uv_model_hyp_chrome(*resarr[:12],resarr[-1],U,V,Ucphase,Vcphase,nxy,dx
        #y,wlens)
        #trip = modcdiffphase[:,0]*modcdiffphase[:,1]*np.conj(modcdiffphase[:,2])
        #modcphase = -np.angle(trip,deg=True)
        #plt.errorbar(np.sqrt(U**2+V**2)*1e-6,modvis,linestyle='none',marker='o')
        plt.xlabel('Baseline (M$\lambda$)')
        plt.ylabel('V$^2$')
        cb=plt.colorbar()
        cb.set_label('PA (rad)')
        plt.title(str(wlens[0])[:4])
        plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_V2T_Residuals_UT_med_s'+seedlabel+'.pdf')
        plt.figure()
        plt.errorbar(Ucphase[:,0],SciCPhase-modcphase,SciCPhaseErr,linestyle='none',marker='o')
        plt.ylim(-180,180)
        #plt.scatter(Ucphase[:,0],modcphase)
        plt.xlabel('U1 (M$\lambda$)')
        plt.ylabel('Closure Phase (deg)')
        plt.title(str(wlens[0])[:4])
        plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_CPhaseT_Residuals_UT_med_s'+seedlabel+'.pdf')    
        Im=hyp_cone_cube_fov(*resarrm[:12],nxy,dxy,wlens,*resarrm[12:15],*resarrm[-5:],seed)
        plt.figure()
        #plt.imshow(Im[0],origin='lower',extent=(nxy*dxy[0]*3600*1000*180/np.pi/2,-nxy*dxy[0]*3600*1
        #000*180/np.pi/2,-nxy*dxy[0]*3600*1000*180/np.pi/2,nxy*dxy[0]*3600*1000*180/np.pi/2))
        plt.xlabel('$\delta$RA (mas)')
        plt.ylabel('$\delta$Dec (mas)')
        yfwhm=135
        xfwhm=58    
        kern=Gaussian2DKernel(x_stddev=0.5*wlens[0]/(2.335*xfwhm*dxy[0]),y_stddev=0.5*wlens[0]/(2.335*yfwhm*dxy[0]),theta=48*np.pi/180,x_size=nxy+1,y_size=nxy+1)
        #im=convolve(Im[0],kern)
        #plt.imshow(im,origin='lower',extent=(nxy*dxy[0]*3600*1000*180/np.pi/2,-nxy*dxy[0]*3600*1000*180/np.pi/2,-nxy*dxy[0]*3600*1000*180/np.pi/2,nxy*dxy[0]*3600*1000*180/np.pi/2))
        plt.title(str(wlens[0])[:4])
        plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_ImConvT_UT_med_s'+seedlabel+'.pdf')    
        plt.figure()
        plt.imshow(Im[0],origin='lower',extent=(nxy*dxy[0]*3600*1000*180/np.pi/2,-nxy*dxy[0]*3600*1000*180/np.pi/2,-nxy*dxy[0]*3600*1000*180/np.pi/2,nxy*dxy[0]*3600*1000*180/np.pi/2))
        plt.title(str(wlens[0])[:4])
        plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_ImT_UT_med_s'+seedlabel+'.pdf')
        #imconv=[convolve(Im[i],Gaussian2DKernel(x_stddev=0.5*wlens[i]/(2.335*xfwhm*dxy[i]),y_stddev=0.5*wlens[i]/(2.335*yfwhm*dxy[i]),theta=48*np.pi/180,x_size=nxy+1,y_size=nxy+1)) for i in range(len(wlens))]
        fig,axs=plt.subplots(3,6, figsize=(10, 7))
        axs=np.hstack(axs)
        #fig=np.hstack(fig)
        for i in range(len(wlens)):
            #plt.subplot(3,6,i+1)
            axs[i].imshow(np.log10(Im[i]),origin='lower',extent=(nxy*dxy[i]*3600*1000*180/np.pi/2,-nxy*dxy[i]*3600*1000*180/np.pi/2,-nxy*dxy[i]*3600*1000*180/np.pi/2,nxy*dxy[i]*3600*1000*180/np.pi/2))
            axs[i].set_title(str(wlens[i]*1e6)[:4])
        plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_IMall_med_s'+seedlabel+'.pdf')
        fig,axs=plt.subplots(3,6, figsize=(10, 7))
        axs=np.hstack(axs)
        #fig=np.hstack(fig)
        for i in range(len(wlens)):
            axs[i].errorbar(np.sqrt(U.reshape((-1,len(wlens)))[:,i]**2+V.reshape((-1,len(wlens)))[:,i]**2)*1e-6,SciVis.reshape((-1,len(wlens)))[:,i],SciVisErr.reshape((-1,len(wlens)))[:,i],c='k',linestyle='none',marker='None',zorder=1)
            axs[i].scatter(np.sqrt(U.reshape((-1,len(wlens)))[:,i]**2+V.reshape((-1,len(wlens)))[:,i]**2)*1e-6,SciVis.reshape((-1,len(wlens)))[:,i],c=PAr.reshape((-1,len(wlens)))[:,i],cmap='jet',vmin=0,vmax=np.pi,zorder=2)
            axs[i].set_title(str(wlens[i]*1e6)[:4])
            axs[i].errorbar(np.sqrt(U.reshape((-1,len(wlens)))[:,i]**2+V.reshape((-1,len(wlens)))[:,i]**2)*1e-6,modvis.reshape((-1,len(wlens)))[:,i],linestyle='none',marker='x',c='r')
        plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_Vall_med_s'+seedlabel+'.pdf')
        fig,axs=plt.subplots(3,6, figsize=(10, 7))
        axs=np.hstack(axs)
        #fig=np.hstack(fig)
        for i in range(len(wlens)):
            arrowang=np.arctan2(Ucphase,Vcphase)
            arrowbs=np.sqrt(Ucphase**2+Vcphase**2)
            long=np.argsort(arrowbs,axis=-1)
            arrowperim=np.sum(arrowbs,axis=-1)
            points=np.array([np.mean(arrowang[k,long[k]][1:]) for k in range(len(long))])
            axs[i].errorbar(arrowperim.reshape((-1,len(wlens)))[:,i],SciCPhase.reshape((-1,len(wlens)))[:,i],SciCPhaseErr.reshape((-1,len(wlens)))[:,i],linestyle='none',c='k',zorder=0)
            axs[i].scatter(arrowperim.reshape((-1,len(wlens)))[:,i],SciCPhase.reshape((-1,len(wlens)))[:,i],marker='o',c=points.reshape((-1,len(wlens)))[:,i],cmap='inferno',zorder=1)
            axs[i].errorbar(arrowperim.reshape((-1,len(wlens)))[:,i],modcphase.reshape((-1,len(wlens)))[:,i],linestyle='none',marker='x',c='r',zorder=2)
            axs[i].set_ylim((-180,180))
            axs[i].set_title(str(wlens[i]*1e6)[:4])
        plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_Call_med_s'+seedlabel+'.pdf')
        fig,axs=plt.subplots(4,4, figsize=(10, 10*1.414),sharey='all',sharex='all')
        #axs=np.hstack(axs)
        #fig=np.hstack(fig)
        for j in range(axs.shape[0]):
            for k in range(axs.shape[1]):
                try:
                    i=j*axs.shape[0]+k
                    arrowang=np.arctan2(Ucphase,Vcphase)
                    arrowbs=np.sqrt(Ucphase**2+Vcphase**2)
                    long=np.argsort(arrowbs,axis=-1)
                    arrowperim=np.sum(arrowbs,axis=-1)
                    points=np.array([np.mean(arrowang[k,long[k]][1:]) for k in range(len(long))])
                    axs[j,k].errorbar(points.reshape((-1,len(wlens)))[:,i]*180/np.pi,SciCPhase.reshape((-1,len(wlens)))[:,i],SciCPhaseErr.reshape((-1,len(wlens)))[:,i],linestyle='none',c='k',zorder=0)
                    axs[j,k].scatter(points.reshape((-1,len(wlens)))[:,i]*180/np.pi,SciCPhase.reshape((-1,len(wlens)))[:,i],marker='o',c=arrowperim.reshape((-1,len(wlens)))[:,i]*wlens[i],cmap='inferno',zorder=1)
                    axs[j,k].errorbar(points.reshape((-1,len(wlens)))[:,i]*180/np.pi,modcphase.reshape((-1,len(wlens)))[:,i],linestyle='none',marker='x',c='r',zorder=2)
                    axs[j,k].set_ylim((-180,180))
                    axs[j,k].legend(title=r'$\lambda =$'+str(wlens[i]*1e6)[:4])
                except(IndexError):
                    continue
        fig.supylabel(r'$\Phi_c$ (deg)')
        fig.supxlabel(r'Arrow Pointing (rad)')
        fig.tight_layout()
        plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_Call_med_alt_s'+seedlabel+'.pdf')
        plt.close('all')
    np.save(savedir+'/'+'resarrlis.npy',resarrlis)
    np.save(savedir+'/'+'Bays.npy',Bays)
    np.save(savedir+'/'+'seeds.npy',seeds)
    #np.save(savedir+'/'+'samples.npy',sampler,allow_pickle=False)
    
    best=np.argmax(Bays)


    modvall=[]
    mcphaseall=[]
    seedlabel='best'
    seed=seeds[best]
    #resarr=np.median(flat_samples,axis=0)
    PAr=np.arctan2(U,V)%np.pi
    for i in range(len(seeds)):
        #seed='42_bugfix'
        #U,V,Ucphase,Vcphase,SciVis,SciCPhase,SciVisErr,SciCPhaseErr,sampler,nxy,dxy,wlens,BIC=reslist[i]
        resarrt=resarrlis[i]
        #resarr=np.median(flat_samples,axis=0)
        PAr=np.arctan2(U,V)%np.pi
        #plt.figure()
        modvis,modcdiffphase=uv_model_hyp_chrome(*resarrt[:19],*resarrt[-6:],U,V,Ucphase,Vcphase,nxy,dxy,wlens,int(seeds[i]))
        trip = modcdiffphase[:,0]*modcdiffphase[:,1]*np.conj(modcdiffphase[:,2])
        modcphase = -np.angle(trip,deg=True)
        modvis[np.isnan(SciVis)]=np.nan
        modcphase[np.isnan(SciCPhase)]=np.nan
        modvall.append(modvis)
        mcphaseall.append(modcphase)
        #print(s[i])




    resarr=resarrlis[best]    
    plt.figure()
    modvis,modcdiffphase=uv_model_hyp_chrome(*resarr[:19],*resarr[-6:],U,V,Ucphase,Vcphase,nxy,dxy,wlens,seed)
    modvis[np.isnan(SciVis)]=np.nan
    plt.errorbar(np.sqrt(U**2+V**2)*1e-6,SciVis,SciVisErr,c='k',linestyle='none',marker='None',zorder=1)
    plt.scatter(np.sqrt(U**2+V**2)*1e-6,SciVis,c=PAr,cmap='jet',vmin=0,vmax=np.pi,zorder=2)
    #modvis,modcdiffphase=uv_model_hyp_chrome(*resarr[:12],resarr[-1],U,V,Ucphase,Vcphase,nxy,dx
    #y,wlens)
    plt.plot(np.sqrt(U**2+V**2)*1e-6,np.transpose(modvall),linestyle='none',alpha=0.03,marker='x',c='k',label='Indv.')
    trip = modcdiffphase[:,0]*modcdiffphase[:,1]*np.conj(modcdiffphase[:,2])
    modcphase = -np.angle(trip,deg=True)
    modcphase[np.isnan(SciCPhase)]=np.nan
    plt.errorbar(np.sqrt(U**2+V**2)*1e-6,modvis,linestyle='none',marker='x',c='purple',label='Best')
    #modvism,modcdiffphasem=uv_model_hyp_chrome(*resarrm[:15],resarrm[-1],U,V,Ucphase,Vcphase,nxy,dxy,wlens,seed)
    plt.errorbar(np.sqrt(U**2+V**2)*1e-6,np.median(modvall,axis=0),np.std(modvall,axis=0),linestyle='none',marker='^',c='green',label='Median')
    plt.xlabel('Baseline (M$\lambda$)')
    plt.ylabel('V$^2$')
    plt.legend()
    cb=plt.colorbar()
    cb.set_label('PA (rad)')
    plt.title(str(wlens[0])[:4])
    plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_V2T_UT_s'+seedlabel+'.pdf')
    plt.figure()
    #plt.scatter(Ucphase[:,0],SciCPhase)
    #plt.scatter(Ucphase[:,0],modcphase)
    plt.errorbar(Ucphase[:,0],SciCPhase,SciCPhaseErr,linestyle='none',marker='o')
    plt.plot(Ucphase[:,0],np.transpose(mcphaseall),alpha=0.03,linestyle='none',marker='x',c='k',label='Indv.')
    plt.errorbar(Ucphase[:,0],modcphase,linestyle='none',marker='x',c='purple',label='Best')
    plt.errorbar(Ucphase[:,0],np.median(mcphaseall,axis=0),np.std(mcphaseall,axis=0),linestyle='none',marker='^',c='green',label='Median')
    plt.legend()
    plt.xlabel('U1 (M$\lambda$)')
    plt.ylabel('Closure Phase (deg)')
    plt.title(str(wlens[0])[:4])
    plt.title(str(seed))
    plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_CPhaseT_UT_s'+seedlabel+'.pdf')
    Im=hyp_cone_cube_fov(*resarr[:16],nxy,dxy,wlens,*resarr[16:20],*resarr[-6:],seed)
    plt.figure()
    #plt.imshow(Im[0],origin='lower',extent=(nxy*dxy[0]*3600*1000*180/np.pi/2,-nxy*dxy[0]*3600*1
    #000*180/np.pi/2,-nxy*dxy[0]*3600*1000*180/np.pi/2,nxy*dxy[0]*3600*1000*180/np.pi/2))
    plt.xlabel('$\delta$RA (mas)')
    plt.ylabel('$\delta$Dec (mas)')
    yfwhm=135
    xfwhm=58
    kern=Gaussian2DKernel(x_stddev=0.5*wlens[0]/(2.335*xfwhm*dxy[0]),y_stddev=0.5*wlens[0]/(2.335*yfwhm*dxy[0]),theta=48*np.pi/180,x_size=nxy+1,y_size=nxy+1)
    #im=convolve(Im[0],kern)
    #plt.imshow(im,origin='lower',extent=(nxy*dxy[0]*3600*1000*180/np.pi/2,-nxy*dxy[0]*3600*1000*180/np.pi/2,-nxy*dxy[0]*3600*1000*180/np.pi/2,nxy*dxy[0]*3600*1000*180/np.pi/2))
    #plt.title(str(wlens[0])[:4])
    #plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_ImConvT_UT_s'+seedlabel+'.pdf')
    plt.figure()
    plt.imshow(Im[0],origin='lower',extent=(nxy*dxy[0]*3600*1000*180/np.pi/2,-nxy*dxy[0]*3600*1000*180/np.pi/2,-nxy*dxy[0]*3600*1000*180/np.pi/2,nxy*dxy[0]*3600*1000*180/np.pi/2))
    plt.title(str(wlens[0])[:4])
    plt.savefig(savedir+'/'+str(wlens[0])[:4]+'_1068_ImT_UT_s'+seedlabel+'.pdf')

exit()
