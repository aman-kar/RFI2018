#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

Author: Aman Kar
Date:   2 August 2018

Execution Format : python SK_data.py 'path to .npy file'

Script Goal:-

To execute the Spectral Kurtosis (SK) Estimator function on a numpy array
file (npy file) and also calculate the Spectral Kurtosis Thresholds. The
SK Estimator will be evaluated for every fine frequency channel.


Credit: Majority of the script was initially created by Evan Smith and
has hence since been modified to satisfy different script goals

Variables in Use:-


infie          : Name of file
k|nfreq        : Number of frequency channels (combine with time resolution to get frequency resolution)
n|nint         : Number of integrations averaged before dumping to dyn_spec
pol            : Pol X (0) or Pol Y (1)

a              : 2D input np array spectrograph of power.Vertical is frequency channel, Horizontal as time. 
                 Shape: (numChans,numInts)
m|nspec        : number of integrations. Should be able to read from len(a)
n              : sub-sums of integrations. Will stay as 1
d              : shape parameter Will stay as 1

realpol        : Real Component of X/Y Polarisation
imagpol        : Imaginary Component of X/Y Polarisation

tsData         : Stores the data from numpy array file
tsLen          : Length of the data array
s1             : Accumulation of instantenous PSD estimates
s2             : Square of the sum of the PSD estimates
sk_est         : Record SK Estimator Value for the given fine frequency channel

upper      
ut
upperThreshold : Upper threshold function output values

lower          
lt
lowerThreshold : Lower threshold function output values

moment_1   
moment_2
moment_3
moment_4       : Generalized SK Statistical Moments 

alpha
beta
delta          : Pearson III Parameters

error_4        : Fourth Moment Error
min_value      : Minimum SK estimator value excluding the centre DC offset

in_arr         : Temporary complex buffer that stores real and imaginary values
out_arr        : FFT Output from in_arr complex buffer
accum          : Converts complex voltage values to Power and accumulates
spec_list      : Power Spectra
dyn_spec       : Dynamic Spectrum or Spectrogram with nspec power spectra, each with nfreq frequency channels
                 This data array can be thought of power values with time along the x-axis and frequency along Y-axis


"""

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import sys
import scipy.special
from matplotlib import rcParams

infile = sys.argv[1]

k = int(raw_input("Number of fine frequency channels? (k value) : "))
pol = int(raw_input("Type of Polarisation? (0 for X | 1 for Y) : "))
n=1
d=1



#Returns 1D np array of SK estimator

#Load .npy files and apply SK Estimator
#plot spectrogram

rcParams.update({'figure.autolayout' : True})
rcParams.update({'axes.formatter.useoffset' : False})

# get the data
tsData = np.load(infile)
#trunc_tsData = tsData[:27092480]
tsLen = tsData.shape[0]

# required polarization channels
realpol = 2*pol
imagpol = realpol+1

# create 512 time bins of 1024 spectra, each with 32 integrations.
# there may be a few bytes not used.
nfreq = k
nint  = n
nspec = int(tsLen / (nfreq * nint))
m=int(raw_input("Enter number of integrations (m value) (must be less than equal to "+str(nspec)+") : "))

if(m<=nspec):
    nspec=m


# empty list of power spectra
spec_list = []

if m*k > 1024*24570:
    sys.exit()

#SK Estimator Function by Evan Smith
def SK_EST(a,n):
    m=a.shape[1]
    #m=ints
    nchans=a.shape[0]
    d=1
    print('Shape: ',a.shape)
    print('Nchans: ',nchans)
    print('M: ',m)
    print('N: ',n)
    print('d: ',d)
    #make s1 and s2 as defined by whiteboard (by 2010b)
    #s2 definition will probably throw error if n does not integer divide m
    #s1=sum(a[chan,:]) same for old s2
    s1=np.sum(a,axis=1)
    #s2=sum(np.sum(a[chan,:].reshape(-1,n)**2,axis=1))
    s2=np.sum(a**2,axis=1)
    #record sk estimator
    sk_est = ((m*n*d+1)/(m-1))*((m*s2)/(s1**2)-1)
    return sk_est

#Threshold functions
#Taken from Nick Joslyn's helpful code https://github.com/NickJoslyn/helpful-BL/blob/master/helpful_BL_programs.py
def upperRoot(x, moment_2, moment_3, p):
    upper = np.abs( (1 - scipy.special.gammainc( (4 * moment_2**3)/moment_3**2, (-(moment_3-2*moment_2**2)/moment_3 + x)/(moment_3/2/moment_2)))-p)
    return upper

def lowerRoot(x, moment_2, moment_3, p):
    lower = np.abs(scipy.special.gammainc( (4 * moment_2**3)/moment_3**2, (-(moment_3-2*moment_2**2)/moment_3 + x)/(moment_3/2/moment_2))-p)
    return lower

def spectralKurtosis_thresholds(M, N = n, d = d, p = 0.0013499):
    Nd = N * d
    #Statistical moments
    moment_1 = 1
    moment_2 = ( 2*(M**2) * Nd * (1 + Nd) ) / ( (M - 1) * (6 + 5*M*Nd + (M**2)*(Nd**2)) )
    moment_3 = ( 8*(M**3)*Nd * (1 + Nd) * (-2 + Nd * (-5 + M * (4+Nd))) ) / ( ((M-1)**2) * (2+M*Nd) *(3+M*Nd)*(4+M*Nd)*(5+M*Nd))
    moment_4 = ( 12*(M**4)*Nd*(1+Nd)*(24+Nd*(48+84*Nd+M*(-32+Nd*(-245-93*Nd+M*(125+Nd*(68+M+(3+M)*Nd)))))) ) / ( ((M-1)**3)*(2+M*Nd)*(3+M*Nd)*(4+M*Nd)*(5+M*Nd)*(6+M*Nd)*(7+M*Nd) )
    #Pearson Type III Parameters
    delta = moment_1 - ( (2*(moment_2**2))/moment_3 )
    beta = 4 * ( (moment_2**3)/(moment_3**2) )
    alpha = moment_3 / (2 * moment_2)
    
    error_4 = np.abs( (100 * 3 * beta * (2+beta) * (alpha**4)) / (moment_4 - 1) )
    x = [1]
    upperThreshold = scipy.optimize.newton(upperRoot, x[0], args = (moment_2, moment_3, p))
    lowerThreshold = scipy.optimize.newton(lowerRoot, x[0], args = (moment_2, moment_3, p))
    return lowerThreshold, upperThreshold

print("processing ", str(nspec), "spectra...")

for s in range(nspec):
    #print("spectrum: ", s)
    winStart = s * (nfreq * nint)
    accum = np.zeros(nfreq)
    for i in range(nint):
        start = winStart + i * nfreq
        end = start + nfreq
        in_arr = np.zeros((nfreq), dtype=np.complex_)
        in_arr.real = tsData[start:end, realpol]
        in_arr.imag = tsData[start:end, imagpol]
        out_arr = np.fft.fftshift(np.fft.fft(in_arr))
        accum += np.abs(out_arr)**2
    spec_list.append(accum/nint)


# convert back to numpy array and transpose to desired order
dyn_spec = np.transpose(np.asarray(spec_list))

#------------------------------------------------------------

sk_result = SK_EST(dyn_spec,n)

lt,ut = spectralKurtosis_thresholds(np.float(nspec), N = np.float(n), d = 1, p = 0.0013499)

#------------------------------------------------------------


min_value=np.concatenate((sk_result[:len(sk_result)//2],sk_result[len(sk_result)//2+1:]),axis=0)



print('SK_value: '+str(np.min(min_value)))

plt.subplots(2,1, figsize=(9,9))
plt.subplot(211)
plt.plot(np.average(dyn_spec,axis=1),'r-')
plt.xlim(0,len(dyn_spec))
plt.gca().invert_xaxis()
plt.ylabel('Power')
plt.title(infile+'_'+str(nspec)+'_'+str(k)+'_'+str(pol))

plt.subplot(212)
plt.plot(sk_result,'b+')
plt.xlim(0,len(sk_result))
plt.gca().invert_xaxis()
plt.title(infile+'_'+str(nspec)+'_'+str(k)+'_'+str(pol))
plt.plot(np.zeros(k)+ut,'r:')
plt.plot(np.zeros(k)+lt,'r:')
plt.xlabel(str(k)+' fine frequncy channels')
plt.ylabel('SK Estimator Value')


plt.text(np.argmin(min_value),(np.max(sk_result)+np.min(sk_result))/2,str(np.min(min_value)))
plt.savefig(infile+'_'+str(m)+'_'+str(k)+'_'+str(pol)+'_sk.png')
plt.show()



