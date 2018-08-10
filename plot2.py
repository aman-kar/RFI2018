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
Date:   10 July 2018

Execution Format : python plot2.py 'path to raw GUPPI file'
Make sure GbtRaw.py is in the same directory as where you execute this script


Script Goal :-

Identical to plot.py. This script is meant to run through all channels
without asking for any user input

Scans through raw data file and produce power 
spectrum, spectrogram and time series plots
for all channels of X and Y polarisations

Credit: Majority of the script was initially created by Richard Prestage and
has hence since been modified to satisfy different script goals

Variables in Use:-

in_file       : Path to the raw GUPPI file
fitsRoot      : Raw Guppi File extension (e.g. '0006.0000')
blocks        : Number of blocks being extracted from raw file
g             : class object of GbtRaw package
n             : Number of blocks that exist in the raw file
obsfreq       : Centre Frequency of observations
obsbw         : Observed Bandwidth
obsnchan      : Number of Coarse Frequency Channels
tsData        : Internal Data Array Buffer( Number of Coarse Channels x Data Samples x Polarisations)
                Example - When extracting a single block from 0006.0000 raw file, you get 32 coarse 
                channels with 8386560 bytes of data in four types of polarisations ( Real & Imag of X & Y)

npol          : Number of Polarisations (Hardcoded to 2 for X & Y)
nfreq         : Number of Fine Frequency Channels (arbitrarily chosen but ideally should opt for powers of 2)
nint          : Number of individual intergrations to put into a single power spectrum
nspec         : Number of Power spectra to put into the spectrogram
                Ideally, nfreq x nint x nspec should equal the number of data samples present in the total
                number of blocks read
blk           : Block Counter

chanData      : Extract data from chosen coarse channel of either Polarisation (X or Y)
in_arr        : Temporary complex buffer that stores real and imaginary values
out_arr       : FFT Output from in_arr complex buffer
accum         : Converts complex voltage values to Power and accumulates
spec_list     : Power Spectra
dyn_spec      : Dynamic Spectrum or Spectrogram with nspec power spectra, each with nfreq frequency channels
                This data array can be thought of power values with time along the x-axis and frequency along Y-axis
channel       : Coarse Channel to be examined
chan          : Python relative of channel

pol           : 0 for X Polarisation and 1 for Y Polarisation
pol_type      : Stores the type of Polarisation chosen for the purposes of formatting filenames
sp            : Start of Polarization
ep            : End Polarization
                sp,ep should basically extract the real and imag components of the chose polarisation (X or Y)

fStart        : Starting Frequency for the chosen coarse channel
fRes          : Frequency Resolution for the output spectrogram
tsamp         : Sampling Rate of the raw data file
tRes          : Time Reolution for the output spectrogram

fitsName      : FitsName extension for filename
freq          : Frequency Domain array for the chosen coarse channel
timed         : Time Domain array for the chosen coarse channel
avg_time_pow  : Power values obtained after averaging along the time domain (To plot time series)
avg_freq_pow  : Power values obtained after averaging along the frequency domain (To power spectrum)

"""


from astropy.io import fits
import numpy as np
import pylab as plt
from matplotlib import rcParams
import os,sys
from GbtRaw import *


def spectroFITS(array, tStart, tRes, fStart, fRes, file_name):
    """Writes out array as an image in a FITS file"""

    # create the dynamic spectrum as the primary image
    hdu = fits.PrimaryHDU(array)

    # add the axes information
    hdu.header['CRPIX1'] = 0.0
    hdu.header['CRVAL1'] = tStart
    hdu.header['CDELT1'] = tRes
    hdu.header['CRPIX2'] = 0.0
    hdu.header['CRVAL2'] = fStart
    hdu.header['CDELT2'] = fRes

    # create the bandpass and timeseries
    bandpass    = np.average(array, axis=1)
    timeseries  = np.average(array, axis=0)

    # and create new image extensions with these
    bphdu = fits.ImageHDU(bandpass,name='BANDPASS')
    tshdu = fits.ImageHDU(timeseries,name='TIMESERIES')
    # uodate these headers.
    bphdu.header['CRPIX1'] = 0.0
    bphdu.header['CRVAL1'] = fStart
    bphdu.header['CDELT1'] = fRes
    tshdu.header['CRPIX1'] = 0.0
    tshdu.header['CRVAL1'] = tStart
    tshdu.header['CDELT1'] = tRes


    hdulist = fits.HDUList([hdu, bphdu, tshdu])
    hdulist.writeto(file_name)


def main():
    
    rcParams.update({'figure.autolayout' : True})
    rcParams.update({'axes.formatter.useoffset' : False})
    
    in_file = sys.argv[1] #Path to the raw GUPPI file
    fitsRoot = in_file[-13:-4] #Raw Guppi File extension (e.g. '0006.0000')
    
    blocks = raw_input("How many blocks to read? : ")  #Number of blocks being extracted from raw file
    g = GbtRaw(in_file) #class object of GbtRaw package
    n = g.get_num_blocks() #Number of blocks that exist in the raw file
    print "the file has", n, "blocks"
      
    npol=2 #Number of Polarisations (Hardcoded to 2 for X & Y)
    
    nfreq=int(raw_input("Number of fine frequency channels ? :")) 
    #Number of Fine Frequency Channels (arbitrarily chosen but ideally should opt for powers of 2)
    
    nint=int(raw_input("Value for nint (159) : "))
    #Number of individual intergrations to put into a single power spectrum
    
    nspec=int(51.5*(1024/nfreq)*int(blocks))
    #Number of Power spectra to put into the spectrogram
    #Ideally, nfreq x nint x nspec should equal the number of data samples present in the total
    #number of blocks read
   

    obsfreq = g.header_dict['OBSFREQ']
    obsbw   = g.header_dict['OBSBW']
    obsnchan= g.header_dict['OBSNCHAN']
    
    blk=0 #Block Counter 
    
    for block in range(0,n,int(blocks)):

        blk+=1
        print "Reading blocks", block+1,"-",block+int(blocks)
        tsData = g.extract(block,int(blocks),overlap=False)
        
        channel=raw_input("Channel to Inspect? (999 to exit) : ")
        
        while(channel!='999'):    
            
            for chan in [int(channel)-1]:#range(int(channel)):
                
                print "Processing channel: ", channel
                
                #Extract header information
                fStart  = obsfreq - float(obsbw)/2.0 + (chan *  float(obsbw) / obsnchan)
                #Starting Frequency for the chosen coarse channel
                
                fRes = float(obsbw) / (nfreq * obsnchan)
                #Frequency Resolution for the output spectrogram
                
                tsamp   = abs(float(obsnchan)/obsbw) * 1.0e-06  # MHz to Hz
                #Sampling Rate of the raw data file
                
                tStart  = 0
                tRes    = tsamp * nfreq * nint
                #Time Reolution for the output spectrogram
                
                #Loop through both (X & Y) Polarisation if npol is 2 or execute only for X polarisation if npol is 1
                for pol in range(npol):
                
                    sp = 2*pol #Start of Polarization
                    ep = sp+2 #End Polarization
                    #sp,ep should basically extract the real and imag components of the chose polarisation (X or Y)
                    
                    chanData = tsData[chan, :, sp:ep] #Extract data from chosen coarse channel of either Polarisation (X or Y)
                    
                    if pol==0:
                        pol_type='X'
                    elif pol==1:
                        pol_type='Y'
                    #Stores the type of Polarisation chosen for the purposes of formatting filenames
                    
                    spec_list = [] # Power Spectra
                                    
                    # loop over the power spectrum we are creating
                    for s in range(nspec):
            
                        # winstart is the start location in chanData of this spectrum
                        winStart = s * (nfreq * nint)
                        #print "WinStart: ",winStart
                        #time.sleep(2)
                        # create an empty array to accumulate FFTs
                        accum = np.zeros(nfreq) 
                        
                        # loop over the integrations we are going to accumulate into
                        # this spectrum - each will require nfreq time samples
                        for i in range(nint):
                            # start and end of this FFT in chanData
                            start = winStart + i * nfreq
                            end = start + nfreq
                            #print "Start: ",start,"; End :",end
                            #time.sleep(2)
                            # put the real and imag values into a temporart complex buffer
                            in_arr = np.zeros((nfreq), dtype=np.complex_)
                            in_arr.real = chanData[start:end, 0]
                            in_arr.imag = chanData[start:end, 1]
                            
                            # do the FFT, and put the output into the "normal" order
                            out_arr = np.fft.fftshift(np.fft.fft(in_arr))
                            
                            # convert from complex voltages to power, and accumulate
                            accum += np.abs(out_arr)**2
                            
                        spec_list.append(accum/nint)
                    
                    dyn_spec = np.transpose(np.asarray(spec_list))
                    #Dynamic Spectrum or Spectrogram with nspec power spectra, each with nfreq frequency channels
                    #This data array can be thought of power values with time along the x-axis and frequency along Y-axis
                    
                    if not os.path.exists('plots'): os.mkdir('plots')
                    if not os.path.exists('plots/ch'+channel): os.mkdir('plots/ch'+channel)
                    
                    fitsName = 'plots/ch' + channel + '/' + fitsRoot +'.c'+ str(chan+1) + '.'+pol_type+'.'+str(blk)+'.fits'
                   
                    if os.path.isfile(fitsName): os.remove(fitsName)
                    #spectroFITS(dyn_spec, tStart, tRes, fStart, fRes, fitsName)                
                    
                    freq=[] #Frequency Domain array for the chosen coarse channel
                    for a in range(nfreq):
                        freq.append(fStart+((fRes)*a))
                        
                    timed=[] #Time Domain array for the chosen coarse channel
                    for b in range(len(spec_list)):
                        timed.append(tStart+(tRes*b))
                    
                    avg_time_pow=[] #Power values obtained after averaging along the time domain (To plot time series)
                    for c in range(len(dyn_spec[:,0])):
                        avg_time_pow.append(np.average(dyn_spec[c,:]))
                    
                    avg_freq_pow=[] #Power values obtained after averaging along the frequency domain (To power spectrum)
                    for f in range(len(dyn_spec[0,:])):
                        avg_freq_pow.append(np.average(dyn_spec[:,f]))        
                    
                    
                    
                    plt.subplots(3,1,figsize=(9,9))
                    plt.subplot(311)
                    plt.plot(freq,np.log(avg_time_pow),'-')
                    plt.xlim(freq[len(freq)-1],freq[0])
                    plt.xlabel('Frequency (in MHz)')
                    plt.subplot(312)
                    plt.imshow(dyn_spec,cmap='hot',aspect='auto')
                    plt.subplot(313)
                    plt.plot(timed,avg_freq_pow,'-')
                    plt.xlim(timed[0],timed[len(timed)-1])
                    plt.xlabel('Time (in s)')
                    plt.savefig('plots/ch'+channel+'/ch'+channel+'_spec_'+pol_type+'_'+str(blk)+'.png')
                    plt.show()
                  
            channel=raw_input("Channel to Inspect? (999 to exit) : ")

                
main()
            
            
            
         
           
     