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

Execution Format : python plot.py 'path to raw GUPPI file'
Make sure GbtRaw.py is in the same directory as where you execute this script

Script Goal :-

Scan through raw data file and produce power 
spectrum, spectrogram and time series plots
for desired channels of X and Y polarisations

Also, examine potential RFI tones that may be
found in the power spectrum and plot raw voltage
values as time series for the specific
fine frequency at which the RFI was detected 
alongwith another complex voltage plot at a 
different fine frequency channel for comparison
purposes and finally plot a histogram for both.

Finally, produces a Numpy array file (npy) to save
data array of the given channel. This file can be
used as input for the SK_data.py script to run
the SK Estimator

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


vreal         : Stores real voltages values of chosen polarisation and accumulates
vimag         : Stores imaginary voltage values of chosen polarisation and accumulates
volt_real     : Stores averaged real Voltage values after accumulating
volt_imag     : Stores averaged imaginary Voltage values after accumulating

fitsName      : FitsName extension for filename
freq          : Frequency Domain array for the chosen coarse channel
timed         : Time Domain array for the chosen coarse channel
avg_time_pow  : Power values obtained after averaging along the time domain (To plot time series)
avg_freq_pow  : Power values obtained after averaging along the frequency domain (To power spectrum)

spike_start_idx : Index value of the starting position of the frequency region where the RFI is located
spike_end_idx : Index value of the ending position of the frequency region where the RFI is located
rfi           : Index value of where the RFI tone is located
random_idx    : Index value of where a random fine frequency channel was chosen for comparison
tone_real     : Stores the real voltage values at the RFI Tone fine frequency
tone_imag     : Stores the imaginary voltage values at the RFI Tone fine frequency
clear_real    : Stores the real voltage values at the random fine frequency
clear_imag    : Stores the imaginary voltage values at the random fine frequency

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
    
    print "The file being read contains", n, "blocks"
    print "Reading", blocks,"blocks from file"

    obsfreq = g.header_dict['OBSFREQ'] #Centre Frequency of observations
    obsbw   = g.header_dict['OBSBW'] #Observed Bandwidth
    obsnchan= g.header_dict['OBSNCHAN'] #Number of Coarse Frequency Channels

    tsData = g.extract(0,int(blocks), overlap=False)
    #Internal Data Array Buffer( Number of Coarse Channels x Data Samples x Polarisations)
    #Example - When extracting a single block from 0006.0000 raw file, you get 32 coarse 
    #channels with 8386560 bytes of data in four types of polarisations ( Real & Imag of X & Y)
    
    npol=2 #Number of Polarisations (Hardcoded to 2 for X & Y)
    
    nfreq=int(raw_input("Number of fine frequency channels ? : ")) 
    #Number of Fine Frequency Channels (arbitrarily chosen but ideally should opt for powers of 2)
    
    nint=int(raw_input("Value for nint (159) : "))
    #Number of individual intergrations to put into a single power spectrum
    
    nspec=int(51.5*(1024/nfreq)*int(blocks))
    #Number of Power spectra to put into the spectrogram
    #Ideally, nfreq x nint x nspec should equal the number of data samples present in the total
    #number of blocks read
    
    channel=raw_input("Which Channel to Inspect? : ")  #Coarse Channel to be examined
    
    #The loop below only executes once but can be changed to loop through all channels
    for chan in [int(channel)-1]:  #Python relative of channel
        
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
        for pol in range(npol): #0 for X Polarisation and 1 for Y Polarisation
        
            sp = 2*pol #Start of Polarization
            ep = sp+2 #End Polarization
            #sp,ep should basically extract the real and imag components of the chose polarisation (X or Y)
            
            chanData = tsData[chan, :, sp:ep] #Extract data from chosen coarse channel of either Polarisation (X or Y)
            
            if pol==0:
                pol_type='X'
            elif pol==1:
                pol_type='Y'
            #Stores the type of Polarisation chosen for the purposes of formatting filenames
            
            print "Processing",pol_type, "Polarisation"
            
            spec_list = [] # Power Spectra
            volt_real=[] #Stores averaged real Voltage values after accumulating
            volt_imag=[] #Stores averaged imaginary Voltage values after accumulating

            # loop over the power spectrum we are creating
            for s in range(nspec):
    
                # winstart is the start location in chanData of this spectrum
                winStart = s * (nfreq * nint)
              
                # create an empty array to accumulate FFTs
                accum = np.zeros(nfreq)
                vreal=np.zeros(nfreq)
                vimag=np.zeros(nfreq)
                # loop over the integrations we are going to accumulate into
                # this spectrum - each will require nfreq time samples
                for i in range(nint):
                    # start and end of this FFT in chanData
                    start = winStart + i * nfreq
                    end = start + nfreq                   
                    # put the real and imag values into a temporart complex buffer
                    in_arr = np.zeros((nfreq), dtype=np.complex_)
                    in_arr.real = chanData[start:end, 0]
                    in_arr.imag = chanData[start:end, 1]
                    
                    # do the FFT, and put the output into the "normal" order
                    out_arr = np.fft.fftshift(np.fft.fft(in_arr))
                    #Keep the real and imag voltage values and accumulate
                    vreal+=out_arr.real 
                    vimag+=out_arr.imag
                    # convert from complex voltages to power, and accumulate
                    accum += np.abs(out_arr)**2
                    
                volt_real.append(vreal/nint)
                volt_imag.append(vimag/nint)            
                spec_list.append(accum/nint)
    
            volt_real = np.transpose(np.asarray(volt_real))
            volt_imag = np.transpose(np.asarray(volt_imag))  
            
            dyn_spec = np.transpose(np.asarray(spec_list)) 
            #Dynamic Spectrum or Spectrogram with nspec power spectra, each with nfreq frequency channels
            #This data array can be thought of power values with time along the x-axis and frequency along Y-axis
             
            if not os.path.exists('plots'): os.mkdir('plots')
            if not os.path.exists('plots/ch'+channel): os.mkdir('plots/ch'+channel)
            
            fitsName = 'plots/ch' + channel + '/' + fitsRoot +'.c'+ str(chan+1) + '.'+pol_type+'.fits'
            #FitsName extension for filename
            
            if os.path.isfile(fitsName): os.remove(fitsName)
            spectroFITS(dyn_spec, tStart, tRes, fStart, fRes, fitsName)                
            
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
            plt.savefig('plots/ch'+channel+'/ch'+channel+'_spec_'+pol_type+'_'+blocks+'.png')
            plt.show()         

            # While loop to keep asking for a power spike inspection. 'n' will skip the script to the next polarisation or finish the 'for' loop
            while(raw_input('Is there a Power Spike to Inspect? (y/n) : ')!='n'):
                
                print "Enter the frequency region where the spike is located"
                spike_start_idx=np.abs(np.asarray(freq)-float(raw_input("Start Frequency (in MHz) : "))).argmin()
                #Index value of the starting position of the frequency region where the RFI is located
                
                spike_end_idx=np.abs(np.asarray(freq)-float(raw_input("End Frequency (in MHz) : "))).argmin()
                #Index value of the ending position of the frequency region where the RFI is located
                
               
                rfi=spike_end_idx+np.argmax(avg_time_pow[spike_end_idx:spike_start_idx])    
                #Index value of where the RFI tone is located
                #the 'freq' array is flipped, hence 'spike_end' comes before 'spike_start'
                
                random_idx = np.abs(np.asarray(freq)-float(raw_input("Enter a random channel frequency for comparison (in MHz) : "))).argmin()
                #Index value of where a random fine frequency channel was chosen for comparison
                
                
                #Time Series Plot at Power Spike (Tone) Found Above
                
                tone_real = [] #Stores the real voltage values at the RFI Tone fine frequency
                clear_real=[] #Stores the real voltage values at the random fine frequency
                for d in range(len(volt_real[0,:])):
                    tone_real.append(volt_real[rfi,d])
                    clear_real.append(volt_real[random_idx,d])
                
                tone_imag = [] #Stores the imaginary voltage values at the RFI Tone fine frequency
                clear_imag = [] #Stores the imaginary voltage values at the random fine frequency
                for e in range(len(volt_imag[0,:])):
                    tone_imag.append(volt_imag[rfi,e])
                    clear_imag.append(volt_imag[random_idx,e])
             
                plt.subplots(2,2,figsize=(9,9))
                
                plt.subplot(221)
                plt.plot(timed,clear_real,'b*',label='Comparison ('+str(round(freq[random_idx],2))+'MHz)')
                plt.plot(timed,tone_real,'g*',label='Tone ('+str(round(freq[rfi],2))+'MHz)')
                plt.xlabel('Time (in s)')
                plt.legend()
                plt.ylim(-127,127)
                plt.title(pol_type+' Real Polarisation')
                
                plt.subplot(222)
                plt.hist(tone_real,bins=50,alpha=0.5,normed=True,color='b',label='Comparison ('+str(round(freq[random_idx],2))+'MHz)')
                plt.hist(clear_real,bins=50,alpha=0.5,normed=True,color='g',label='Tone ('+str(round(freq[rfi],2))+'MHz)')
                plt.xlim(-127,127)
                plt.legend()
                plt.title(pol_type+' Real Polarisation')
                
                plt.subplot(223)
                plt.plot(timed,clear_imag,'b*',label='Comparison ('+str(round(freq[random_idx],2))+'MHz)')
                plt.plot(timed,tone_imag,'g*',label='Tone ('+str(round(freq[rfi],2))+'MHz)')
                plt.xlabel('Time (in s)')
                plt.legend()
                plt.ylim(-127,127)
                plt.title(pol_type+' Imaginary Polarisation')
                
                plt.subplot(224)
                plt.hist(tone_imag,bins=50,alpha=0.5,normed=True,color='b',label='Comparison ('+str(round(freq[random_idx],2))+'MHz)')
                plt.hist(clear_imag,bins=50,alpha=0.5,normed=True,color='g',label='Tone ('+str(round(freq[rfi],2))+'MHz)')
                plt.xlim(-127,127)
                plt.legend()
                plt.title(pol_type+' Imaginary Polarisation')
                
                plt.savefig('plots/ch'+channel+'/ch'+channel+'_'+pol_type+'_freq_'+str(round((freq[rfi]),3))+'_'+blocks+'.png')                    
                plt.show()
                
        #Create .npy input files to run the SK Estimator on it (SK_data.py)
        if(raw_input("Do you want to save this channel data array? (for SK_data.py) (y/n)")=='y'):
            np.save('chan'+channel+'_'+blocks+'.npy',tsData[chan,:,:])
            
            
main()
            
            
            
         
           
     