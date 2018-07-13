#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 12:43:44 2018

@author: amankar
"""
from astropy.io import fits
import numpy as np
import pylab as plt
from matplotlib import rcParams
import os
from GbtRaw import *
import sys


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
    
    npol=2
    nfreq=1024
    nint=159
    
    
    path    = '/Users/amankar/Downloads/realTimeRfi-master/'
    in_file = 'guppi_56465_J1713+0747_0006.0000.raw'
    fitsRoot = '0006.0000'
    
    blocks = raw_input("How many blocks to read? ")
    nspec=int(51.5*int(blocks))
    g = GbtRaw(path+in_file)
    n = g.get_num_blocks()
    print "the file has", n, "blocks"

    obsfreq = g.header_dict['OBSFREQ']
    obsbw   = g.header_dict['OBSBW']
    obsnchan= g.header_dict['OBSNCHAN']

    tsData = g.extract(0,int(blocks), overlap=False)
    
    channel=raw_input("Channel to Inspect? : ")
    
    for chan in [int(channel)-1]:
        
        print "Processing channel: ", channel

        fStart  = obsfreq - float(obsbw)/2.0 + (chan *  float(obsbw) / obsnchan)
        fRes = float(obsbw) / (nfreq * obsnchan)
        tsamp   = abs(float(obsnchan)/obsbw) * 1.0e-06  # MHz to Hz
        tStart  = 0
        tRes    = tsamp * nfreq * nint

        for pol in range(npol):
        
            sp = 2*pol
            ep = sp+2
            
            chanData = tsData[chan, :, sp:ep]
            
            if pol==0:
                pol_type='X'
            elif pol==1:
                pol_type='Y'
            
            spec_list = []
        
            # do the work
            kurt_real=[]
            kurt_imag=[]
            # loop over the power spectrum we are creating
            for s in range(nspec):
    
                # winstart is the start location in chanData of this spectrum
                winStart = s * (nfreq * nint)
                #print "WinStart: ",winStart
                #time.sleep(2)
                # create an empty array to accumulate FFTs
                accum = np.zeros(nfreq)
                kreal=np.zeros(nfreq)
                kimag=np.zeros(nfreq)
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
                    kreal+=out_arr.real
                    kimag+=out_arr.imag
                    # convert from complex voltages to power, and accumulate
                    accum += np.abs(out_arr)**2
                    
                kurt_real.append(kreal/nint)
                kurt_imag.append(kimag/nint)
                
                spec_list.append(accum/nint)
    
            kurt_real = np.transpose(np.asarray(kurt_real))
            kurt_imag = np.transpose(np.asarray(kurt_imag))
            
            dyn_spec = np.transpose(np.asarray(spec_list))
             
            if not os.path.exists('plots'): os.mkdir('plots')
            if not os.path.exists('plots/ch'+channel): os.mkdir('plots/ch'+channel)
            
            fitsName = path + 'plots/ch' + channel + '/' + fitsRoot +'.c'+ str(chan+1) + '.'+pol_type+'.fits'
            spectroFITS(dyn_spec, tStart, tRes, fStart, fRes, fitsName)                
            
            freq=[]
            for a in range(nfreq):
                freq.append(fStart+((fRes)*a))
                
            timed=[]
            for b in range(len(spec_list)):
                timed.append(tStart+(tRes*b))
            
            avg_time_pow=[]
            for c in range(len(dyn_spec[:,0])):
                avg_time_pow.append(np.average(dyn_spec[c,:]))
            
            avg_freq_pow=[]
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
            plt.savefig(path+'plots/ch'+channel+'/ch'+channel+'_spec_'+pol_type+'_'+blocks+'.png')
            plt.show()
            plt.close()
            
                
                
            while(raw_input('Is there a power Spike to Inspect? (y/n) : ')!='n'):
                
                print "Enter the frequency region where the spike is located"
                spike_start_idx=np.abs(np.asarray(freq)-float(raw_input("Start Frequency (in MHz) : "))).argmin()
                spike_end_idx=np.abs(np.asarray(freq)-float(raw_input("End Frequency (in MHz) : "))).argmin()
                
                #the 'freq' array is flipped, hence spike_end comes before spike_start
                rfi=spike_end_idx+np.argmax(avg_time_pow[spike_end_idx:spike_start_idx])    
                
                random_idx = np.abs(np.asarray(freq)-float(raw_input("Enter a random channel frequency for comparison (in MHz) : "))).argmin()
                
                #Time Series Plot at Power Spike (Tone) Found Above
                
                tone_real = []
                clear_real=[]
                for d in range(len(kurt_real[0,:])):
                    tone_real.append(kurt_real[rfi,d])
                    clear_real.append(kurt_real[random_idx,d])
                
                tone_imag = []
                clear_imag = []
                for e in range(len(kurt_imag[0,:])):
                    tone_imag.append(kurt_imag[rfi,e])
                    clear_imag.append(kurt_imag[random_idx,e])
             
                plt.subplots(2,2,figsize=(9,9))
                plt.subplot(221)
                plt.plot(timed,clear_real,'b*',label='Comparison')
                plt.plot(timed,tone_real,'g*',label='Tone')
                plt.xlabel('Time (in s)')
                plt.legend()
                plt.ylim(-127,127)
                plt.title(pol_type+' Real Polarisation')
                plt.subplot(222)
                plt.hist(tone_real,bins=50,alpha=0.5,normed=True,color='b',label='Comparison')
                plt.hist(clear_real,bins=50,alpha=0.5,normed=True,color='g',label='Tone')
                plt.legend()
                plt.title(pol_type+' Real Polarisation')
                plt.subplot(223)
                plt.plot(timed,clear_imag,'b*',label='Comparison')
                plt.plot(timed,tone_imag,'g*',label='Tone')
                plt.xlabel('Time (in s)')
                plt.legend()
                plt.ylim(-127,127)
                plt.title(pol_type+' Imaginary Polarisation')
                plt.subplot(224)
                plt.hist(tone_imag,bins=50,alpha=0.5,normed=True,color='b',label='Comparison')
                plt.hist(clear_imag,bins=50,alpha=0.5,normed=True,color='g',label='Tone')
                plt.legend()
                plt.title(pol_type+' Imaginary Polarisation')
                plt.savefig(path+'plots/ch'+channel+'/ch'+channel+'_'+pol_type+'_freq_'+str(round((freq[rfi]),3))+'_'+blocks+'.png')
                plt.show()
                plt.close()
            
main()
            
            
            
         
           
     