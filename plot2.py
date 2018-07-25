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
    
    blocks = raw_input("How many blocks to read at a time? ")
    nspec=int(51.5*int(blocks))
    g = GbtRaw(path+in_file)
    n = g.get_num_blocks()
    print "the file has", n, "blocks"

    obsfreq = g.header_dict['OBSFREQ']
    obsbw   = g.header_dict['OBSBW']
    obsnchan= g.header_dict['OBSNCHAN']
    
    blk=0
    
    for block in range(0,n,int(blocks)):

        blk+=1
        print "Reading blocks", block+1,"-",block+int(blocks)
        tsData = g.extract(block,int(blocks),overlap=False)
        
        channel=raw_input("Channel to Inspect? : ")
        
        while(channel!='999'):    
            
            for chan in [int(channel)-1]:#range(int(channel)):
                
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
                     
                    if not os.path.exists('plots'): os.mkdir('plots')
                    if not os.path.exists('plots/ch'+channel): os.mkdir('plots/ch'+channel)
                    
                    fitsName = path + 'plots/ch' + channel + '/' + fitsRoot +'.c'+ str(chan+1) + '.'+pol_type+'.'+str(blk)+'.fits'
                   
                    if os.path.isfile(fitsName): os.remove(fitsName)
                    #spectroFITS(dyn_spec, tStart, tRes, fStart, fRes, fitsName)                
                    
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
                    #plt.savefig(path+'plots/ch'+channel+'/ch'+channel+'_spec_'+pol_type+'_'+str(blk)+'.png')
                    plt.show()
                    plt.close()                        
                  
            channel=raw_input("Channel to Inspect? : ")

                
main()
            
            
            
         
           
     