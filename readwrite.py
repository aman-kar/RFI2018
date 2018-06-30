#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:25:19 2018

@author: amankar


Attempting to Reading in Raw GUPPI File, FFT the Data
IFFT The Data, and write the file back in the original raw format
"""



from astropy.io import fits
import numpy as np
import pylab as plt
from matplotlib import rcParams
import time
from GbtRaw import *
import scipy.stats as stats




def main():
    blocks=5
    nfreq=1024
    nspec=618
    nint=159
    pol=0
    path    = '/Users/amankar/Downloads/realTimeRfi-master/'
    in_file = 'guppi_56465_J1713+0747_0006.0000.raw'
    fitsRoot = '0006.0000'
    g = GbtRaw(path+in_file)
    n = g.get_num_blocks()
    print "the file has", n, "blocks"
    
    obsfreq = g.header_dict['OBSFREQ']
    obsbw   = g.header_dict['OBSBW']
    obsnchan= g.header_dict['OBSNCHAN']
    
    input_tsData = g.extract(0,blocks, overlap=False)
    output_tsData=np.empty_like(input_tsData)
    
    sp = 2*pol
    ep = sp+2

    for chan in range(32):
        print "Processing channel: ", chan

        fStart  = obsfreq - float(obsbw)/2.0 + (chan *  float(obsbw) / obsnchan)
        fRes = float(obsbw) / (nfreq * obsnchan)

        tsamp   = abs(float(obsnchan)/obsbw) * 1.0e-06  # MHz to Hz
        tStart  = 0
        tRes    = tsamp * nfreq * nint

        input_chanData = input_tsData[chan, :, sp:ep]
        output_chanData=np.empty_like(input_chanData)
        print fStart,fRes,tsamp,tRes
        print input_chanData
        
        for s in range(nspec):

            winStart = s * (nfreq * nint)

            for i in range(nint):

                start = winStart + i * nfreq
                end = start + nfreq

                in_arr = np.zeros((nfreq), dtype=np.complex_)
                out_arr = np.zeros((nfreq), dtype=np.complex_)
                in_arr.real = input_chanData[start:end, 0]
                in_arr.imag = input_chanData[start:end, 1]
                
                out_arr = np.fft.ifft(np.fft.fft(in_arr))
                
                output_chanData[start:end,0]=out_arr.real
                output_chanData[start:end,1]=out_arr.imag
        
        print output_chanData
        output_tsData[chan,:,sp:ep]=output_chanData
        
                
                
