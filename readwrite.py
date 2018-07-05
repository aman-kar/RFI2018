#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Attempting to Reading in Raw GUPPI File, FFT the Data
IFFT The Data, and write the file back in the original raw format
"""



import numpy as np
import pylab as plt
import os
from GbtRaw import *
import scipy.stats as stats
from shutil import copyfile



def main():

    nfreq=1024
    nspec=53
    nint=159
    
    path    = '/Users/amankar/Downloads/realTimeRfi-master/'
    in_file = 'backup.raw'
    out_file = 'guppi_56465_J1713+0747_0006.0000.raw'

    print "Copying Orginial Raw File"
    copyfile(path+in_file,path+out_file)
    print "Successfully copied"
    g = GbtRaw(path+out_file,update=True)
    
    n = g.get_num_blocks()
    print "the file has", n, "blocks"
    
    obsfreq = g.header_dict['OBSFREQ']
    obsbw   = g.header_dict['OBSBW']
    obsnchan= g.header_dict['OBSNCHAN']
    
    for block in range(n):        
        
        header,input_tsData=g.get_block(block)
        output_tsData=np.empty_like(input_tsData)
        print "Prcoessing Block: ", (block+1)
    
        for chan in range(32):
            
            print "Processing Channel: ", (chan+1)
            fStart  = obsfreq - float(obsbw)/2.0 + (chan *  float(obsbw) / obsnchan)
            fRes = float(obsbw) / (nfreq * obsnchan)
    
            tsamp   = abs(float(obsnchan)/obsbw) * 1.0e-06  # MHz to Hz
            tStart  = 0
            tRes    = tsamp * nfreq * nint
            
            for pol in range(2):

                sp=2*pol
                ep=sp+2
        
                input_chanData = input_tsData[chan, :, sp:ep]
                output_chanData=np.empty_like(input_chanData)
                
                for s in range(nspec):
        
                    winStart = s * (nfreq * nint)
                    for i in range(nint):
        
                        start = winStart + i * nfreq
                        end = start + nfreq
                        in_arr = np.zeros((nfreq), dtype=np.complex_)
                        out_arr = np.zeros((nfreq), dtype=np.complex_)
                        
                        if(start>input_tsData.shape[1]):
                            break

                        if((end>input_tsData.shape[1])):

                            end=input_tsData.shape[1]
                            in_arr = np.zeros((end-start), dtype=np.complex_)
                            out_arr = np.zeros((end-start), dtype=np.complex_)
                            output_chanData[start:end,0],output_chanData[start:end,1]=g.modify(in_arr,out_arr,input_chanData,output_chanData,start,end)
                            break
                        
                        output_chanData[start:end,0],output_chanData[start:end,1]=g.modify(in_arr,out_arr,input_chanData,output_chanData,start,end)

                output_tsData[chan,:,sp:ep]=output_chanData
                
        g.put_block(header,output_tsData,block)
        print "Writing block" ,block," data back in to the copied raw file"
    
    
    
        
                
                
