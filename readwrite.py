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

Execution Format : python readwrite.py 'path to raw GUPPI file'
Make sure GbtRaw.py is in the same directory as where you execute this script


Script Goal :-


Attempting to read in a Raw GUPPI File, FFT the Data,
IFFT The Data, and write the file back into the original raw file

Credit: Majority of the script was initially created by Richard Prestage and
has hence since been modified to satisfy different script goals

Variables in Use:-

in_file              : Path to the raw GUPPI file
blocks               : Number of blocks being extracted from raw file
g                    : class object of GbtRaw package


header               : Obtain the header information for the current block
input_tsData         : Internal Data Array Buffer( Number of Coarse Channels x Data Samples x Polarisations)
                       Example - When extracting a single block from 0006.0000 raw file, you get 32 coarse 
                       channels with 8386560 bytes of data in four types of polarisations ( Real & Imag of X & Y)
output_tsData        : Output Data Array Buffer that is created with the exact same shape and size as input_tsData
                       and holds the output after the FFT/IFFT processing
                       
chan                 : Python relative of channel
obsnchan             : Number of Coarse Frequency Channels
pol                  : 0 for X Polarisation and 1 for Y Polarisation
sp                   : Start of Polarization
ep                   : End Polarization
                       sp,ep should basically extract the real and imag components of the chose polarisation (X or Y)

input_chanData       : Extract data from chosen coarse channel of either Polarisation (X or Y)
output_chanData      : Created with exact shape and size of input_chanData to hold the output of FFT/IFFT processing 
                       on input_chanData
                       
nfreq                : Number of Fine Frequency Channels (arbitrarily chosen but ideally should opt for powers of 2)
nint                 : Number of individual intergrations to put into a single power spectrum
nspec                : Number of Power spectra to put into the spectrogram
                       Ideally, nfreq x nint x nspec should equal the number of data samples present in the total
                       number of blocks read
winStart             : Start location in input_chanData of this spectrum
input_array          : Temporary buffer of complex voltage values
output_array         : Array that holds the output of the FFT/IFFT processing on input_array

"""

import numpy as np
from GbtRaw import *
from numba import jit
import sys

#Reads a temporary buffer of complex voltages, perform FFT/IFFT the array, returns the output
@jit
def modify(input_chanData,start,end):
    input_array=np.zeros((end-start),dtype=np.complex_)
    input_array.real=input_chanData[start:end,0]
    input_array.imag=input_chanData[start:end,1]
    output_array=np.rint(np.fft.ifft(np.fft.fft(input_array,norm='ortho'),norm='ortho'))
    return output_array.real, output_array.imag

#Function to read in the raw data file, perform modify() on the each block and write it
#back into the orginal raw data file.
@jit
def readwrite(g,blocks,obsnchan,npol,nspec,nfreq,nint):
    
    for block in range(blocks):        
        
        header,input_tsData=g.get_block(block) 
        #Obtain the header information and the data content for the current block
        output_tsData=np.empty_like(input_tsData)
        #Output Data Array Buffer that is created with the exact same shape and size as input_tsData
        #and holds the output after the FFT/IFFT processing
        
        print "Prcoessing Block: ", (block+1)
        
        for chan in range(obsnchan): #Python relative of channel
            
            for pol in range(npol): #0 for X Polarisation and 1 for Y Polarisation
                
                sp=2*pol #Start of Polarization
                ep=sp+2 #End Polarization
                
                input_chanData = input_tsData[chan, :, sp:ep] 
                #Extract data from chosen coarse channel of either Polarisation (X or Y)
                
                output_chanData=np.empty_like(input_chanData)
                #Created with exact shape and size of input_chanData to hold the output of FFT/IFFT processing 
                #on input_chanData
  
                for s in range(nspec):
        
                    winStart = s * (nfreq * nint) #Start location in input_chanData of this spectrum
                    for i in range(nint):
        
                        start = winStart + i * nfreq
                        end = start + nfreq
                        #Start and end points for the given chanData
                        
                        if(start>input_tsData.shape[1]):
                            break

                        if((end>input_tsData.shape[1])): #Condition to ensure all leftover data samples are processed

                            end=input_tsData.shape[1]
                            output_chanData[start:end,0],output_chanData[start:end,1]=modify(input_chanData,start,end)
                            break
                        
                        output_chanData[start:end,0],output_chanData[start:end,1]=modify(input_chanData,start,end)

                output_tsData[chan,:,sp:ep]=output_chanData
                
        g.put_block(header,output_tsData,block) #Write processed data block into the original file itself
        print "Writing block" ,(block+1)," data back in to the raw file"

def main():    
    
    in_file = sys.argv[1]
    #Path to the raw GUPPI file
    
    g = GbtRaw(in_file,update=True) #class object of GbtRaw package
    
    blocks = g.get_num_blocks() #Number of blocks being extracted from raw file
    print "the file has", blocks, "blocks"
    
    npol=2 #Number of Polarisations (Hardcoded to 2 for X & Y)
    
    nfreq=int(raw_input("Number of fine frequency channels ? :")) 
    #Number of Fine Frequency Channels (arbitrarily chosen but ideally should opt for powers of 2)
    
    nint=int(raw_input("Value for nint (159) : "))
    #Number of individual intergrations to put into a single power spectrum
    
    nspec=int(51.5*(1024/nfreq)*int(blocks))
    #Number of Power spectra to put into the spectrogram
    #Ideally, nfreq x nint x nspec should equal the number of data samples present in the total
    #number of blocks read
    
    obsnchan = g.header_dict['OBSNCHAN'] #Number of Coarse Frequency Channels
    
    readwrite(g,blocks,obsnchan,npol,nspec,nfreq,nint)  
    
        
                
                
