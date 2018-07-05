#!/users/rprestag/venv/bin/python

from astropy.io import fits
import numpy as np
import pylab as plt
from matplotlib import rcParams
import time
from GbtRaw import *
import scipy.stats as stats

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

    # arbitrarily look at pol 5 (which I think is X)
    # just look at the first 5 blocks in the file (20 would be better)
    pol = 0
    blocks = 8

    # the number of blocks, frequency channels, power spectra, 
    # and integrations per power spectrum are overconstrained, so they
    # cannot all be chosen independently. Long term, a semi-intelligent
    # algorithm should choose them. For now, I set the following values 
    # by hand, on the basis of previous analyses (e.g. by Steve Ellingson
    # or intuition


    # create 200 time bins of 1024 spectra, each with 159 integrations.
    # there will be a few bytes not used.

    # number of frequency channels - fairly arbitrary, but powers of 2
    # are good for some programs / doing FFTs, etc, and ~ 1024 seems reasonable
    nfreq = 1024

    # number of power spectra to put into the spectrogram. Why not chose
    # something different from 256, just for the heck of it
    nspec = 412

    # number of individual "integrations" (output of FFT) to put into a single
    # power spectrum. At some point in the past, this number made 
    # nfreq * nspec * nint reasonably close to the number of data samples in
    # a block (or a file, can't remember).

    # this sets the time resolution in the spectogram, so we should get back
    # to this, and choose it more carefully, at some point.
    nint  = 159

    path    = '/Users/amankar/Downloads/realTimeRfi-master/'
    in_file = 'guppi_56465_J1713+0747_0006.0000.raw'
    fitsRoot = '0006.0000'
    g = GbtRaw(path+in_file)
    n = g.get_num_blocks()
    print "the file has", n, "blocks"

    # get frequency info, etc, from header
    obsfreq = g.header_dict['OBSFREQ']
    obsbw   = g.header_dict['OBSBW']
    obsnchan= g.header_dict['OBSNCHAN']

    # get the data. tsData is an internal (to this program) buffer, which
    # is just a numpy array of time series (2 pols, real and imag), without 
    # all of the header info, etc.
    tsData = g.extract(0,blocks, overlap=False)

    # required polarization channels
    # sp = start pol, ep = end pol, should extract real and imag for input
    # polarization                                       
    sp = 2*pol
    ep = sp+2

    # now loop over requested  channels - here set to just a single one
    for chan in [29]:
        print "Processing channel: ", chan

        # using the frequency information read from the header of the GUPPI
        # raw data file, calculate the starting frequency for the output
        # spectrogram. Then calculate the frequency resolution for the output
        # spectrogram, given the width of a single input GUPPI coarse channel
        # (obsbw / obsnchan), and the number of frequency channels we want
        # in the output spectrogram (nfreq)

        # if the bandwidth is -ve, the first channel is at high frequency
        fStart  = obsfreq - float(obsbw)/2.0 + (chan *  float(obsbw) / obsnchan)
        fRes = float(obsbw) / (nfreq * obsnchan)

        # calculate:
        #   tsamp  - the sampling rate of the raw data in the GUPPI file
        #   tstart - should be read from the GUPPI file, but not important
        #   tRes   - the time resolution in the output spectrogram.
        #            This is the sampling rate of the raw data, times the
        #            number of samples per FFT, times the number of FFTs
        #            per power spectrum.
        tsamp   = abs(float(obsnchan)/obsbw) * 1.0e-06  # MHz to Hz
        tStart  = 0
        tRes    = tsamp * nfreq * nint

        # extract the required channel
        # chanData is an internal (to this program) numpy array, which has
        # only the channel and polarization of interest
        chanData = tsData[chan, :, sp:ep]
        print fStart,fRes,tsamp,tRes
        print chanData
        # empty list of power spectra - these will become the spectrogram
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
                
            # append this power spectrum to the list of power spectra
            kurt_real.append(kreal/nint)
            kurt_imag.append(kimag/nint)
            
            spec_list.append(accum/nint)


        # convert the python list of 1-d power spectra into a numpy 2-d
        # array. Transpose it so that time is on the X axis, and frequency
        # on the Y-axis. This is Python / numpy voodoo

        # dyn_spec is then our dynamic spectum - aka spectrogram, with
        # nspec power spectra, each with nfreq frequency channels
        dyn_spec = np.transpose(np.asarray(spec_list))
        
        k_realval=[]
        k_imagval=[]
        pval_real=[]
        pval_imag=[]
        for a in range(len(kurt_real[0])):
            k_realval.append(stats.kurtosis(np.asarray(kurt_real)[:,a]))
            pval_real.append(stats.kurtosistest(np.asarray(kurt_real)[:,a])[1])
        for b in range(len(kurt_imag[0])):
            k_imagval.append(stats.kurtosis(np.asarray(kurt_imag)[:,b]))
            pval_imag.append(stats.kurtosistest(np.asarray(kurt_imag)[:,b])[1])


        freq=[]
        for c in range(nfreq):
            freq.append(fStart+(fRes*c))
            
        timed=[]
        for d in range(len(spec_list)):
            timed.append(tStart+(tRes*d))
            
        #Temporary Plot Command to Check Results  
        #Plot Time Domain
        avg_pow2=[]
        for f in range(len(dyn_spec[0,:])):
            avg_pow2.append(np.average(dyn_spec[:,f]))
        plt.plot(timed,avg_pow2)
        plt.xlabel('Time (in s)')   
        plt.savefig('time26.png')
        #PLot Frequency Domain
        avg_pow=[]
        for e in range(len(dyn_spec[:,0])):
            avg_pow.append(np.average(dyn_spec[e,:]))
        plt.plot(freq,avg_pow)
        plt.xlabel('Frequency (MHz)')
        plt.savefig('avg_freq_dom26.png')
        #plt.plot(timed,np.asarray(spec_list)[:,0])
        
        
        #plt.figure(figsize=(10,5)).add_subplot(221).plot(freq,k_realval)
        plt.plot(freq,k_realval)
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Excess Kurtosis')
        plt.title('X Real')
        plt.savefig('xreal.png')
        #plt.figure(figsize=(10,5)).add_subplot(223).plot(freq,k_imagval)
        plt.plot(freq,k_imagval)
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Excess Kurtosis')
        plt.title('X Imaginary')
        plt.savefig('ximag.png')
        

        plt.plot(freq,pval_real)
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('P Value')
        
        # write it to a fits file
        fitsName = path + fitsRoot + '.c' + str(chan+1) + '.fits'
        spectroFITS(dyn_spec, tStart, tRes, fStart, fRes, fitsName)    

if __name__ == "__main__":
    main()


