#!/users/rprestag/venv/bin/python

# Copyright (C) 2017 Richard M. Prestage 

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

"""Utilities to manipulate GBT "raw" data files.

Utilities to manipulate GBT raw data files. The hope is that
one day this will handle all GBT raw formats, including GUPPI,
VEGAS pulsar, VEGAS spectral line, and BTL. Currently, it only
handles GUPPI.

Author: Richard M. Prestage
Data:   18 November 2017

"""

import numpy as np
import os
import os.path
import pprint
# from astropy.io import fits

class GbtRaw(object):
    """ Python class to handle GBT "raw" data files

    Args:
        in_file (str): name of the file to open

    """

    def __init__(self, in_file, update=False):

        self.card_length = 80 # number of characters in a card

        # memory map the file
        if update:
            self.in_obj    = np.memmap(in_file, dtype = 'int8', mode = 'r+')
            self.read_only = False
        else:
            self.in_obj    = np.memmap(in_file, dtype = 'int8', mode = 'r')
            self.read_only = True
        self.in_file   = in_file
        self.in_size   = os.path.getsize(in_file)

        # read the first header block, to initialize key variables
        self.read_header(0)

        # consistency check - the file nust contain an integer
        # number of blocks
	print self.in_size
	print self.hd_len
        if (self.in_size % self.hd_len) != 0:
            raise ValueError
        self.nblocks = self.in_size / self.hd_len


    def get_num_blocks(self):
        """Return the number of blocks in the file."""

        # assume that blocsize is correct, but re-check the file
        # size in case that has been updated
        self.in_size   = os.path.getsize(self.in_file)
        self.nblocks = self.in_size / self.hd_len
        return self.nblocks


    def get_header(self, loc = 0):
        """Get a single header from a header/data block

        Args: loc (str): Location to start reading from.

        """
        
        # initialize variables
        header_len = 0
        more = True
        start = loc
        curr_loc = loc

        # loop through the file until 'END' is seen
        while more:
            if curr_loc + self.card_length > self.in_size:
                raise EOFError
            card = ''.join(map(chr,
                           self.in_obj[curr_loc:curr_loc + self.card_length]))
            curr_loc += self.card_length
            header_len += self.card_length
            if card[:3] == 'END':
                more = False
                
        # return the header
        return self.in_obj[start : start + header_len]
        

    def parse_header(self, header):
        """parse a header, and return key meta-data"""

        header_len = len(header)
        header_dict = {}
        ncards = header_len / self.card_length
        for i in range(ncards):
            card = ''.join(map(chr,
                      header[i*self.card_length : (i+1)*self.card_length]))
            if card[:3] == 'END':
                break
            else:
                # extract information into dictionary
                key, value = card.split('=')
                key = key.strip()
                if "'" in value:
                    value = value.strip()
                elif "." in value:
                    value = float(value)
                else:
                    value = int(value)
                header_dict[key] = value

        # extract key meta-data
        obsnchan = header_dict['OBSNCHAN']
        npol = header_dict['NPOL']
        nbits = header_dict['NBITS']
        overlap = header_dict['OVERLAP']
        blocsize = header_dict['BLOCSIZE']
        nsamples = blocsize / (obsnchan * npol * (nbits / 8))

        # return metadata, and dictionary
        return obsnchan, npol, nbits, nsamples, overlap, blocsize, header_dict


    def read_header(self, loc = None):
        """Read a header block, and update meta-data"""

        header = self.get_header(loc)
        self.header_len = len(header)
        self.obsnchan, self.npol, self.nbits, self.nsamples, self.overlap, \
             self.blocsize, self.header_dict = self.parse_header(header)
        self.hd_len = self.header_len + self.blocsize
    
    def copy(self, out_file, start_block = 0, blocks = None):
        """Copys blocks from to out_file.

        If number of blocks is not specified, copy to end of file
        """

        if os.path.exists(out_file):
            print "File already exists"
            return

        if (blocks == None):
            blocks = self.nblocks - start_block

        # create the output file of the appropriate size
        out_size = blocks * self.hd_len
        out_obj = np.memmap(out_file, dtype='int8', mode = 'w+', 
                            shape = (out_size))

        # copy appropriate blocks
        start = start_block * self.hd_len
        end = start + blocks * self.hd_len
        out_obj[:] = self.in_obj[start:end]
        out_obj.flush()
        
    def modify(self, input_array, output_array,input_chanData,output_chanData,start,end):
        """ Reads a temporary buffer of complex voltages, FFT/IFFT the array, returns the output"""
        input_array.real=input_chanData[start:end,0]
        input_array.imag=input_chanData[start:end,1]
        output_array=np.rint(np.fft.ifft(np.fft.fft(input_array,norm='ortho'),norm='ortho'))
        return output_array.real, output_array.imag

    def get_block(self, block = 0):
        """Read a header / data block from file. """

        if (block < 0) or (block > self.nblocks):
            print "Attempt to get invalid block"
            return
        else:
            start      = block*self.hd_len
            end_header = start + self.header_len
            end_data   = end_header + self.blocsize
            header     = self.in_obj[start : end_header]
            data       = self.in_obj[end_header : end_data]
            data       = data.reshape((self.obsnchan, -1, self.npol))
            return header, data


    def put_block(self, header, data, block = 0):
        """ Write a header / data block to file."""

        if (block < 0) or (block > self.nblocks):
            print "Attempt to put invalid block"
            return

        if self.read_only:
            print "Attempt to write to read-only file."""
            return

        data = data.reshape((self.obsnchan * self.nsamples * self.npol))
        start = block * self.hd_len
        data_start = start + self.header_len
        data_end = data_start + self.blocsize
        self.in_obj[start:data_start] = header
        self.in_obj[data_start:data_end] = data
        self.in_obj.flush()


    def extract(self, start_block = 0, blocks = 1, overlap = False):
        """Extracts data from requested blocks

        The method is to read each block, reshape it to the
        3-d shape (obsnchan, nsamples, npol), and write the btyes
        to the correct place in the output buffer.

        if overlap == True, nsamples is the full block
        if overlap == False, don't copy the overlap region

"""
        # calculate the dimensions to use for the output array
        if overlap:
            nsamples_out = self.nsamples
            transfer_len = self.blocsize
        else:
            nsamples_out = self.nsamples - self.overlap
            transfer_len = self.obsnchan * nsamples_out * self.npol

        # create an empty array of the correct size and shape
        data = np.zeros(shape = blocks * transfer_len, dtype='int8')
        data = data.reshape(self.obsnchan, -1, self.npol)

        # loop through requested blocks, transferring data
        for curr_block in range(blocks):
            in_start = ((start_block + curr_block) * self.hd_len) + \
                                                           self.header_len
            out_start = curr_block * nsamples_out
            temp = self.in_obj[in_start:in_start + self.blocsize]
            temp = temp.reshape(self.obsnchan, -1, self.npol)
            data[:, out_start:out_start + nsamples_out,:] = \
                    temp[:,:nsamples_out,:]

        # all done    
        return data

    def pow_spec(self, start_block = 0, blocks = 10, pol = 0, chan=0, \
                nfreq = 2048, avg = 105, overlap = False, out_file = None):
        """Create a dynamic power spectrum, and optionally write it to file."""
        
        # required polarization channels
        sp = 2*pol
        ep = sp+2

        # empty list of power spectra
        spec_list = []

        # loop over all blocks, processing one at a time
        for b in range(start_block, start_block+blocks):
            print "Getting block: ", b
            data = self.extract(b, 1, overlap)[chan, :, sp:ep]

            # loop over as many windows as possible, forming power spectra
            start = 0
            for win in range(data.shape[0]/(nfreq*avg)):
                print "Processing window: ", win
                accum = np.zeros(nfreq)
                for i in range(avg):
                    in_arr = np.zeros((nfreq), dtype=np.complex_)
                    in_arr.real = data[start:start+nfreq, 0]
                    in_arr.imag = data[start:start+nfreq, 1]
                    out_arr = np.fft.fftshift(np.fft.fft(in_arr))
                    accum += np.abs(out_arr)**2
                start = start + avg
                spec_list.append(accum /avg)

        # convert back to numpy array and transpose to desired order
        dyn_spec = np.transpose(np.asarray(spec_list))

        # if requested, save as fits file
        if out_file:
                hdu = fits.PrimaryHDU(dyn_spec)
                hdu.writeto(out_file)


    def print_header(self):
        """pretty-prints the header"""
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.header_dict)


    def __del__(self):
        del self.in_file


def main1():
    """Demonstrate some aspects of GbtRaw. """

    # read an example, small GUPPI data file, and print some information
    g = GbtRaw('/Users/amankar/GBO_Emily/example.raw')
    g.print_header()
    print "file size in bytes:  ", g.in_size
    print "block size in bytes: ", g.blocsize
    print "number of blocks:    ", g.nblocks

    # copy the first block to a new file

    print "\n Copying file... \n"

    g.copy("/Users/amankar/GBO_Emily/temp.raw", 0, 1)
    del g

    # read the new file in, and show its sizes
    h = GbtRaw('/Users/amankar/GBO_Emily/temp.raw')
    print "file size in bytes:  ", h.in_size
    print "block size in bytes: ", h.blocsize
    print "number of blocks:    ", h.nblocks


def main2():
    import pylab as plt
    g = GbtRaw('/Users/amankar/GBO_Emily/example.raw')
    print "the file has", g.get_num_blocks(), "blocks"

    data = g.extract(0,3)
    print "the data array shape is:", data.shape

    # Cedric idenfied radar intereference starting 24.5M samples
    # from the start of the GUPPI raw data file.
    # correct for the fact that we skipped the first block
    # when creating the little example file.
    # (not sure why we need the factor of 2 with the overlap)
    start = 24500000 - (g.nsamples - 2 * g.overlap)
    end = start + 500000
    plt.plot(data[22,start:end,2])
    plt.xlabel('samples from arbitrary offset')
    plt.ylabel('Y-pol, real component')
    plt.title('extract from guppi_56465_J1713+0747_0006.0000.raw')
    plt.show()


def main3():
    f1 = GbtRaw('/Users/amankar/GBO_Emily/example.raw')
    f1.copy('/Users/amankar/GBO_Emily/copy.raw')
    f2 = GbtRaw('/Users/amankar/GBO_Emily/copy.raw', update=True)

    # simply read in each block, and then write it back...
    for i in range(f2.get_num_blocks()):
        print "copying block: ", i
        header, data = f2.get_block(i)
        f2.put_block(header, data, i)

def main4():
    import pylab as plt
    path    = '/Users/amankar/GBO_Emily/'
    in_file = 'guppi_56465_J1713+0747_0006.0000.raw'
    g = GbtRaw(path+in_file)
    n = g.get_num_blocks()
    blocks=5
    print "the file has", n, "blocks"
    tsData = g.extract(0,blocks, overlap=False)
    #g.pow_spec(0,n,0,29, overlap=False, out_file = 'new.fits')
    
if __name__ == "__main__":
#    print "executing main1"
#    main1()
#    print "executing main2"
#    main2()
#    print "executing main3"
#    main3()
    print "executing main4"
    main4()
    
