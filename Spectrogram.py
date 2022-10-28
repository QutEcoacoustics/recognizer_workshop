
import numpy as np
from PIL import Image as im



class Spectrogram:
    '''
    This class implements the generation of spectrograms from a buffer
    of audio samples.
    '''

    def __init__(self, data_, sample_rate_, duration_):
        
        self.sample_rate = sample_rate_
        self.duration = duration_
        self.data = data_
        self.log_coef = 8
        self.num_freq_bins = 0                        
        self.img = None


    def make_spec(self, fft_win_size, fft_overlap, max_freq, time_win):
        """
        Creates a linear spectrogram image given the spectrogram parameters 
        """
        num_freq_bins = int( max_freq / self.sample_rate / 2 * fft_win_size)
        data_ = self.data
        
        noverlap = int( fft_overlap * fft_win_size )

        step = fft_win_size - noverlap
        shape = (fft_win_size, (data_.shape[-1]-noverlap)//step)
        strides = (data_.strides[0], step * data_.strides[0])
        
        d_wins = np.lib.stride_tricks.as_strided(data_, shape=shape, strides=strides)

        # apply window
        d_wins = np.hanning(d_wins.shape[0]).astype(np.float32)[..., np.newaxis] * d_wins
        complex_spec = np.fft.rfft(d_wins, axis=0)

        # calculate magnitude and take sqrts
        self.spec = np.sqrt(np.sqrt(np.absolute(complex_spec)**2))

        # remove DC component and flip vertical orientation
        self.spec = np.flipud(self.spec[1:, :])

        # Get only the part of the spectrogram that we need
        self.spec = self.spec[256-(num_freq_bins*2):256,:]

        # log scale the values
        self.spec = np.log1p( self.log_coef * self.spec )

        # convert to byte values
        max = np.max(self.spec)    
        self.spec = self.spec * 255/max
        self.spec = self.spec.astype(np.uint8)        

        # Make image
        time_units = self.duration / time_win    
        self.img = im.fromarray(self.spec)
        self.img = self.img.resize( (int(256 * time_units), 128), resample = im.LANCZOS )


    def get_image_patch(self, offset, duration, pixels_per_second):
        """
        Get the image patch given the offset and the duration
        """        
        xpos1 = offset  * pixels_per_second
        xpos2 = xpos1 + duration * pixels_per_second
        return self.img.crop((xpos1, 0, xpos2, 128))


    def get_image(self):
        """
        Get the entire spectrogram image
        """
        return self.img






