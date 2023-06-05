from scipy import signal
from scipy.io import wavfile
from PIL import Image
import numpy as np
import math
#TODO: look into torchlibrosa for generating mel spectrograms on gpu?
import librosa
import os


def create_spectrogram(file_source, image_destination, cols_per_second=(224/4), h=224, mel=True, overlap=0.2):

    # we have decided to use 4 seconds of audio as imput to the 224 width cnn

    # the librosa version reads the samples as floating point, the wavefile.read as int
    samples, sample_rate = librosa.core.load(file_source, sr=16000)
    duration = len(samples) / sample_rate

    # window-size should be nearest power of 2 of double the desired height

    window_length = nearest_power_of_2(h*2, 'ceil')
    # leave this as a float for now to avoid accumulating rounding errors
    total_cols = cols_per_second * duration
    hop_length = round(len(samples) / total_cols)

    window = signal.get_window('hamming', window_length)


    if mel:
        # ok so we do things a bit differently with the mel spectrogram, since
        # the process of it allows us to specify the number of fbands (and also discard low frequencies)
        spectrogram = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_fft=window_length,
                                       hop_length=hop_length, window=window, n_mels=180, fmin=200)
    else:
        overlap = window_length - hop_length
        frequencies, times, spectrogram = signal.spectrogram(x=samples, fs=sample_rate, window=window, noverlap=overlap)

    vals = rescale_array_for_image(spectrogram)
    #https://pillow.readthedocs.io/en/stable/_modules/PIL/Image.html#Image.resize
    destination_size = (round(total_cols), h)

    array_to_3channel_greyscale(vals, image_destination, size=destination_size)




def create_spectrogram_fixed_width(file_source, image_destination, w=224, h=224, mel = True):
    """
    Creates a spectrogram from an audio file and saves it as a (png?)
    :param file_source: string audio file path
    :param image_destination: string png file path
    :param mel: bool; whether to use a mel scale
    :param w: width
    :param h: height
    :return:
    """

    print(f"creating spectrogram: source: {file_source}, destination: {image_destination}")
    if not os.path.exists(file_source):
        print("wav file missing")
        return False

    #sample_rate, samples = wavfile.read(file_source)

    # the librosa version reads the samples as floating point, the wavefile.read as int
    samples, sample_rate = librosa.core.load(file_source, sr=None)
    duration = len(samples) / sample_rate

    # we put a constraint that we want at least this much overlap.
    # this ensures that all the signal is used (edges of the window are removed through the window function)
    min_overlap = 0.2
    min_window_size_for_time_frames_sec = (duration / w) / (1-min_overlap)
    min_window_size_for_time_frames_samples = min_window_size_for_time_frames_sec * sample_rate
    min_window_size_for_freq_bins_samples = h*2

    window_length = nearest_power_of_2(max(min_window_size_for_time_frames_samples, min_window_size_for_freq_bins_samples), 'ceil')


    # we want the output spectrogram to be 224x224
    # we can control the width accurately with the overlap
    # we can't control the height because we are limited to a power of 2
    # the height is half of the window size, but the window size must be a power of 2
    # therefore the window size should be the nearest power of 2 above the , and then we will resize the image to fit

    # if mel:
    #     # for mel, we want a higher frequency resolution so that the mel filtering has more to work with
    #     # in the lower frequencies
    #     window_length = nearest_power_of_2(h*4, 'ceil')
    #
    # else:
    #     # num freq bins is half this, which is close to 224 (cnn input image size)
    #     # we resize the image later, but we want to start with as close as possible
    #     window_length = nearest_power_of_2(h*2, 'ceil')

    # the perfect overlap is determined by the frequency and the window size

    # hop length means a spectrogram column every this many samples
    hop_length = math.floor((len(samples) - window_length) / (w - 1))

    # I guess this could be negative (space between windows) if the recording is long or high sample rate
    # meaning there is a gap
    overlap = window_length - hop_length

    window = signal.get_window('hamming', window_length)

    # to do: understand the kind of numbers coming out of this
    # need to know the range of values, especially the max value
    # so it can be normalised to 255

    if mel:
        # ok so we do things a bit differently with the mel spectrogram, since
        # the process of it allows us to specify the number of fbands (and also discard low frequencies)
        spectrogram = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_fft=window_length,
                                       hop_length=hop_length, window=window, n_mels=100, fmin=200)
    else:
        frequencies, times, spectrogram = signal.spectrogram(x=samples, fs=sample_rate, window=window, noverlap=overlap)



    vals = rescale_array_for_image(spectrogram)

    array_to_3channel_greyscale(vals, image_destination)


    # plt.pcolormesh(times, frequencies, spectrogram)
    # plt.imshow(spectrogram)
    # plt.savefig(image_destination)

    # this is the success status. I guess if we get here it's success
    return True

def rescale_array_for_image(vals, use_db=True):

    # we need to convert the spectrogram values to proper values between 0 and 255,
    # but we don't want to just normalise or it won't be consistent between samples.
    # ... not sure the best way to do this. I guess we can work out the max possible
    # value and make that equal to 255? Max would be

    max_val = vals.max()
    if (use_db):
        # add a tiny number to stop it going to neg infinity
        # the higher the number, the more of the low-level stuff is removed
        # this seems to need different values for mel or not mel - the mel has higher values and can tolerate a bigger value here.
        # not sure what's going on.
        vals = np.log(vals + 0.0000001)

    # for now we will just normalise
    # invert it so that high is black (just becasue that's what I am used to

    # shift so min val is zero
    shifted_zero = vals - vals.min()
    # scale so max val is 1
    scaled_one = shifted_zero / shifted_zero.max()

    # at this point everything is in the range [0,1]
    # scale so that the max is where it should be within [0,1]

    #max_possible_original_val = 100
    #vals = vals * (max_val / max_possible_original_val)
    # todo: clip at 255
    # scale and invert (black is high and white is low
    scaled_255 = ((scaled_one * 255) - 255) * -1

    # make it upside down
    flipped = np.flip(scaled_255, axis=0)

    return flipped

def array_to_3channel_greyscale(vals, image_destination, size=(224,224)):
    # duplicate to RGB channels. Kind of a massive waste of space.
    # size is width by height
    h = vals.shape[0]
    w = vals.shape[1]
    image_data = np.zeros((h, w, 3), dtype=np.uint8)
    image_data[0:h,0:w,0] = vals
    image_data[0:h,0:w,1] = vals
    image_data[0:h,0:w,2] = vals

    img = Image.fromarray(image_data, 'RGB')
    # for now we are just using pillow's resizing.
    # maybe for future we will consider a more explicit way to do this to have more control
    img = img.resize(size)
    img.save(image_destination)


def nearest_power_of_2(val, mode='ceil'):
    """
    returns the nearest power of two to a given value
    :param val: numeric
    :param mode: str; ceil, floor or round
    :return: int
    """
    if mode == 'ceil':
        exp = math.ceil(math.log2(val))
    elif mode == 'floor':
        exp = math.floor(math.log2(val))
    else:
        exp = round(math.log2(val))

    return int(math.pow(2, exp))



def test():
    dir = os.path.dirname(os.path.realpath(__file__))
    source_file = os.path.join(dir, 'test_files/4s_16khz.wav')
    destination_file = os.path.join(dir, 'test_files/test_output/4s_16khz.png')
    create_spectrogram(source_file, destination_file)


#test()