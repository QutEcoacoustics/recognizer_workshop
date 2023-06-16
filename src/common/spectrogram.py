from scipy import signal
from PIL import Image
import numpy as np
import math
#TODO: look into torchlibrosa for generating mel spectrograms on gpu?
import librosa
import os


import common.utils


def create_spectrogram(file_source, image_destination=None, cols_per_second=(224/4), h=224, mel=True, highpass=200, w=-1, limit=10000):
    """
    Creates a spectrogram from the wav file at file_source
    :param file_source: str
    :param image_destination: str optional. Where to save the png files. If omitted will not save files
    :param cols_per_second: number of spectrogram columns per second
    :param h: height of the spectrogram
    :param mel: bool, whether to use a mel spectrogram
    :param highpass: int. Remove frequencies below this
    :param w: int. If > 0, will return a list of spectrograms of this width (in pixels)
    :param limit: int. If w > 0, will limit the number of segments in the list to this many, i.e. will only return the first [limit] segments. Set to a very high value for all segments
    :return: np.array w < 1, else list of np.array
    """

    # we have decided to use 4 seconds of audio as imput to the 224 width cnn

    # the librosa version reads the samples as floating point, the wavefile.read as int
    # todo: use the 'duration' parameter to limit number of samples for debug
    # todo: try using res_type 'kaiser_fast' to increase the speed of this
    samples, sample_rate = librosa.core.load(file_source, sr=16000)
    duration = len(samples) / sample_rate

    if duration > 300:
        print(f"creating spectrogram of duration {duration / 60} mins,  this could take a while")

    # window-size should be nearest power of 2 of double the desired height
    # todo: consider longer window to give greater overlap. information at the edges of the window is
    # lost, therefore this method can include more of the original audio, but ends up blurring inthe time direction
    window_length = nearest_power_of_2(h*2, 'ceil')
    # leave this as a float for now to avoid accumulating rounding errors
    total_cols = cols_per_second * duration

    hop_length = round(len(samples) / total_cols)

    window = signal.get_window('hamming', window_length)

    if mel:
        # ok so we do things a bit differently with the mel spectrogram, since
        # the process of it allows us to specify the number of fbands (and also discard low frequencies)
        # it seems like we get problems from giving too many frequency bands.
        # for now, we will do less and then extrapolate.

        spectrogram = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_fft=window_length,
                                       hop_length=hop_length, window=window, n_mels=180, fmin=highpass)
    else:
        overlap = window_length - hop_length
        frequencies, times, spectrogram = signal.spectrogram(x=samples, fs=sample_rate, window=window, noverlap=overlap)
        # todo: highpass

    print(f'created spectrogram from file of length {duration} ({len(samples)} samples). sr={sample_rate}, window={window_length}, hop={hop_length}')

    vals = interp_height(spectrogram, h)

    if w > 0:
        # select the relevant part of the full spectrogram for each segment, into a list
        num_segments = math.floor(vals.shape[1] / w)
        if (limit < num_segments):
            print(f'segmenting {limit} ot of a possible {num_segments}')
            num_segments = limit

        segments = [vals[:, (i * w):(i * w + w)] for i in range(0, num_segments)]
    else:
        segments = [vals]

    # rescales/normalises values and replicates to 3 channels
    # TODO: should we do this in the dataloader pipeline as a transformation after random crop?
    # it would be more consistent, since training segments are variable length, and therefore more likely to have less contrast
    # in the randomly cropped segment.
    segments = [rescale_array_for_image(segment) for segment in segments]
    segments = [array_to_3channel_greyscale(segment) for segment in segments]

    if image_destination is not None:
        for seg_num in range(0, len(segments)):
            # if width was not specified, seg num will not be appended to fn
            fn = utils.fn_append(image_destination, seg_num) if w > 1 else image_destination
            save_image(segments[seg_num], fn, size=None)

    # if we are not segmenting, don't wrap in a list
    # if we are segmenting but we only happen to have 1 segment, do keep wrapped in list
    return segments if w > 1 else segments[0]



def rescale_array_for_image(vals, use_db=True):
    """
    converts to log scale, normalises the values to between 0 and 255, flips vertically so low freq is at the bottom.
    :param vals: np Array
    :param use_db: boolean. If true, will first take the log.
    :return: np array
    """

    max_val = vals.max()
    if (use_db):
        # add a tiny number to stop it going to neg infinity
        # the higher the number, the more of the low-level stuff is removed
        # this seems to need different values for mel or not mel - the mel has higher values and can tolerate a bigger value here.
        # not sure what's going on.
        vals = np.log(vals + 0.0000001)

    # shift so min val is zero
    shifted_zero = vals - vals.min()
    # scale so max val is 1
    scaled_one = shifted_zero / shifted_zero.max()

    # at this point everything is in the range [0,1]
    # scale so that the max is where it should be within [0,1]

    #max_possible_original_val = 100
    #vals = vals * (max_val / max_possible_original_val)
    # todo: clip at 255
    # scale and invert (black is high and white is low, just becasue that's what I am used to)
    scaled_255 = ((scaled_one * 255) - 255) * -1

    # make it upside down (low freq at bottom)
    flipped = np.flip(scaled_255, axis=0)

    return flipped


def array_to_3channel_greyscale(vals):
    """
    duplicates an nxm 2d array into a nxmx3 3d array
    This is so that we can put a spectrogram into a CNN architecture that expects color images.
    :param vals: np array
    :return: np array
    """
    # duplicate to RGB channels. Kind of a massive waste of space.
    # size is width by height
    h = vals.shape[0]
    w = vals.shape[1]
    image_data = np.zeros((h, w, 3), dtype=np.uint8)
    image_data[0:h, 0:w, 0] = vals
    image_data[0:h, 0:w, 1] = vals
    image_data[0:h, 0:w, 2] = vals

    return image_data


def save_images(image_data_list, destination_folder, size=None):
    """
    Saves each np array in the given list to an image file
    :param image_data_list: list of np arrays
    :param destination_folder: str
    :param size: type (int, int) optional, if supplied will resize the image before saving
    :return: None
    """
    for i in range(0, len(image_data_list)):
        cur_image_destination = f'{destination_folder}/{i}.png'
        save_image(image_data_list[i], cur_image_destination, size)

def save_image(image_data, image_destination, size=(224, 224)):
    """
    Saves an np array to an image file
    :param image_data: np array
    :param destination_folder: str
    :param size: type (int, int) optional, if supplied will resize the image before saving
    :return: None
    """

    img = Image.fromarray(image_data, 'RGB')
    # for now we are just using pillow's resizing.
    # maybe for future we will consider a more explicit way to do this to have more control

    if type(size) is tuple and image_data.shape != size:
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



# def test():
#     dir = os.path.dirname(os.path.realpath(__file__))
#     source_file = os.path.join(dir, 'test_files/15s_16khz.wav')
#     #destination_file = os.path.join(dir, 'test_files/test_output/15s_16khz.png')
#     create_segmented_spectrogram(source_file)


def interp_height(vals, new_height):
    """
    converts a column of the spectrogram to a different height with linear interpolation
    :param vals: 2d array
    :param height: the desired height of the 2d array
    :return: np.Array
    """
    # the target x coordinates
    x = np.linspace(0, vals.shape[0] - 1, new_height)
    # the current x coordinates are just the current row numbers (zero to current_height-1)
    xp = np.arange(vals.shape[0])
    # iterate over the cols by transposing and iterating over the arrays, then transposing back
    return np.array([np.interp(x, xp, col) for col in vals.transpose()]).transpose()




#test()
