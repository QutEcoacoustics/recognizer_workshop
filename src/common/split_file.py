import subprocess
import re
import os
import math
import time

import spectrogram

redo_if_wav_exists = False
redo_if_img_exists = False
# actually creating the wav/spectrograms takes the most time
# set this to false to run through creation of the segments dataframe without actually creating the files
create_files = True

delete_wavs = True

def get_duration(filename):
    """
    Determines the duration of the given audio file by calling the sox command and
    parsing the output with regex
    :param filename:
    :return:
    """
    command = ['sox', filename, '-n', 'stat']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    stats = stderr.decode('ascii')
    try:
        #duration = float(re.search('([0-9.]+)', stats.splitlines()[1]).group(1))
        duration = float(re.search('Length.+seconds[^0-9.]+([0-9.]+)', stats).group(1))

    except:
        print("something went wrong with command: " + ' '.join(command))
        print(stats)
        duration = False

    return duration



#test_get_start_offsets()


def get_start_offsets(duration, num_segments=-1, spacing=1, length=4):
    """
    For a given duration and number of segments and length, or spacing, returns the start offsets
    if num_segments is supplied:
    - it will space them out as wide as possible, if only 1 it will be in the middle
      (todo: maybe make the space between the edge the same as the space between segments?)
    - spacing parameter is ignored
    if num_segments is not supplied, will simply use spacing

    :param duration:
    :param num_segments:
    :param spacing:
    :param length:
    :return:
    """

    # default to no overlap
    if spacing is None:
        spacing = length

    if num_segments == 1:
        # if there is only one segment, put it in the middle of the recording
        start_offsets = [(duration - length) / 2]
    else:
        if num_segments < 1:
            # calculate num segments from spacing
            num_segments = int(math.floor((duration - length) / spacing) + 1)
        else:
            # adjust spacing we we go all the way to the end of the file
            spacing = (duration - length) / (num_segments - 1)
        start_offsets = [s*spacing for s in range(0, num_segments)]

    return start_offsets


def split_file(filename, destination_folder, num_segments=-1, spacing=1, segment_duration=4, recording_duration=None):
    """

    Splits a file up into either the number of segments specified or with the spacing specified
    if the number of segments is specified, the spacing will be determined from that (using maximum spacing possible)

    :param filename: str; the source filename
    :param destination_folder: str;
    :param duration: float; the duration of the source file. If omitted, will use sox stat to look it up
    :param num_segments: int; how many segments do we want, optional: if not supplied, is determined by spacing
    :param spacing: float; how far apart are the start of each segment. optional; only needed if num_segments not given.
    :param length: float; duration of segments
    :return:
    """

    wav_paths = []
    image_paths = []

    wav_folder = os.path.join(destination_folder, 'wav')
    png_folder = os.path.join(destination_folder, 'png')
    if not os.path.exists(wav_folder): os.makedirs(wav_folder)
    if not os.path.exists(png_folder): os.makedirs(png_folder)

    # get the duration
    if not recording_duration:
        recording_duration = get_duration(filename)

    start_offsets = get_start_offsets(recording_duration, num_segments, spacing, segment_duration)

    seg_nums = range(0, len(start_offsets))

    # basename_no_ext = os.path.splitext(os.path.basename(filename))[0]
    # def create_dest_name(seg_num):

    destination_names = [os.path.splitext(os.path.basename(filename))[0] + "_" + str(seg_num).zfill(2) + '_' + str(int(round(start_offsets[seg_num] * 100))).zfill(4) for seg_num in seg_nums]
    wav_filenames = [destination_name + ".wav" for destination_name in destination_names]
    image_filenames = [destination_name + ".png" for destination_name in destination_names]
    # wav_paths = wav_paths + [destination_wav_name]
    # image_paths = image_paths + [destination_image_name]
    destination_wav_paths = [os.path.join(wav_folder, wav_fn) for wav_fn in wav_filenames]
    destination_image_paths = [os.path.join(png_folder, png_fn) for png_fn in image_filenames]


    for seg_num in seg_nums:
        start_offset = start_offsets[seg_num]
        #print(f'extract clip {seg_num} from {start_offset} to {start_offset + length}')
        # segment name contains the filename, the index, and the number of ms offset (padded to 2 and 4 respectivly

        if not create_files:
            # during debugging
            continue

        destination_image_path = destination_image_paths[seg_num]
        destination_wav_path = destination_wav_paths[seg_num]

        if not os.path.exists(destination_image_path) or redo_if_img_exists:

            # wav is only created if image doesn't exist!
            if not os.path.exists(destination_wav_path) or redo_if_wav_exists:
                do_split(filename, destination_wav_path, start_offset, segment_duration)

            # sometimes the wav file is mysteriously not there. In this case, wait a sec I guess
            if not os.path.exists(destination_wav_path):
                time.sleep(0.1)

            try:
                success = spectrogram.create_spectrogram(destination_wav_path, destination_image_path)
                if success and delete_wavs:
                    os.remove(destination_wav_path)
            except:
                success = False
                print("An exception occurred")



        else:
            print("*", end='')

    #TODO: handle image creation failure

    return wav_filenames, image_filenames



def do_split(filename, destination_wav_path, start_offset, length):
    """
    Extracts a single segment from a wav file using sox
    :param filename:
    :param destination_wav_path:
    :param start_offset:
    :param length:
    :return:
    """

    #todo: don't fail so silently

    sox_command = ['sox', filename, '-r 16000', destination_wav_path, 'trim', str(start_offset), str(length)]
    print("sox command: " + " ".join(sox_command))
    process = subprocess.Popen(sox_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()


def test_resample (
        input_file='test_files/15s_16khz.wav',
        output_folder='test_files/test_output/resampled'):

    # this is just to see the difference between the sox command with the resample before or after the destination
    # file. Seems to make a difference.

    # this one seems to just change the 'declared' sample rate, so it plays samples at a faster rate
    # which shortens the duration and increases the frequency
    output_1 = output_folder + '/resample_before.wav'
    sox_1 = ['sox', '-r', '22050', input_file,  output_1, 'trim', '0', '4']

    process = subprocess.Popen(sox_1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print("sox error: " + stderr.decode('ascii'))

    # this actually resamples, which leaves a gap in the top frequency in the spectrogram (since we are upsampling)
    output_2 = output_folder + '/resample_after.wav'
    sox_2 = ['sox', input_file, '-r', '22050',  output_2, 'trim', '0', '4']

    process = subprocess.Popen(sox_2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print("sox error: " + stderr.decode('ascii'))

    output_3 = output_folder + '/resample_none.wav'
    sox_3 = ['sox', input_file,  output_3, 'trim', '0', '4']

    process = subprocess.Popen(sox_3, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print("sox error: " + stderr.decode('ascii'))


def split_file_to_arrays(filename, destination_folder, num_segments=-1, spacing=1, segment_duration=4, recording_duration=None):
    """
    Converts a file to spectrogram, then splits the spectorgram into segments and returns this as a list of arrays
    :param filename:
    :param destination_folder:
    :param num_segments:
    :param spacing:
    :param segment_duration:
    :param recording_duration:
    :return:
    """

    spectro = spectrogram.create_spectrogram(destination_wav_path, destination_image_path)





#print(split_file('/Users/eichinsp/Documents/github/species_classifier/downloads/australasian_pipit_92841.wav', '/Users/eichinsp/Documents/github/species_classifier/dataset_files'))
