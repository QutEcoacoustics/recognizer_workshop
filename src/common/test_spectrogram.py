from common.spectrogram import *
import numpy as np


def test_interp_height():
    from_array = np.array([[1, 4, 2, 3], [2, 2, 4, 1], [0, 0, 1, 0]])
    to_array = np.array([[1, 4, 2, 3],
       [1.5, 3, 3, 2],
       [2, 2, 4, 1],
       [1, 1, 2.5, 0.5],
       [0, 0, 1, 0.]])

    result = interp_height(from_array, 5)
    assert np.all(to_array == result)


def test_nearest_power_of_2():

    inputs = [1, 3, 30, 71, 1000]
    outputs_ceil = [1, 4, 32, 128, 1024]
    outputs_floor = [1, 2, 16, 64, 512]
    outputs_round = [1, 4, 32, 64, 1024]

    assert [nearest_power_of_2(val) for val in inputs] == outputs_ceil
    assert [nearest_power_of_2(val, 'ceil') for val in inputs] == outputs_ceil
    assert [nearest_power_of_2(val, 'floor') for val in inputs] == outputs_floor
    assert [nearest_power_of_2(val, 'round') for val in inputs] == outputs_round


def test_array_to_3channel_greyscale():

    input = np.array([[1, 2, 3], [4, 3, 2], [5, 1, 1]])
    output = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]],
       [[4, 4, 4], [3, 3, 3], [2, 2, 2]],
       [[5, 5, 5],  [1, 1, 1], [1, 1, 1]]])
    assert np.all(output == array_to_3channel_greyscale(input))

def test_spectrogram():
    """
    end to end spectrogram generation
    Check that the generated spectrogram exactly matches the saved example.
    :return:
    """

    res = create_spectrogram(file_source='./test_files/inference/4s_16khz.wav',
                       image_destination=None,
                       cols_per_second=224/4,
                       h=224,
                       mel=True,
                       highpass=200,
                       w=224,
                       limit=20)

    # it is a 4 sec recordings, so there should be only 1 segment
    assert len(res) == 1
    # check the shape is right
    assert res[0].shape == (224, 224, 3)
    # this particular recording happens to have this value as the centre pixel and mean
    assert res[0][111, 111, 0] == 121
    assert np.mean(res[0]).round(2) == 147.66



    print(type(res))







