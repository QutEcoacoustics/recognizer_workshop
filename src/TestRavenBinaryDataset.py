

import RavenBinaryDataset
import Spectrogram
import librosa
import numpy as np
import random
import pandas as pd


wav_path_pos = "./data_whipbird/pos"
wav_path_neg = "./data_whipbird/neg"


wav_path_pos = "./data_boobook/positive"
wav_path_neg = "./data_boobook/negative"


spec_image_dir_path = "./specs2"

spec_params = {}
spec_params["fftOverlap"] = 0.5
spec_params["fftWinSize"] = 512
spec_params["maxFreq"] = 4000
spec_params["timeWin"] = 2.0


RavenBinaryDataset.prepare_data( wav_path_pos, wav_path_neg, spec_image_dir_path, spec_params)


'''
class_repetitions = {}
class_repetitions["pos"] = 5
class_repetitions["neg"] = 5


ds_train, ds_test = RavenBinaryDataset.MakeRavenBinaryDatasetSplit( "data.csv", "data_large.csv", 9135, 10, spec_params, class_repetitions )  


ds_train.__getitem__(150)
ds_train.__getitem__(150)
ds_train.__getitem__(150)
#ds_train.__getitem__(4)
#ds_train.__getitem__(4)
#ds_train.__getitem__(3)
'''












