# duplication of the notebook as a plain python file

import os
from os import listdir
import argparse

import importlib

#import math
#import numpy as np
##import pandas as pd
import librosa
#import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image

#import configparser

#import torch
#import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim

import utils
import config
#import Spectrogram
import RavenBinaryDataset
import TrainTest
import NeuralNets_3FCL as NeuralNets

import Inference

importlib.reload(utils)

parser = argparse.ArgumentParser(description='Run training and inference')

parser.add_argument('--config', type=str,
                    help='path to config file')
parser.add_argument('--pos', type=str, 
                    help='folder of positive training examples')
parser.add_argument('--neg', type=str, 
                    help='folder of negative training examples')
parser.add_argument('--specs' , type=str, 
                    help='folder where to save/read spectrograms')
parser.add_argument('--wav-dir' , type=str, 
                    help='folder where find audio for inference')

args = parser.parse_args()

spec_params = config.read_spec_params(args.config)
config.print_params(spec_params)

train_params = config.read_train_params(args.config)
config.print_params(train_params)

wav_path_pos = args.pos
wav_path_neg = args.neg
spec_image_dir_path = args.specs

RavenBinaryDataset.prepare_data( wav_path_pos, wav_path_neg, spec_image_dir_path, spec_params, train_params['dataCSV'])

train_params = config.read_train_params(args.config)
config.print_params(train_params)

TrainTest.train(train_params, spec_params)

infer_params = config.read_infer_params(args.config)
config.print_params(infer_params)

Inference.do_inference( infer_params, spec_params, args.wav_dir)


