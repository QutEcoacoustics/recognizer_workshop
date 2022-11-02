

from os import listdir
import math
import random
import configparser
import pandas as pd
from typing import List
import librosa

import Spectrogram


class Annot:
    """
    Simple class for annotations
    """
    start: float
    end: float
    ctype: str

    def __init__(self, start, end):
        self.start = start
        self.end = end

    # Check if the given time is inside the interval represented by this annotation
    def time_is_inside(self, val: float):
        if val >= self.start and val <= self.end:
            return True
        else:
            return False
        
    
def normalize_map_key_names(params):
    """
    Modifies the keys of the given map so that they are in a 'normalized' form.

    This is useful when reading in values from a config file.
    """
    resultMap = {}
    for k,v in params.items():
        strs = k.split("-")
        newStr = strs[0]
        for s in strs[1:len(strs)]:
            sl = list(s)
            sl[0] = sl[0].upper()
            newStr += ("".join(sl))        
        resultMap[newStr] = v
    return resultMap


def read_config(file_name: str, section: str):
    """
    Reads the values in the given section of a config file 
    """
    if file_name != None:  
        config = configparser.ConfigParser()
        config.read(file_name)
        params = dict(config.items(section))
        params = normalize_map_key_names(params)
        return params
    else:
        return {}

    
def read_spec_params(file_name: str):
    """
    This is a more specific version of the "read_config" function 
    """
    if file_name != None:  
        config = configparser.ConfigParser()
        config.read(file_name)
        
        params = {}       
        params["maxFreq"] = int(config["spec"]["max-freq"])
        params["timeWin"] = float(config["spec"]["time-win"])
        params["fftWinSize"] = int(config["spec"]["fft-win-size"])

        # Keep this fixed for the moment        
        params["fftOverlap"] = 0.5
        
        return params
    else:
        return {}
    
    
def read_train_params(file_name: str):
    """
    This is a more specific version of the "read_config" function 
    """    
    if file_name != None:  
        config = configparser.ConfigParser()
        config.read(file_name)
        
        params = {}       
        params["dataCSV"] = config["train"]["data-csv"]
        params["epochs"] = int(config["train"]["epochs"])
        params["lr"] = float(config["train"]["lr"])        
        params["batchSize"] = int(config["train"]["batch-size"])
        params["testSetSize"] = int(config["train"]["test-set-size"]) 
        params["baseModel"] = config["train"]["base-model"]        
        params["trainedModel"] = config["train"]["trained-model"]
        params["log"] = config["train"]["log"]
        params["saveImagePatches"] = config["train"]["save-image-patches"] 
        params["randomSeed"] = int(config["train"]["random-seed"])         
        return params
    else:
        return {}    
    

def read_infer_params(file_name: str):
    """
    This is a more specific version of the "read_config" function 
    """    
    if file_name != None:  
        config = configparser.ConfigParser()
        config.read(file_name)
        
        section = "infer"
        
        params = {}       
        params["hop"] = float(config[section]["hop"])
        params["model"] = config[section]["model"]
        params["speciesName"] = config[section]["species-name"]        
        params["wavFile"] = config[section]["wav-file"]
        params["wavDir"] = config[section]["wav-dir"]        
        params["recursive"] = config[section]["recursive"]
        params["imageDir"] = config[section]["image-dir"] 
        params["outputDir"] = config[section]["output-dir"]
        params["maxFileImages"] = int(config[section]["max-file-images"]) 
        params["log"] = config[section]["log"]         
        
        return params
    else:
        return {} 
    

def print_params(params):
    """
    Prints the keys and values in a map
    """
    for k,v in params.items():
        print("" + k + " : " + str(v)) 
        

def params_to_string(params):
    """
    Constructs a string from the keys and values in a map
    """
    str_value = ""
    for k,v in params.items():
        str_value = str_value + k + " : " + str(v) + "\n" 

    return str_value


def get_count_str(count: int):
    """
    Converts an integer count into a string with the correct number of zeros 
    """
    count_str = "000"
    if count>9:
        count_str = "00"
    if count>99:
        count_str = "0"
    if count>999:
        count_str = ""
    count_str += str(count)
    return count_str


def get_annotations(ann_path: str):
    """
    Loads a Raven annoytations file and stores it as a list of .
    Need to be careful to not duplicate rows - 1 for waveform and 1 for spectrogram.
    """

    # Read annotation file
    df = pd.read_csv(ann_path, delimiter='\t' )
    
    # Set of selection numbers
    selnums = set()

    # Get annotations
    anns: List[Annot] = []

    step = 1
    
    for i in range(0,df.shape[0], step):
        # Make sure that we don't include annotation more than once
        selnum = df.iloc[i]['Selection']
        if selnum in selnums:
            continue
        selnums.add(selnum)
        start_time = df.iloc[i]['Begin Time (s)']
        end_time = df.iloc[i]['End Time (s)']    
        a = Annot(start_time, end_time)
        if 'Call Type' in df.columns:
            a.ctype = df.iloc[i]['Call Type']
        else:
            a.ctype = ""
        if 'Label' in df.columns:
            a.ctype = df.iloc[i]['Label']
        anns.append(a)
    
    return anns


def find_filenames( path_to_dir, suffix=".csv" ):
    """
    Retrieve list of files with given extension
    """
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]


def get_wav_length(fpath):
    """
    Gets the length of a wav file using librosa
    """
    samples, sample_rate = librosa.core.load(fpath)
    duration = len(samples)/sample_rate
    return duration


def matching_annotation_file(dir, wfile):
    """
    Check if wav file has a matching annotation file by matching the prefix 
    """
        
    prefix = wfile[0:-4]
    files = find_filenames( dir, suffix=".txt" )    
    for f in files:
         if f.startswith(prefix):
             return f

    return ""   


def write_raven_anns(anns: List[Annot], file_path):
    """
    Write a Raven annotation file with a given set of column headers
    """
    
    header = "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tDelta Time (s)\tDelta Freq (Hz)\tPeak Amp (U)\tSpecies"

    with open(file_path, 'w') as f:
        f.write(header+"\n")
        for idx, ann in enumerate(anns):
            dtime = ann.end - ann.start
            astr = str(ann.start)+"\t"+str(ann.end)+"\t500.00\t7000.00\t"+str(dtime)+"\t6500.00\t0\t" + ann.ctype + "\n"
            f.write(""+str(idx+1)+"\tSpectrogram 1\t1\t"+astr)
            

def makeSpecFromWav(wav_file_path, spec_image_path, fftWinSize, fftOverlap, maxFreq, timeWin):
    
    '''
    Processes a single wav file.
    Generates spectrogram for entire wav file and adds to the spec directory.
    '''    

    data, sample_rate = librosa.core.load( wav_file_path, sr=None )
    duration = len(data)/sample_rate

    spec = Spectrogram.Spectrogram(data, sample_rate, duration)
    spec.make_spec( fftWinSize, fftOverlap, maxFreq, timeWin)
    img = spec.get_image()
    img.save( spec_image_path )
    
 
def balance_dataset(df):
    '''
    oversamples from the poorer class so that the number of rows is the same per class
    '''

    print(f'balancing dataset of {df.shape[0]} rows')

    counts = df['label'].value_counts()
    target_num = max(counts)
    new_indexes = []
    
    random.seed(4321)

    for label in counts.index:

        cur_label_indexes = list(df.index[df['label'] == label])
        # the number times to repeat all the examples
        num_reps = math.floor(target_num / counts[label])
        # the number of random rows to add so that we get exactly the target number
        num_remainder = target_num % counts[label]
        # add these repetitions and randomly selected indexes
        cur_label_new_indexes = (cur_label_indexes * num_reps) + random.sample(cur_label_indexes, num_remainder)
        new_indexes = new_indexes + cur_label_new_indexes

        print(f'number of {label} examples went from {counts[label]} to {len(cur_label_new_indexes)}')

    new_df = df.iloc[new_indexes].reset_index(drop=True)

    return(new_df)



