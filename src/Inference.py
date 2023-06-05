
import os
import time
from datetime import datetime
import torch
import torchvision.transforms as transforms
import numpy as np
from typing import List
import librosa

# Our includes
import utils
from utils import Annot
import NeuralNets
import Spectrogram



def process_detections(f_path, out_dir, anns: List[Annot], logFile):
    """
    Process the detections
    Merge overlapping detections into single detection
    Write Raven file
    Append to log file
    """
    # Make file name for output Raven file
    head, f_name = os.path.split(f_path)
    f_name = os.path.splitext(f_name)[0]+'.txt'
    
    # Do any merging of annotations
    anns_merged: List[utils.Annot] = []
    ann_prev : Annot = None
    for ann in anns:
        if ann_prev != None:
            if ann_prev.time_is_inside(ann.start):
                ann_prev.end = ann.end
            else:
                anns_merged.append(ann)
                ann_prev = ann
        else:
            anns_merged.append(ann)
            ann_prev = ann
    
    # Write to log
    #fout = open( log_file_path, 'a')
    logFile.write(f_path + "\n")
    for ann in anns_merged:
        logFile.write(str(ann.start) + " " + str(ann.end) + "\n")
    
    # Write Raven file
    out_path = os.path.join(out_dir, f_name)
    utils.write_raven_anns(anns_merged, out_path)



def infer_wav_file(network, wav_path, infer_params, spec_params):
    """
    Do inference on a wav file using the given network
    """

    # First read the wav file
    start_time = time.time()

    print("\n---------------------------------")
    print("Reading wav file: " + wav_path)

    data, sample_rate = librosa.core.load( wav_path, sr=None )
    num_samples = data.size
    duration = num_samples / sample_rate

    print("Number of samples: " + str(num_samples))
    print("Duration: " + str(duration))

    # Get parameters for easy access
    maxFreq = int(spec_params["maxFreq"])
    fftOverlap = float(spec_params["fftOverlap"])
    fftWinSize = int(spec_params["fftWinSize"])
    timeWin = float(spec_params["timeWin"])
    pixelsPerSecond = 256.0 / timeWin
    step_size = infer_params["hop"]
    tot_steps = int(duration / step_size) - 1

    # This is the maximum number of image patches per wav file that we will save to disk
    # when there are detections within a file
    maxFileImages = int(infer_params["maxFileImages"])
    
    # Gnerate the spectrogram
    print("\n---------------------------------")
    print("Generating spectrogram")    
    print("FFT Window size: " + str(fftWinSize))
    print("FFT overlap: " + str(fftOverlap))
    print("Time window: " + str(timeWin))
    print("Max frequency: " + str(maxFreq))    
    print("Pixels per second: " + str(pixelsPerSecond))
    print("Hop size: " + str(float(infer_params["hop"])))

    spec = Spectrogram.Spectrogram(data, sample_rate, duration)
    spec.make_spec(fftWinSize, fftOverlap, maxFreq, timeWin)

    img = spec.get_image()
    
    # May need to revisit this transform
    # Will be better to pass transform in as a parameter
    transform = transforms.Compose([
        transforms.ToTensor(),            
        transforms.Normalize([0.3], [0.3])    
    ])

    anns: List[utils.Annot] = []
    detection_count = 0
    maxFileImages = int(infer_params["maxFileImages"])

    if not os.path.exists(infer_params["imageDir"]):
        os.makedirs(infer_params["imageDir"], exist_ok=True)

    # Step along the spectrogram 
    for i in range(0, tot_steps):

        # Get spec image patch as a tensor
        step = step_size * i  
        img = spec.get_image_patch(step, timeWin, pixelsPerSecond)
        img_tensor = transform(img).unsqueeze_(0)

        # Put image data through NN
        output = network(img_tensor)

        # Get prediction
        idx = np.argmax(output.data).item()

        # 0 is the background/noise class        
        if idx != 0:

            detection_count += 1

            # Make annotation and append to list
            ann = utils.Annot( step, step + timeWin)
            ann.ctype = infer_params["speciesName"]
            anns.append(ann)
            
            # Print detections
            print("Detection: " + str(step))
            
            # Check if need to write images
            if maxFileImages > 0 and detection_count <= maxFileImages:
                # x 10 is for making image names unique 
                img_path = os.path.join(infer_params["imageDir"], utils.get_count_str(int(step*10)) + ".png") 
                img.save(img_path)

        if step % 200 == 0:
            print(step)

    print("Elapsed time:  %.2f" % (time.time() - start_time))
        
    return anns



def do_inference( infer_params, spec_params):

    """
    Loop over wav files in the given directory and do inference
    """
    
    # Setup network
    network = NeuralNets.CNN_4_Layers(512, 2, 12, 24, 32, 48)
    network_state_dict = torch.load( infer_params["model"] )
    network.load_state_dict(network_state_dict)
    network.eval()

    # We don't want to update the gradients
    torch.no_grad()

    # Start the log file with parameter values
    log_dir = os.path.dirname(infer_params["log"])
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log = open( infer_params["log"], 'w')
    log.write( "\n-----------\nParameters\n-----------\n\n" )
    log.write( utils.params_to_string(spec_params) )
    log.write( "\n\n" )
    log.write( utils.params_to_string(infer_params) )
    log.write( "\n\n" )    
    log.write( "\n-----------\nProcessing\n-----------\n" )

    # Write the current time to the log file
    now = datetime.now()    
    log.write( "\nTime:  " + now.strftime("%H:%M:%S") + "\n\n")

    # Loop over files in directories
    for root, d_names, f_names in os.walk( infer_params["wavDir"] ):
        for f in f_names:
            if f.endswith(".wav") or f.endswith(".WAV"):
                f_path = os.path.join(root, f)
                anns = infer_wav_file(network, f_path, infer_params, spec_params)
                process_detections(f_path, infer_params["outputDir"], anns, log)
                log.flush()









