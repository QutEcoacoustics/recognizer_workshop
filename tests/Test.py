
from turtle import width
from PIL import Image as im

import utils
import NeuralNets
import RavenBinaryDataset
import TrainTest
import Inference


print("------------\n Spec params\n--------------")
spec_params = utils.read_spec_params( "config.ini" )
utils.print_params(spec_params)

print("\n------------\n Train params\n--------------")
train_params = utils.read_train_params( "config.ini" )
utils.print_params(train_params)

print("\n------------\n Inference params\n--------------")
infer_params = utils.read_infer_params( "config.ini" )
utils.print_params(infer_params)



wav_path_pos = "./sample_data/whipbird/pos"
wav_path_neg = "./sample_data/whipbird/neg"
spec_image_dir_path = "./training_output/preparation/specs/"



#---------------------------
#
# Prepare the data
#
#----------------------------

#RavenBinaryDataset.prepare_data( wav_path_pos, wav_path_neg, spec_image_dir_path, spec_params, "./data.csv")



#---------------------------
#
# Train
#
#----------------------------

train_params["baseModel"] = "./models/model_synth_base.pth" 

train_params["epochs"] = 10
#train_params["testSetSize"] = 4


TrainTest.train(train_params, spec_params)



#---------------------------
#
# Inference
#
#----------------------------

# infer_params["wavDir"] = "C:\Acoustics\Deployment_1"

# Inference.do_inference( infer_params, spec_params)










