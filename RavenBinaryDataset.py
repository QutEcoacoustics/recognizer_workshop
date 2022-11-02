

import os
import random
import pandas as pd

from PIL import Image as im

from torch.utils.data import Dataset
import torchvision.transforms as transforms

import utils as utils



class RavenBinaryDataset(Dataset):
    """
    This class implements a custom PyTorch image dataset with only two classes.
    The classes are interpreted as "pos" and "neg".
    
    Information about the paths to the images and the annotations associated with the images
    is provided in a Pandas datafram.
    Each row of the dataframe is a single data item.
    The columns of the dataframe are expected to be:
    - label, start, end, wavpath, specpath, test

    This class is specific to Raven because the annotations are provided  
    """
    def __init__(self, data_frame, spec_params, rand_seed, transform = None):

        self.ran_Seed = rand_seed
        self.spec_params = spec_params

        # Stpre parameters for easy access and compute necessary derivative values
        self.patch_pixels_width = 256
        self.patch_pixels_height = 128
        self.fft_win_size = int(self.spec_params["fftWinSize"])
        self.fft_overlap = float(self.spec_params["fftOverlap"])
        self.time_win = float(self.spec_params["timeWin"])
        self.pix_per_sec = float(self.patch_pixels_width) / self.time_win
        self.max_freq = int(self.spec_params["maxFreq"])

        # Load data file
        self.data = data_frame          
        self.columns = self.data.columns.tolist()

        # Get labels - should only be two labels - "pos" and "neg"
        labels = list(self.data.label.unique())
        if not len(labels) == 2:            
            print("ERROR: There are not exactly 2 labels in the dataset.")

        # Initialise label to integer mapping
        self.label_to_index = {}
        self.create_label_to_index_map()

        # Set up transform
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()                    
            ])
        
        self.save_image_dir_path = ""
        self.log_file_name = ""
        
        self.log_file = None

        self.item_list = []


    def get_item_list(self):
        return self.item_list


    def __len__(self):
        return len(self.data.index)  


    def __getitem__(self, idx):
        # Get row of data frame for data item
        row = self.data.iloc[idx]
        
        label = row[0]
        start = float(row[1])
        end = float(row[2])
        wav_file_path = row[3]
        spec_image_file_path = row[4]
        
        class_idx = self.label_to_index[label]

        # Read spectrogram image
        img = im.open(spec_image_file_path)
        
        interval = end - start
        pos = 0
        left = 0.0
        right = 0.0
        
        # There are 2 cases
        # 1) short annotation -> sample around the box
        # 2) long annotation -> sample within the box

        # Case 1 - Sample around the box
        if interval <= self.time_win:

            left = end - self.time_win
            # Check for going off edge of image
            if left < 0:
                left = 0

            right = start + self.time_win

            # Check for going off edge of image
            if right * self.pix_per_sec > img.width:
                right = float(img.width) / self.pix_per_sec

            rand_offset_range = int((right - left - self.time_win) * self.pix_per_sec)
            left_pixel_pos = int(left * self.pix_per_sec)
            pos = left_pixel_pos + random.randint( 0, rand_offset_range )

        # Case 2 - Sample within the box        
        else:
            left = start
            right = end
            rand_offset_range = int((interval - self.time_win) * self.pix_per_sec)
            left_pixel_pos = int(start * self.pix_per_sec)
            pos = left_pixel_pos + random.randint( 0, rand_offset_range )

        img = img.crop((pos, 0, pos + self.patch_pixels_width, self.patch_pixels_height))  

        pos_in_secs = str(float(pos)/self.patch_pixels_width)

        # Check if we have already filled the list
        if len(self.item_list) == len(self):
            self.item_list = []    

        self.item_list.append( (idx, class_idx, pos_in_secs, wav_file_path ) )
        
        '''
        cstr = utils.get_count_str(idx)
        if class_idx==0:
            img.save("./trainPatches/0/p_"+cstr+".png")
        
        if class_idx==1:
            img.save("./trainPatches/1/p_"+cstr+".png")
            
        #print(img.width, img.height)
        self.log_file.write(cstr+"   " + row[3]+ "   "+str(left)+"   "+str(right)+"   "+str(float(pos)/self.patch_pixels_width*self.time_win)+"\n") 
        '''

        #self.log_file.write(str(idx) + "  :  " + str(class_idx) + "\n")

        img_tensor = self.transform(img)

        return img_tensor, class_idx
       

    def create_label_to_index_map(self):
        '''
        Maps the class names to an integer        
        '''        
        self.label_to_index["pos"] = 1
        self.label_to_index["neg"] = 0


def create_spec( wav_path, spec_dir, spec_params):
    """
    Creates the spectrogram for a wav file and puts it in spec_dir.
    The spectrogram file name is made by counting the number of existing ".png"
    files in the directory and making a string from this count.

    The spectrogram file name therefore doesn't have any relationship to the name
    of the wav file from which it is computed. This is because it is possible that wav
    files with the same name may appear in different directories. 
    """
    # Get number of existing png files in directory
    files = utils.find_filenames( spec_dir, suffix=".png" )
    file_name = utils.get_count_str( len(files) ) + ".png"
    file_path = os.path.join( spec_dir, file_name )

    utils.makeSpecFromWav( wav_path, file_path, spec_params["fftWinSize"], spec_params["fftOverlap"], spec_params["maxFreq"], 
        spec_params["timeWin"])

    return file_path


def randomizeDataRows(dataframe, columns):
    """
    Creates a copy of the given dataframe with the rows randomly shuffled.
    """    
    num_rows = len(dataframe.index)
    row_idxs = list(range(0, num_rows))
    random.shuffle(row_idxs)
    shuffled = pd.DataFrame( [], columns=columns )
    for idx in row_idxs:
        shuffled.iloc[len(shuffled.index)] = dataframe.iloc[idx]

    return shuffled


def parse_dataset_directory(dpath, label):
    """
    Recursively parses a directory and builds a Pandas dataframe with wav and annotation
    file information
    """
    columns=['label', 'start', 'end', 'wavpath', 'specpath', 'test']
    
    files = utils.find_filenames( dpath, suffix=".wav" )
    dframe = pd.DataFrame( [], columns=columns )   
    
    for root, dirs, files in os.walk(dpath):

        for f in files:
            if f.endswith(".wav"):
                wav_path =  os.path.join( root, f)
                ann_file = utils.matching_annotation_file( root, f )
                print(wav_path)

                if ann_file == "":
                    length = utils.get_wav_length( wav_path )
                    row = [label, 0.0, length, wav_path, "", 0]
                    dframe.loc[len(dframe.index)] = row
                else:
                    ann_path = os.path.join( root, ann_file)
                    anns = utils.get_annotations( ann_path )
                    for ann in anns:
                        row = [label, ann.start, ann.end, wav_path, "",0]
                        dframe.loc[len(dframe.index)] = row      
    
    return dframe


def prepare_data(wav_path_pos, wav_path_neg, spec_image_dir_path, spec_params, data_frame_file_path):
    '''
    Parses the given directories for pos and neg data and populates the data items.

    The dataframe is saved as a csv
    '''
    if not os.path.exists( wav_path_neg ):
        print("ERROR: Negative data directory is invalid.")
        return False

    if not os.path.exists( wav_path_pos ):
        print("ERROR: Positive data directory is invalid.")
        return False    

    if not os.path.exists( spec_image_dir_path ):
        os.makedirs(spec_image_dir_path, exist_ok=True)

    # Get pos and neg data items from filesystem
    data_pos = parse_dataset_directory( wav_path_pos, "pos")
    data_neg = parse_dataset_directory( wav_path_neg, "neg")

    # Make single dataframe with all data items
    data_all = data_pos.copy()
    data_all = pd.concat( (data_all, data_neg), axis = 0, ignore_index=True)

    spec_for_wav_path = {}

    # Make one spectrogram for each wav file path and add the spec path to the dataframe
    for index, row in data_all.iterrows():
        wp = row.values[3]
        if wp in spec_for_wav_path:
            #print(wp)
            spec_path = spec_for_wav_path[wp]
            data_all.iloc[index, 4] = spec_path    
        else:
            spec_path = create_spec( wp, spec_image_dir_path, spec_params )
            data_all.iloc[index, 4] = spec_path 
            spec_for_wav_path[wp] = spec_path

    data_all.to_csv(data_frame_file_path, index=False)

    

def makeSplit(data_frame, test_set_size):
    """
    Creates two new dataframes - train and test - from the given dataframe
    
    The test dataframe is assumed to be smaller and is sampled first from the
    given dataframe. Care is taken to make sure that data items for a single wav file
    only appear in either the test or train dataframes, not both.
    """
    columns = data_frame.columns.tolist()
    
    wfiles = list(data_frame.wavpath.unique())
    random.shuffle(wfiles)
    count = 0
    
    data_test = pd.DataFrame( [], columns=columns ) 
    data_train = pd.DataFrame( [], columns=columns )

    train_flag = False
    for wp in wfiles:        
        
        # Retrieve all items with wp wav path
        rs = data_frame.loc[data_frame['wavpath'] == wp]
        for index, row in rs.iterrows():
        
            row_df = pd.DataFrame([row])
            
            if train_flag == True:
                data_train = pd.concat( [data_train, row_df], axis = 0, ignore_index=True ) 
        
            else:    
                data_test = pd.concat( [data_test, row_df],  axis = 0, ignore_index=True )                 
                count += 1
                if count >= test_set_size:
                    train_flag = True
                    break

    return data_train, data_test


def duplicateXTimes(data_frame, num_times):
    """
    Duplicates the rows in the dataframe num_times.
    
    This does not seem like a very efficient way of doing it but it will do for
    our purposes.
    """ 
    data_new = data_frame.copy()

    for i in range(1, num_times):
        data_new = pd.concat( [data_new, data_frame],  axis = 0, ignore_index=True ) 

    return data_new


def MakeRavenBinaryDatasetSplit( data_file, data_new_file, rand_seed, test_set_size, spec_params, class_repetitions = {"pos" : 1, "neg" : 1}, transform = None ):        
    """
    Makes a split Creates the spectrogram for a wav file and puts it in spec_dir.
    The spectrogram file name is made by counting the number of existing ".png"
    files in the directory and making a string from this count.

    The spectrogram file name therefore doesn't have any relationship to the name
    of the wav file from which it is computed. This is because it is possible that wav
    files with the same name may appear in different directories. 
    """    
    # Load data file
    df = pd.read_csv( data_file )  

    # balance dataset
    df = utils.balance_dataset(df)
   
    # First get pos and neg rows as separate dataframes
    pos = df.loc[df['label'] == "pos"]
    neg = df.loc[df['label'] == "neg"]

    random.seed(rand_seed)
    
    # Make a split between training and test with equal number of pos and neg in each
    data_pos_train, data_pos_test = makeSplit(pos, test_set_size / 2)
    data_neg_train, data_neg_test = makeSplit(neg, test_set_size / 2)

    # Check for bad train-test construction
    if len(data_pos_train.index) == 0:
         print("ERROR: The number of positive training examples is 0.")
         return (None, None)   
    if len(data_neg_train.index) == 0:
         print("ERROR: The number of negative training examples is 0.")
         return (None, None)   
    if len(data_pos_test.index) == 0:
         print("ERROR: The number of positive test examples is 0.")
         return (None, None) 
    if len(data_neg_test.index) == 0:
         print("ERROR: The number of negative test examples is 0.")
         return (None, None) 

    # Make train and test dataframes
    data_train = pd.concat([data_pos_train, data_neg_train],  axis = 0, ignore_index=True)
    data_test = pd.concat([data_pos_test, data_neg_test],  axis = 0, ignore_index=True)

    # Update the 'test' column
    data_test.loc[data_test["test"] == 0, "test"] = 1

    # Make a new dataframe which contains both train and test
    data_all = data_train.copy()    
    data_all = pd.concat( [data_train, data_test],  axis = 0, ignore_index=True )     

    # Save the data as csv
    data_all.to_csv(data_new_file, index=False)

    train_ds = RavenBinaryDataset( data_train, spec_params, rand_seed, transform )
    test_ds = RavenBinaryDataset( data_test, spec_params, rand_seed, transform )

    # Return the datasets 
    return train_ds, test_ds










