
import os

from PIL import Image as im
from PIL import ImageFont, ImageDraw


# Need to get the train params


# Read the training log
#f = open('the-zen-of-python.txt','r')
#log = open( train_params["log"], 'w')



def makeMosaicFromImages(image_list):
    
    '''
    Makes a (3 x n) mosaic image from a list of image patches of size 256 x 128.    
    '''    

    num_images = len(image_list)

    margin = 30
    images_vert = int(num_images / 3)
    h = images_vert * 128 + (images_vert-1) * margin + margin
    w = 3 * 256 + 2 * margin

    font = ImageFont.truetype('notebook_images/ayar.ttf', 16) 
    new_image = im.new( "L", (w, h))
    draw = ImageDraw.Draw(new_image) 

    for i in range(0, num_images):
        y = int(i / 3) * (128 + margin) + margin
        x = (i % 3) * (256 + margin)
        new_image.paste(image_list[i][1], (x, y))

        draw.text( (x+120, y-25), str(image_list[i][0]), fill ="white", font = font, align ="left") 

    new_image.save('images.png')



def get_all_spectrograms(log_file):
    
    # Open training file
    f = open(log_file,'r')

    item_flag = False
    spec_set = set()
    image_hash = {}
    
    for line in f:
        line = line.strip()
        if line.startswith("TRAIN SET") or line.startswith("TEST SET"):
            item_flag = True
            continue  
        if len(line)==0:
            item_flag = False
        if item_flag == True:
            toks = line.split(",")
            spec_set.add( toks[2].strip() )

    for s in spec_set:
        img = im.open(s)
        image_hash[s] = img

    return image_hash



def get_cropped_from_spectrogram_image(img, offset, pixels_per_sec):

    patch_pixels_width = 256
    patch_pixels_height = 128

    pos = int(offset * pixels_per_sec)

    img_ = img.crop( (pos, 0, pos + patch_pixels_width, patch_pixels_height) )
    
    return img_



def do_analysis(train_params, spec_params, output_path):
    """
    Reads the training log file and makes a combined image of all the incorrect train and test images
    """

    log_file = train_params["log"]

    # This loads all spectrograms referred to in the log file, into memory
    image_hash = get_all_spectrograms(log_file)

    # Open training file
    f = open( log_file, 'r')

    test_flag = False
    test_list = []
    train_list = []

    # Parse the log file
    for line in f:
        line = line.strip()
        if line.startswith("TEST SET"):
            test_list = []
            test_flag = True
            continue  
        if line.startswith("TRAIN SET"):
            train_list = []
            train_flag = True                    
            continue
        if len(line)==0:
            test_flag = False
            train_flag = False
        if test_flag == True:
            toks = line.split(",")
            label = int(toks[0])
            pred = int(toks[1])
            if not label==pred:
                test_list.append( (int(toks[0]), int(toks[1]), toks[2].strip(), float(toks[3]), toks[4].strip()) )
        if train_flag == True:
            toks = line.split(",")
            label = int(toks[0])
            pred = int(toks[1])
            if not label==pred:                        
                train_list.append( (int(toks[0]), int(toks[1]), toks[2].strip(), float(toks[3]), toks[4].strip()) )

    images_len = len(train_list) + len(test_list)

    margin = 30
    offset_from_top = 30
    h = images_len * 128 + (images_len-1) * margin + margin
    w = 256 + 2 * margin + 600

    new_image = im.new( "L", (w, h + 100 + offset_from_top))
    draw = ImageDraw.Draw(new_image) 

    font = ImageFont.truetype('notebook_images/ayar.ttf', 16)
    font2 = ImageFont.truetype('notebook_images/ayar.ttf', 24)

    patch_pixels_width = 256
    time_win = float(spec_params["timeWin"])
    pix_per_sec = float(patch_pixels_width) / time_win


    draw.text( (margin, 20), "Training ", fill ="white", font = font2, align ="left") 

    y = 0
    for i in range(0, len(train_list)):
        y = i * (128 + margin) + margin + offset_from_top
        y_cursor = y
        x = margin
        label = str(train_list[i][0])
        pred = str(train_list[i][1])         
        ipath = train_list[i][2]
        wpath = train_list[i][4] 
        offset = float( train_list[i][3] )
        img = get_cropped_from_spectrogram_image( image_hash[ipath], offset, pix_per_sec)
        new_image.paste( img, (x, y))
        draw.text( (x+280, y + 25), "Label: " + label, fill ="white", font = font, align ="left") 
        draw.text( (x+280, y + 50), "Prediction: " + pred, fill ="white", font = font, align ="left")         
        draw.text( (x+280, y + 75), wpath, fill ="white", font = font, align ="left") 

    y_cursor = y + 30

    draw.text( (margin, y_cursor + 150), "Test", fill ="white", font = font2, align ="left")

    for i in range(0, len(test_list)):
        y = (i+1) * (128 + margin) + margin + y_cursor
        x = margin
        label = str(test_list[i][0])
        pred = str(test_list[i][1])        
        ipath = test_list[i][2]
        wpath = test_list[i][4] 
        offset = float( test_list[i][3] )
        img = get_cropped_from_spectrogram_image( image_hash[ipath], offset, pix_per_sec)
        new_image.paste( img, (x, y))
        draw.text( (x+280, y + 25), "Label: " + label, fill ="white", font = font, align ="left") 
        draw.text( (x+280, y + 50), "Prediction: " + pred, fill ="white", font = font, align ="left")                 
        draw.text( (x+280, y + 75), wpath, fill ="white", font = font, align ="left") 

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    new_image.save(output_path)