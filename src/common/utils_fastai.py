import os

from fastai import *
from fastai.vision.all import *

import augmentation

from common.utils import *

# contains functions shared between train, inference and interpret scriptws
# using fastai framework



def get_learner(dataloaders, pkl=None, architecture='resnet32'):
    """
    gets a fastai learner, either from a saved pkl file or if that's not available then downloaded
    :param pkl: string; path to a saved learner
    :param architecture: string; an available pretrained network
    :return:
    """

    # allow full relative path or filename only in the trained_models folder
    if not os.path.exists(pkl):
        pkl = os.path.join('training/output/trained_models', pkl)

    if pkl and os.path.exists(pkl):
        mylog('Loading learner from pkl')
        #learn = cnn_learner(dataloader, resnet34, metrics=error_rate)
        learn = load_learner(pkl)

        # the following seemed to be necessary. It looks like that when a learner is exported
        # the dls property is set to an empty dataloaders
        #https://github.com/fastai/fastai/blob/f633356359a29f8d869ce36659f7aa25660e946a/fastai/learner.py  # L533
        learn.dls = dataloaders
    else:
        mylog('downloading learner')
        learn = cnn_learner(dataloaders, resnet34, metrics=error_rate)

    return learn




def get_dataset(dataset_csv, image_folder='training/dataset_files/png/', start=0, end=1, priority_only=False):
    """
    Creates a dataframe from a csv and removes rows where the image file doesn't exist
    and subsets specified rows
    :param dataset_csv:
    :param image_folder:
    :param start:
    :param end:
    :param priority_only boolean; whether to only include examples from 'priority sites'. This is used for evaluation on cotton
    :return:
    """
    mylog(f'reading dataset csv: {dataset_csv}')
    print(f'working dir: {os.getcwd()}')
    df = pd.read_csv(dataset_csv)

    mylog(f'before filters, dataframe has {len(df)} rows')
    #mylog(df['label_text'].value_counts())

    if priority_only:
        #TODO there are some nan values in that column, we need to fix the script that makes that csv
        mylog('including only priority sites')
        df = df[df['is_priority_site'] == True].reset_index(drop=True)

    # exclude images that are in the dataset csv but don't actually exist

    #debug_path_to_test = os.path.join(image_folder, df.segment_png[0])
    #print(f'checking files such as {debug_path_to_test}')
    #print(f'os.path.exists("{debug_path_to_test}"): {os.path.exists(debug_path_to_test)}')

    if not os.path.exists(image_folder):
        raise Exception(f'dataset image folder does not exist: {image_folder}')


    has_file = [os.path.exists(os.path.join(image_folder, fn)) for fn in list(df.segment_png)]
    mylog(str(len(has_file) - sum(has_file)) + ' of ' + str(len(has_file)) + ' PNGs missing, will be removed from the df' )
    df = df[has_file].reset_index(drop=True)

    df = make_subset(df, start, end)
    mylog(f'after filters, dataframe has {len(df)} rows')
    #mylog(df['label_text'].value_counts())
    return df


def make_subset(df, start, end):
   """
   for each class, and for each of train/test, get only the amount between start and end
   where start is in the range [0,1) and end is > start and in the range (0,1)
   :param df:
   :param start:
   :param end:
   :return:
   """

   df = df.sample(frac=1, random_state=1).reset_index(drop=True)

   include = np.zeros((df.shape[0]), dtype=bool)

   labels = df['label_text'].unique()
   types = df['is_validation'].unique()

   for l in labels:
       for t in types:

           # todo: check if there are enough examples for each of these

           is_match = (df['label_text'] == l) & (df['is_validation'] == t)
           row_nums = df.index[is_match]

           subset_start = math.floor(len(row_nums) * start)
           subset_end= math.ceil(len(row_nums) * end)

           included_row_nums = list(row_nums[range(subset_start, subset_end)])
           include[included_row_nums] = True

           mylog(f'class {l} with validation {t} has {len(row_nums)} examples before subsetting')


   mylog(f'{np.sum(include)} of {df.shape[0]} examples will be included')
   subset_df = df.iloc[include].reset_index(drop=True)
   return subset_df


def get_dataloaders_old(df, bs=16, img_folder='training/dataset_files/png'):
    # https://docs.fast.ai/vision.data.html#ImageDataLoaders.from_df
    dls =ImageDataLoaders.from_df(df=df,
                                  valid_col='is_validation',  #1 to use for validation, 0 for training
                                  seed=1,  # for reproducing the random stuff
                                  fn_col='segment_png',  # column numbers of the image filenames
                                  folder=img_folder,
                                  suff='',
                                  label_col='label_text', #column number of the labels
                                  label_delim=None, y_block=None, item_tfms=None, batch_tfms=None,
                                  bs=bs, val_bs=None, shuffle_train=True, device=None)

    return dls


def get_dataloaders2(df, bs=16, img_folder='training/dataset_files/png/', use_aug=False):

    transforms = [
        # original spectrograms are 224 tall varible width (at least 224). This will randomly crop
        RandomCrop(224, order=1),

        # this selects a random segment from 'other' or 'nothing' and overlays (blends) it.
        # augmentation.OverlayTransform(order=2)

        # this warps. We want to keep the warping very slight for our purposes.
        # Warp(magnitude=0.05)

    ]

    if use_aug:
        print("using augmentation")
        transforms = transforms + [augmentation.OverlayTransform(order=2)]
    else:
        print("not using augmentation")

    # https://docs.fast.ai/vision.data.html#ImageDataLoaders.from_df
    # https://towardsdatascience.com/advanced-dataloaders-with-fastai2-ecea62a7d97e
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        getters=[ColReader('segment_png', pref=os.path.join(img_folder, '')),  # image input
                 ColReader('label_text')],  # label 3
        splitter=ColSplitter(col='is_validation'),  # train/validation split
        item_tfms=transforms)

    #dls = dblock.dataloaders(df, bs=bs, num_workers=0)

    if sys.platform == 'darwin':
        # workers more than 0 seems to cause (at least my) mac to freak out with
        # [W ParallelNative.cpp:206] Warning: Cannot set number of intraop threads after parallel work has started ...
        dls = dblock.dataloaders(df, bs=bs, num_workers=0)
    else:
        dls = dblock.dataloaders(df, bs=bs)

    return dls

