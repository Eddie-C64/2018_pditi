import os
import cv2
import sys
import random
import warnings
import numpy as np
import pandas as pd 
import matplotlib
from matplotlib import pyplot as plt
import math
import re
import time

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
 
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
 
import tensorflow as tf
 
from config import Config
import utils
import model as modellib
import visualize
from model import log
 
from subprocess import check_output
 

 
# Root directory of the project
ROOT_DIR = os.getcwd()
 
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
 
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

###############################################################################
############################# CONFIG ##########################################
###############################################################################      
    

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    LEARNING_RATE = 0.001

    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (56, 56)

    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    bs = 2

    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 100  

    NUM_CLASSES = 1 + 1

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    TRAIN_ROIS_PER_IMAGE = 128

    MAX_GT_INSTANCES = 300

    DETECTION_MAX_INSTANCES = 300


    
config = ShapesConfig()
config.display()

###############################################################################
############################# IMPORT DATA #####################################
###############################################################################       

 
 
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
   
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
 
#image_file = "/home/ubuntu/keras/examples/Mask_RCNN/Kaggle/stage1_train/{}/images/{}.png".format(image_id,im$
#mask_file = "/home/ubuntu/keras/examples/Mask_RCNN/Kaggle/stage1_train/{}/masks/*.png".format(image_id)

class ShapesDataset(utils.Dataset):
       
        def start(self, image_ids):
            global image_id
            i = 0
            for ax_index, image_id in enumerate(image_ids):
                self.add_class("shapes", 1, "nuclei")
                shapes = "nuclei"
                self.add_image("shapes", image_id=i, path = None, shapes = shapes) 
                print(image_id)
                print(i)
                image_id = i
                self.load_image(image_id)
                self.load_mask(image_id)
                self.image_reference(image_id)        
                i = i + 1
 
        def read_image_labels(self, image_id):
            image_file = "/home/ubuntu/keras/examples/Mask_RCNN/Kaggle/stage1_train/{}/images/{}.png".format(image_id,image_id)
            mask_file = "/home/ubuntu/keras/examples/Mask_RCNN/Kaggle/stage1_train/{}/masks/*.png".format(image_id)
            image = skimage.io.imread(image_file)[:,:,:3]
            masks = skimage.io.imread_collection(mask_file).concatenate()   
            height, width, _ = image.shape
            num_masks = masks.shape[0]
            labels = np.zeros((height, width), np.uint16)
            for index in range(0, num_masks): 
                labels[masks[index] > 0] = index + 1
            return image, labels
           
 

        def load_image(self,image_id):
            info = self.image_info[image_id]
            image_info = self.image_info[image_id]
            image_ids = check_output(["ls", "Kaggle/stage1_train/"]).decode("utf8").split()
            image_id = image_ids[image_id]
            image, labels = self.read_image_labels(image_id)
            IMG_WIDTH = 128
            IMG_HEIGHT = 128
            IMG_CHANNELS = 3
            img = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            return img
            
 
        def load_mask(self,image_id):
            info = self.image_info[image_id]
            shapes = info['shapes']
            image_ids = check_output(["ls", "Kaggle/stage1_train/"]).decode("utf8").split()
            image_id = image_ids[image_id]
            image, labels = self.read_image_labels(image_id)
            IMG_WIDTH = 128
            IMG_HEIGHT = 128
            mask = np.expand_dims(resize(labels, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                      preserve_range=True), axis=-1)
            class_ids = []
            class_id = 1
            class_ids.append(class_id)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
 
        def image_reference(self, image_id):
            info = self.image_info[image_id]
            if info["source"] == "shapes":
                return info["shapes"]
            else:
                super(self.__class__).image_reference(self, image_id)
 
###############################################################################
############################# TRAINING ########################################
###############################################################################           

from sklearn.cross_validation import train_test_split
data = check_output(["ls", "Kaggle/stage1_train/"]).decode("utf8").split()
x_train ,x_test = train_test_split(data,test_size=0.25) 

data_train = x_train 
image_ids_train = data_train
print(len(image_ids_train))
 
dataset_train = ShapesDataset()
dataset_train.start(image_ids_train)
dataset_train.prepare()
 

data_val = x_test
image_ids_val = data_val
print(len(image_ids_val))
 
dataset_val = ShapesDataset()
dataset_val.start(image_ids_val)
dataset_val.prepare()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
 
# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last
 
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)
 

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=10,
            layers="all")
 
class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
inference_config = InferenceConfig()
 
# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)
 
# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]
 
# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)
 
###############################################################################
############################# IMPORT TEST DATA ################################
###############################################################################
 
print('Testing')
data_test = check_output(["ls", "Kaggle/stage1_test/"]).decode("utf8").split()
image_ids_test = data_test
print(len(image_ids_test))
 
dataset_test = ShapesDataset()
dataset_test.start(image_ids_test)
dataset_test.prepare()

################################################################################
############################# OUTPUT FILE ######################################
################################################################################

from skimage.morphology import label # label regions
def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths
 
def prob_to_rles(x, cut_off = 0.5):
    lab_img = label(x>cut_off)
    if lab_img.max()<1:
        lab_img[0,0] = 1 # ensure at least one prediction per image
    for i in range(1, lab_img.max()+1):
        yield rle_encoding(lab_img==i)
 
test_path= 'Kaggle/stage1_test/'
test_ids = check_output(["ls", "Kaggle/stage1_test/"]).decode("utf8").split()
print(test_ids) 
x = 0 
rles = [] 
new_test_ids = [] 
 
for image_id in dataset_test.image_ids:
 
    image_id = test_ids[image_id]
 
    image = cv2.imread(os.path.join(test_path, image_id, 'images', image_id) + '.png', 1)
    resized_image, window, scale, padding = utils.resize_image(
    image,
    min_dim=config.IMAGE_MIN_DIM,
    max_dim=config.IMAGE_MAX_DIM,
    padding=config.IMAGE_PADDING)
 
    # Run object detection
    results = model.detect([resized_image], verbose=1)
    r = results[0]    
 
    image_id = test_ids[x]
    x = x + 1     
    for index in range(r['masks'].shape[-1]):
        rle = list(prob_to_rles(np.squeeze(r['masks'][:, :, index])))
        rles.extend(rle)
        new_test_ids.extend([image_id] * len(rle))
 
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('predictions.csv', index=False)
