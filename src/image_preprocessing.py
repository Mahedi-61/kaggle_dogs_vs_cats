"""
Author : Md. Mahedi Hasan
Date   : 2017-10-19
Project: kaggle_dogs_vs_cats
Description: this file is for image preprocessing
"""

import numpy as np
import PIL
import pickle
import os
from PIL import Image
import cv2


###################### Training & Testing Parameters ######################
#for local machine
#TRAIN_DIR = r"../input/train/"
#TEST_DIR = r"../input/test/"

#for cloud
TRAIN_DIR = r"/input/input/train/"
TEST_DIR = r"/input/input/test/"


IMG_SIZE = 200
IMG_CHANNEL = 3
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_CHANNEL)

NUM_CLASSES = 2
NUM_TRAIN_SAMPLES = 16000
NUM_TEST_SAMPLES =  12500




def preprocess_input(x):
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2.0)
    return x



def resize_image(img, size):
    #Pillow return images size as (w, h)
    width, height = img.size 

    if(width > height):
        new_width = size
        new_height = int(size * (height / width) + 0.5)

    else:
        new_height = size
        new_width = int(size * (width / height) + 0.5)

    #resize for keeping aspect ratio
    img_res = img.resize((new_width, new_height), resample = PIL.Image.BICUBIC)

    #Pad the borders to create a square image
    img_pad = Image.new("RGB", (size, size), (128, 128, 128))
    ulc = ((size - new_width) // 2, (size - new_height) // 2)
    img_pad.paste(img_res, ulc)

    return img_pad



def normalization(X):
    #keras weights are supposed images are in BGR mode
 
    # 'RGB'->'BGR'
    X = X.astype(np.float32)
    X = X[..., ::-1]

    #The mean pixel values are taken from the VGG authors
    vgg_mean = [103.939, 116.779, 123.68]

    # Zero-center by mean pixel
    for c in range(3):
        X[:, :, c] -=  vgg_mean[c]

    return X




def luminance_norm_image(img):
    
    #Normalizes luminance to (mean,std) = (0,1), and
    #applies a [1%, 99%] contrast stretch 
    img_y, img_b, img_r = img.convert('YCbCr').split()

    img_y_np = np.asarray(img_y).astype(float)
    
    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()
    
    
    scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                    np.abs(np.percentile(img_y_np, 99.0))])


    img_y_np = img_y_np / scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    img_y_np = (img_y_np + 1.0) / 2.0

    img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)

    img_y = Image.fromarray(img_y_np)
    img_ybr = Image.merge("YCbCr", (img_y, img_b, img_r))
    img_nrm = img_ybr.convert("RGB")
    
    return img_nrm






def prep_images(images):
    data = np.ndarray((len(images), IMG_SIZE, IMG_SIZE, IMG_CHANNEL),
                      dtype = np.uint8)

    #Pillow returns numpy array of (width, height,channel(RGB))
    for i, image_file in enumerate(images):
        img = Image.open(image_file)
        img = resize_image(img, IMG_SIZE)
        img = luminance_norm_image(img)
        img_px = np.array(img)  #convert PIL image to numpy array

        data[i] = img_px

        if (i % 1000 == 0):
            print("Processing {0} images".format(i))
    return data






def process_train_images():
    
    #train index start from 0 like cat.0.jpg and dog.0.jpg
    cat_images = [TRAIN_DIR + "cat." + str(i) +
                  ".jpg" for i in range(0, int(NUM_TRAIN_SAMPLES/2))]

    #first col(probability of being cat) and second col (prob. of being dog)
    cat_images_label = [[1, 0] for i in range(0,  int(NUM_TRAIN_SAMPLES/2))]

    
    dog_images = [TRAIN_DIR + "dog." + str(i) +
                  ".jpg" for i in range(0, int(NUM_TRAIN_SAMPLES/2))]

    dog_images_label = [[0, 1] for i in range(0,  int(NUM_TRAIN_SAMPLES/2))]

    train_images = []
    train_labels = []
    for i in range(0,  int(NUM_TRAIN_SAMPLES // 2)):    
        train_images.append(cat_images[i])
        train_images.append(dog_images[i])

        train_labels.append(cat_images_label[i])
        train_labels.append(dog_images_label[i])

    return prep_images(train_images), train_labels 





def process_test_images():
    #test index start from 1 
    test_images = [TEST_DIR + str(i) +
                   ".jpg" for i in range(1,  NUM_TEST_SAMPLES + 1)]

    return prep_images(test_images)





def load_train_data():
    print("Start Preprocessing train data")

    train_data, train_labels = process_train_images() # np array
    train_data = normalization(train_data)            # python list

    return train_data, train_labels  # np array, python list





def load_test_data():
    print("Start Preprocessing test data")

    test_data =  process_test_images()
    test_data =  normalization(test_data)
    return test_data



# saving and restoring data ....
def save_data(data, path):
    file = open(path, "wb")

    #serializing a Python object
    pickle.dump(data, file)  
    file.close()




def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, "rb")
        
        #de-serializing a Python object
        data = pickle.load(file)
    return data



def save_train_data():
    print("Start Preprocessing train data")
    train_data, train_labels = process_train_images()
    train_data = normalization(train_data)

    path = "/output/train_data_1.dat"
    save_data(train_data[:7000], path)

    path = "/output/train_data_2.dat"
    save_data(train_data[7000:14000], path)

    path = "/output/train_labels.dat"
    save_data(train_labels, path)
    
    del train_data, train_labels
    



process_train_images()



