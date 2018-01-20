"""
Author : Md. Mahedi Hasan
Date   : 2017-10-19
Project: kaggle_dogs_vs_cats
Description: this file contains training model
"""

import image_preprocessing

from model.resnet_50 import ResNet50
from model.resnet_101 import ResNet101
from model.resnet_152 import ResNet152
from model.inception_v3 import InceptionV3
from model.inception_v4 import InceptionV4

from keras.models import Model, model_from_json
from keras.layers import (Flatten, Input, Dense, Dropout, AveragePooling2D)


RESNET50_WEIGHTS = "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
RESNET101_WEIGHTS = "resnet101_weights_tf.h5"
RESNET152_WEIGHTS = "resnet152_weights_tf.h5"
INCEPTION_V3_WEIGHTS = "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
INCEPTION_V4_WEIGHTS = "inception_v4_weights_tf_dim_ordering_tf_kernels_notop.h5"

#Network & Training Parameter 
WEIGHTS_PATH = r"../weight/"          #for local machine
#WEIGHTS_PATH = r"/weight/"           #for cloud

img_shape = image_preprocessing.IMG_SHAPE
nb_class = image_preprocessing.NUM_CLASSES



def model_resnet50():
    base_model = ResNet50(
                 include_top = False,
                 input_shape = img_shape,
                 weights_path = WEIGHTS_PATH + RESNET50_WEIGHTS)

    print("Resnet-50 model without top is loaded")
    base_model.trainable = False

    img_input = Input(shape = img_shape)
    x = base_model(img_input)
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)

    #building a FC model (for 2 classes) on top of the resenet-50 model
    x = Dense(nb_class, activation='softmax', name='fc1')(x)

    model = Model(img_input, x)
    return model




def model_resnet101():
    base_model = ResNet101(
                 include_top = False,
                 input_shape = img_shape,
                 weights_path = WEIGHTS_PATH + RESNET101_WEIGHTS)

    print("Resnet-101 model without top is loaded")
    base_model.trainable = False

    img_input = Input(shape = img_shape)
    x = base_model(img_input)
    
    #img_shape should be atleast (197, 197, 3) for using (7, 7) pooling window
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)

    #building a FC  model (for 2 classes) on top of the resenet-101 model
    x = Dense(nb_class, activation='softmax', name='fc1')(x)

    model = Model(img_input, x)
    return model





def model_resnet152():
    base_model = ResNet152(
                 include_top = False,
                 input_shape = img_shape,
                 weights_path = WEIGHTS_PATH + RESNET152_WEIGHTS)

    print("Resnet-152 model without top is loaded")
    base_model.trainable = False

    img_input = Input(shape = img_shape)
    x = base_model(img_input)

    #img_shape should be atleast (197, 197, 3) for using (7, 7) pooling window
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)

    #building a FC  model (for 2 classes) on top of the resenet model
    x = Dense(nb_class,  activation='softmax', name='fc1')(x)

    model = Model(img_input, x)
    return model

    

def model_inception_v3():
    base_model = InceptionV3(
                 include_top = False,
                 input_shape = img_shape,
                 weights_path = WEIGHTS_PATH + INCEPTION_V3_WEIGHTS,
                 pooling = "avg")

    print("Inception v3 model without top is loaded")
    base_model.trainable = False

    img_input = Input(shape = img_shape)
    x = base_model(img_input)

    #building a FC  model (for 2 classes) on top of the inception model
    x = Dense(nb_class,  activation='softmax', name='fc1')(x)

    model = Model(img_input, x)
    return model




def model_inception_v4():
    base_model = InceptionV4(
                 include_top = False,
                 input_shape = img_shape,
                 weights_path = WEIGHTS_PATH + INCEPTION_V4_WEIGHTS)

    print("Inception v4 model without top is loaded")
    base_model.trainable = False

    img_input = Input(shape = img_shape)
    x = base_model(img_input)
    x = AveragePooling2D((4, 4), padding='valid')(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)

    #building a FC  model (for 1 classes) on top of the inception model
    x = Dense(nb_class,  activation='softmax', name='fc1')(x)

    model = Model(img_input, x)
    return model


