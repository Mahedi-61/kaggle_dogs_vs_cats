"""
Author     : Md. Mahedi Hasan
Date       : 2017-10-19
Project    : kaggle_dogs_vs_cats
Description: this file contains code for handling model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.callbacks import (EarlyStopping,
                             Callback,
                             ModelCheckpoint,
                             ReduceLROnPlateau)



#Train model architecture and weight
#for cloud machine remove (.)
MODEL_MOUNT_PATH = "./model/"
MODEL_SAVE_PATH = "./output/"

MODEL_NAME = "my_train_resnet101_model.json"
MODEL_WEIGHT = "my_train_resnet101_weight.h5"


def save_model(model):
    print("saving model.....")
    
    json_string = model.to_json()
    open(MODEL_SAVE_PATH + MODEL_NAME, 'w').write(json_string)




def read_model():
    print("reading stored model architecute and weight")
    
    json_string = open(MODEL_MOUNT_PATH + MODEL_NAME).read()

    model = model_from_json(json_string, {"Scale" : Scale})
    model.load_weights(MODEL_MOUNT_PATH + MODEL_WEIGHT)

    return model




class LossHistory(Callback):
    def on_train_begin(self, batch, logs = {}):
        self.losses = []
        self.val_losses = []
        

    def on_epoch_end(self, batch, logs = {}):
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))




def set_early_stopping():
    return EarlyStopping(monitor = "val_loss",
                               patience = 6,
                               mode = "auto",
                               verbose = 2)




def set_model_checkpoint():
    return ModelCheckpoint(MODEL_SAVE_PATH + MODEL_WEIGHT,
                monitor = 'val_loss',
                verbose = 2,
                save_best_only = True,
                save_weights_only = True,
                mode = 'auto',
                period = 2)





def set_reduce_lr():
    return ReduceLROnPlateau(monitor='val_loss',
                             factor = 0.1,
                             patience = 4,
                             min_lr = 1e-6)






def show_loss_function(loss, val_loss, nb_epochs):
    plt.xlabel("Epochs ------>")
    plt.ylabel("Loss -------->")
    plt.title("Loss function")
    plt.plot(loss, "blue", label = "Training Loss")
    plt.plot(val_loss, "green", label = "Validation Loss")
    plt.xticks(range(0, nb_epochs)[0::2])
    plt.legend()
    plt.show()




    
