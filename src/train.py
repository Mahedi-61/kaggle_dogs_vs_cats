"""
Author : Md. Mahedi Hasan
Date   : 2017-10-19
Project: kaggle_dogs_vs_cats
Description: train a different model for kaggle dogs_vs_cats dataset
"""

import my_models
import utils

import numpy as np
import image_preprocessing
from keras.optimizers import SGD



#Network & Training Parameter 
nb_epochs = 4
batch_size = 32


# Preprocessing
train_data, train_labels = image_preprocessing.load_train_data()
print("Trian Data: ", train_data.shape)
print("Train Lable: ", len(train_labels))



# for store train data matrix
"""
path = "/data/train_data_1.dat"
train_data = image_preprocessing.restore_data(path)

path = "/data/train_data_2.dat"
train_data_2 = image_preprocessing.restore_data(path)

train_data = np.vstack((train_data, train_data_2))
del train_data_2


path = "/data/train_labels.dat"
train_labels = image_preprocessing.restore_data(path)
"""


# Constructing Resnet-152 Architecture 
model = my_models.model_resnet101()


optimizers = SGD(lr = 1e-3,
                  momentum = 0.9,
                  nesterov = True)

objective = "categorical_crossentropy"

model.compile(optimizer = optimizers,
              loss =     objective,
              metrics = ['accuracy'])




# Training and Evaluating 
history = utils.LossHistory()
early_stopping = utils.set_early_stopping()
model_cp = utils.set_model_checkpoint()
reduce_lr = utils.set_reduce_lr()


model.fit(train_data,
          train_labels,
          batch_size = batch_size,
          shuffle = True,
          epochs = nb_epochs,
          callbacks = [history, early_stopping, model_cp, reduce_lr],
          verbose = 2,
          validation_split=0.2)


my_models.save_model(model)


# Drawing historical loss function
"""
loss = history.losses
val_loss = history.val_losses
utils.show_loss_function(loss, val_loss, num_epochs)
"""




























