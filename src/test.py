"""
Author : Md. Mahedi Hasan
Date   : 2017-10-19
Project: kaggle_dogs_vs_cats
Description : this file is for test dataset
"""

import my_models
import resnet_101
import resnet_152
import image_preprocessing
import utils

import numpy as np
from keras.models import load_model


#for cloud remove (.)
path_1 = "/model_1/my_train_resnet50_model.h5"
path_2 = "/model_2/my_train_resnet101_model.h5"
path_3 = "/model_3/my_train_resnet152_model.h5"


path_list = [path_1, path_2, path_3]
custom_obj = [None, {"Scale" :resnet_101.Scale}, {"Scale" :resnet_152.Scale}]
result_list = []



#Preparing test images
batch_size = 32
nb_test_images = image_preprocessing.NUM_TEST_SAMPLES
test_data = image_preprocessing.saveing_test_data()


for i, path in enumerate(path_list):
    print("Loading Model_{0}.....".format(str(i)))

    model = load_model(path, custom_objects = custom_obj[i])
    #model = my_models.read_model()

    print("Start Predicting.....")
    predictions = model.predict(test_data, batch_size, verbose = 2)

    #formatting result; clipping for avoiding logloss(1, 0) danger
    result = predictions[:, 1]
    result = np.clip(result, 0.01, 0.99)
    result_list.append(result)
    
del model


#Making ensemble of this 3 model
#Give Priority 2 to ResNet50 for it's better accuracy
ens_result = np.vstack((result_list[0] * 2, result_list[1]))
ens_result = np.vstack((ens_result, result_list[2]))
ens_result = np.sum(ens_result, axis = 0) / 4


"""
print("Resnet50: ",  result_list[0])
print("Resnet101: ", result_list[1])
print("Resnet152: ", result_list[2])
print("Ensemble: ",  ens_result)

s = []
for i in range(30, 39):
    s.append("resnet50: " + str(result_1[i]) + "\nresnet101: "  + 
             str(result_2[i]) + "\nresult: " + str(result[i]))

            
#for ploting
images = test_data[30:39]
utils.plot_gallery(images, s)
"""

print("preparing for result submission")
utils.submit_result(ens_result, nb_test_images)








