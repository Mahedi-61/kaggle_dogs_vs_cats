Kaggle: Dogs vs Cats Kernel Redux Edition
------------------------------------------------------------

This repository consists my simple solution for the kaggle https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition challenge<br/>
This solution scored a logloss of **0.06499** on the public leaderboard

Steps for better accuracy
--------------------------
-> Several keras pretrained models like Resnet-50, Resnet-102, InceptionV3 are used with finetunning. <br/>
-> Ensembling different models increase model accuracy significantly. <br/>
-> Aggresive Dropout on Dense layers has been used to reduce overfitting. <br/>
-> Images are being pre-processed to and reduced to size (224, 224, 3). <br/>
-> diffenrent data augmentation like sheering, zooming,  horizontal-vertical shifts and flips are done. <br/>
-> PReLU activation gives best result <br/>

