# Layers-Adaptive-NN-in-Numpy
The scripts here is an implementaion of an automatic 'Layers-Adaptive-NN' in Numpy, with minibatch, regualrization and Adam algorithm.   

There are two dataset using here for this algorithm. The first one is the cat detetion, which you can find the training and testing data from the 'train_catvnoncat.h5' and test_catvnoncat.h5 directly. On the other hand, the second dataset is a typical scatter points classification problem. The '_data_load.py_' and '_dataload_Moon.py_' are the scripts to load them, respectively. Furthermore, a function helping expand the training dataset has also been implemented, which is called '_expand.py_'.

For the cat-detection, the '**_week4_Deep_network.py_**' is the main file to help train the model. From line 425-445 are where you can tune the hyperparameters. You can just build a NN as deeper as you want, as long as you create a corresponding n_h_?, and put into the [__layers_dims__] vector, where holds the information of the hidden layers and the hidden units inside. Besides, you can also try to tune the hyparameters from line 438-445 to influence the model's profermance. The output of the model training will be stored into the <well_trained_parameters_NN_cat.txt>, and the corresponding testing file is '**_week4_Deep_network_TEST.py_**'.  

The entire scripts is executable. And if you wanna run the scripts, please leave the relative path like what I left here (let the folder 'original_image' be outside of the folder 'layer_adaptive_NN'). Since the entire algorithm was built basing on the Numpy, so there is no extra need to configure the operating environment. 

Enjoy!
