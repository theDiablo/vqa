# Visual Question Answering using Deep Learning techniques.

This project uses Keras to train a variety of **Feedforward** and **Recurrent Neural Networks** for the task of Visual Question Answering. It is designed to work with the [VQA](http://visualqa.org) dataset. 

Models Implemented:

|BOW+CNN Model  |  LSTM + CNN Model |
|--------------------------------------|-------------------------| 
| <img src="https://raw.githubusercontent.com/kshitiz38/vqa/master/model_1.jpg" alt="alt text" width="400" height=""> | <img src="https://raw.githubusercontent.com/kshitiz38/vqa/master/lstm_encoder.jpg" alt="alt text" width="300" height="whatever"> |


##Requirements
1. [Keras 0.20](http://keras.io/)
2. [scikit-learn 0.16](http://scikit-learn.org/)
3. Nvidia CUDA 7.5 (optional, for GPU acceleration)

Tested with Python 2.7 on Ubuntu 16.04.

**Notes**:

1. Keras needs the latest Tensorflow, which in turn needs Numpy/Scipy. 

##Installation Guide
Follow instructions in the Readme file of each directory to download all dependencies.

##Using Pre-trained models


##The Numbers
Performance on the **validation set** of the [VQA Challenge](http://visualqa.org/challenge.html):

| Model     		   | val           |
| ---------------------|:-------------:|
| BOW+CNN              | 48.46%		   |
| LSTM-Language only   | 44.17%        | 
| LSTM+CNN             | 51.63%        | 

Note: For validation set, the model was trained on the training set.

There is a **lot** of scope for hyperparameter tuning here

##Get Started
Have a look at the `demo_prefeat_lstm` script in the `scripts` folder. Also, have a look at the readme present in each of the folders.

##Feedback
All kind of feedback (code style, bugs, comments etc.) is welcome. Please open an issue on this repo instead of mailing me, since it helps me keep track of things better.

##License
MIT