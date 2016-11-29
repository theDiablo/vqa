from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Flatten, Dense, Dropout, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.embeddings import Embedding
from keras.layers.pooling import AveragePooling2D
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
import keras.models as models
import pandas as pd

import keras.layers.core as core
from scipy import misc
from PIL import Image
from keras.models import load_model
import numpy as np
import h5py

#File to read questions and answers
questionsFile="/scratch/maths/btech/mt1140594/VQA/PythonHelperTools/train2014/questions.txt"
answersFile="/scratch/maths/btech/mt1140594/VQA/PythonHelperTools/train2014/answers.txt"
image_File="/scratch/maths/btech/mt1140594/VQA/PythonHelperTools/train2014/imageData.csv"
image_index_File="/scratch/maths/btech/mt1140594/VQA/PythonHelperTools/train2014/image_ids.txt"

questionsFileVal="/scratch/maths/btech/mt1140594/VQA/PythonHelperTools/val2014/questions.txt"
answersFileVal="/scratch/maths/btech/mt1140594/VQA/PythonHelperTools/val2014/answers.txt"
image_FileVal="/scratch/maths/btech/mt1140594/VQA/PythonHelperTools/val2014/imageData.csv"
image_index_FileVal="/scratch/maths/btech/mt1140594/VQA/PythonHelperTools/val2014/image_ids.txt"

threshQ=4
threshA=14
MAX_SEQUENCE_LENGTH=10
nb_epoch = 30
batch_size = 64

print('Threshold for Questions Vocab:' +str(threshQ))
print('Threshold for Answer Vocab: '+str(threshA))
print('Max Sequence Length:' +str(MAX_SEQUENCE_LENGTH))
print('Max Epochs allowed: '+str(nb_epoch))
print('Batch Size: '+str(batch_size))

######################################### Question Vocab ###############################################

def prepareQuestionVocab( questionsFile, threshQ ):

    print('Preparing training data')
    data=(open(questionsFile)).read()
    print('Questions File found, Reading data')
    data=data.replace('\n',' ').split()

    n=len(data)
    print('Making Questions Vocabularoy')
    vocab_map={}

    for i in range(n-1):
        word=data[i]
        if vocab_map.has_key(word):
            count=vocab_map[word]
            vocab_map[word]=count+1
        else:
            vocab_map[word]=1

    vocab={}
    current=1
    for word in vocab_map :
        if vocab_map[word]>=threshQ :
            vocab[word]=current
            current=current+1

    vocab_size = len(vocab)
    print('Questions Vocabularoy done')
    print('Questions Vocab size:'+str(vocab_size))

    return [ vocab, vocab_size ]

######################################### Question Data ########################################################

def prepareQuestionData( vocab, questionsFile, questionsFileVal ):

    data=(open(questionsFile)).read().split('\n')
    print('Total number of questions in File :'+str(len(data)-1))

    Xtrain=[]
    for index in range( len(data)-1):
        question =data[index]
        question=question.split()
        row=[0]*10
        currentword=0
        for i in range(min(len(question),10)):
            if vocab.has_key(question[i]):
                value=vocab[question[i]]
                row[currentword]=value
                currentword=currentword+1
        Xtrain.append(row)

    training_question_size=len(Xtrain)
    print('Questions for training data done')


    data=(open(questionsFileVal)).read().split('\n')
    print('Total number of questions in Validation File :'+str(len(data)-1))

    Xval=[]
    for index in range( len(data)-1):
        question =data[index]
        question=question.split()
        row=[0]*10
        currentword=0
        for i in range(min(len(question),10)):
            if vocab.has_key(question[i]):
                value=vocab[question[i]]
                row[currentword]=value
                currentword=currentword+1
        Xval.append(row)

    val_question_size=len(Xval)
    print('Questions for Validation data done')

    Xtrain = np.array(Xtrain, dtype='float16')
    Xval = np.array(Xval, dtype='float16')

    return [Xtrain, Xval]

############################################ Answer Vocab ###########################################################

def prepareAnswerVocab( answersFile, threshA ):

    ans_data=(open(answersFile)).read()
    print('Answers File found, Reading data')
    ans_data=ans_data.split('\n')
    n=len(ans_data)
    print('Total number of answers in File :'+str(n-1))
    print('Making Answers Vocabularoy')

    ans_vocab_map={}
    for i in range(n-1): #should be n-1
        word=ans_data[i]
        if ans_vocab_map.has_key(word):
            count=ans_vocab_map[word]
            ans_vocab_map[word]=count+1
        else:
            ans_vocab_map[word]=1

    ans_vocab={}

    current=1
    for word in ans_vocab_map :
        if ans_vocab_map[word]>=threshA :
            ans_vocab[word]=current
            current=current+1  

    print('Answers Vocabularoy done')
    print('Answers Vocab size:'+str(len(ans_vocab)))

    return ans_vocab

########################################### Answer Data ########################3################################

def prepareAnswerData( ans_vocab, answersFile, answersFileVal ):

    ans_data=(open(answersFile)).read().split('\n')
    y_train=[]
    for index in range(len(ans_data)-1 ): #should be len(ans_data)-1
        answer =ans_data[index]
        row=[0]*len(ans_vocab)
        if ans_vocab.has_key(answer):
            value=ans_vocab[answer]
            row[value-1]=1
        y_train.append(row)

    training_answer_size=len(y_train)
    print('Answers for Training data done..')

    ans_data=(open(answersFileVal)).read().split('\n')
    y_val=[]
    for index in range(len(ans_data)-1): #len(ans_data)-1
        answer =ans_data[index]
        row=[0]*len(ans_vocab)
        if ans_vocab.has_key(answer):
            value=ans_vocab[answer]
            row[value-1]=1
        y_val.append(row)

    val_answer_size=len(y_val)
    print('Answers for Validation data done..')

    y_train = np.array(y_train, dtype='float16')
    y_val = np.array(y_val, dtype='float16')

    return [y_train, y_val]

##########################3################# Image Data ###########################################################

#Image training data
def prepareImageData( nTrainImages, nValImages, image_index_File, image_index_FileVal ):
    print('Reading training Image data')

    mean_pixel = [ 123.68, 116.779, 103.939]
        
    data=(open(image_index_File)).read().split('\n')
    print('Image Index File found, Reading data')
    print('Total number of Image index:' + str(len(data)))

    x_train=np.zeros((nTrainImages,3,224,224), dtype='float16')
    for i in range(len(data)-1): #shold be range(len(data)-1)
        index=len(data[i])
        zero=''
        for it in range(6-index):
            zero = zero + '0'
        img = Image.open('/scratch/maths/btech/mt1140594/VQA/Images/train2014/COCO_train2014_000000'+zero+data[i]+'.jpg').convert('RGB')
        img = img.resize((224,224))
        image = np.array(img)
        image = image.flatten()
        image = image.reshape((3,224,224))
        image=image.astype(np.float16)
        for c in range(3):
            image[c,:, :] = image[c,:, :] - mean_pixel[c]
        #print i
        x_train[i,:,:,:] = image

    print('Images training data done')
    print('Total number of Images :'+ str(len(x_train[0])))

    print('Reading val Image data')
    data=(open(image_index_FileVal)).read().split('\n')
    print('Image Index File found, Reading data')
    print('Total number of Image index:' + str(len(data)))

    x_val=np.zeros((nValImages,3,224,224),dtype='float16')
    for i in range(len(data)-1): #shold be range(len(data)-1)
        index=len(data[i])
        zero=''
        for it in range(6-index):
            zero = zero + '0'
        img = Image.open('/scratch/maths/btech/mt1140594/VQA/Images/val2014/COCO_val2014_000000'+zero+data[i]+'.jpg').convert('RGB')
        img = img.resize((224,224))
        image = np.array(img)
        image = image.flatten()
        image = image.reshape((3,224,224))
        image=image.astype(np.float16)
        for c in range(3):
            image[c,:, :] = image[c,:, :] - mean_pixel[c]
        
        #print i 
        x_val[i,:,:,:] = image


    print('Images val data done')
    print('Total number of Images :' +str(len(x_val[0])))

    return [x_train, x_val]


###################################### CNN Image Model #######################################################

def loadCNNModel():
    
    print('Making CNN model..')

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(0.5))

    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    weights_path = 'vgg16_weights.h5'

    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
        model.layers[k].trainable=False
    f.close()
    print('Model loaded.')

    model.add(Dense(1024, activation='relu', name='img_feature'))
    model.summary()

    return model

######################################## Text Model #######################################################

def loadTextModel(vocab_size, MAX_SEQUENCE_LENGTH):

    textmodel= Sequential()
    textmodel.add(Embedding(input_dim=(vocab_size + 1), input_length=MAX_SEQUENCE_LENGTH, output_dim=1024, mask_zero=True))
    textmodel.add(LSTM(output_dim=512, return_sequences=True))
    textmodel.add(LSTM(output_dim=512, return_sequences=False))
    textmodel.add(core.Dense(1024, activation="relu"))
    textmodel.summary()
    return textmodel

######################################### Merge Models #######################################################

def mergeModel(model, textmodel):

    merged = core.Merge([model, textmodel], mode='mul')
    return merged



######################################### FinalModel - Make changes here ############################################

[vocab, vocab_size] = prepareQuestionVocab( questionsFile, threshQ )
[Xtrain, Xval] = prepareQuestionData ( vocab, questionsFile, questionsFileVal )
ans_vocab = prepareAnswerVocab( answersFile, threshA )
[y_train, y_val] = prepareAnswerData ( ans_vocab, answersFile, answersFileVal )
#[x_train, x_val] = prepareImageData( 248349, 121512, image_index_File, image_index_FileVal )

#model = loadCNNModel()
textmodel = loadTextModel(vocab_size, MAX_SEQUENCE_LENGTH)
#merged = mergeModel(model, textmodel)

finalmodel= Sequential()
#finalmodel.add(merged)
finalmodel.add(textmodel)
#finalmodel.add(model)
finalmodel.add(core.Dense(4096, activation="relu"))
finalmodel.add(Dropout(0.5))
finalmodel.add(core.Dense(len(ans_vocab), activation="softmax"))
finalmodel.summary()

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
finalmodel.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

checkpointer = ModelCheckpoint(filepath="/home/maths/btech/mt1140594/workspace/VisionProject/scripts/weights_lstm_only1.hdf5", verbose=1, save_best_only=True)

#tmp = finalmodel.fit([x_train,Xtrain], y_train, validation_data=([x_val,Xval], y_val), batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, callbacks=[checkpointer])
#tmp = finalmodel.fit( x_train, y_train, validation_data=( x_val, y_val), batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
tmp = finalmodel.fit( Xtrain, y_train, validation_data=( Xval, y_val), batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, callbacks=[checkpointer])

#finalmodel.save('model1.h5')
