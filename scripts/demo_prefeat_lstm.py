from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Flatten, Dense, Dropout, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.embeddings import Embedding
from keras.layers.pooling import AveragePooling2D
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
import keras.models as models
import pandas as pd
from keras.layers import LSTM
import keras.layers.core as core
from scipy import misc
from PIL import Image
from keras.models import load_model
import numpy as np
import h5py
import scipy.io
import sys
import heapq

#File to read questions and answers
questionsFile="/scratch/maths/btech/mt1140594/VQA/PythonHelperTools/train2014/questions.txt"
answersFile="/scratch/maths/btech/mt1140594/VQA/PythonHelperTools/train2014/answers.txt"
image_File="/scratch/maths/btech/mt1140594/VQA/PythonHelperTools/train2014/imageData.csv"
image_index_File="/scratch/maths/btech/mt1140594/VQA/PythonHelperTools/train2014/image_ids.txt"

questionsFileVal="/scratch/maths/btech/mt1140594/VQA/PythonHelperTools/val2014/questions.txt"
answersFileVal="/scratch/maths/btech/mt1140594/VQA/PythonHelperTools/val2014/answers.txt"
image_FileVal="/scratch/maths/btech/mt1140594/VQA/PythonHelperTools/val2014/imageData.csv"
image_index_FileVal="/scratch/maths/btech/mt1140594/VQA/PythonHelperTools/val2014/image_ids.txt"

weightsFile='../models/weights_prefeat_lstm3.hdf5'
vggFeaturesFile='../resources/cnn-features/vgg_feats.mat'
imageIndexMapFile='../resources/cnn-features/img_ids_map.txt'

threshQ=4
threshA=13
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

def prepareQuestionData( vocab, question):

    #data=(open(questionsFile)).read().split('\n')
    #print('Total number of questions in File :'+str(len(data)-1))

    question=question.lower().split()
    row=[0]*10
    currentword=0
    for i in range(min(len(question),10)):
        if vocab.has_key(question[i]):
            value=vocab[question[i]]
            row[currentword]=value
            currentword=currentword+1

    return row
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

def prepareAnswerData( ans_vocab, answer_index ):

	for key in ans_vocab.keys():
		if ans_vocab[key]==answer_index+1:
			#print key
			return key


	return ''
    
#############################################################Image Data
def prepareImageData():
    
    #img_id_val=(open(image_index_FileVal)).read().split('\n')
    #img_id_train=(open(image_index_File)).read().split('\n')

    mat = scipy.io.loadmat(vggFeaturesFile)
    mat = mat['feats']
    Imap = {}
    with open(imageIndexMapFile, 'r') as document:
        for line in document:
            line = line.split(' ')
            if not line:  # empty line?
                continue
            Imap[int(line[0])] = line[1]

    return [Imap,mat]

def ImageVector(index,Imap,mat):


    row=np.zeros(4096, dtype='float16')
    #print Imap
    if Imap.has_key(index):
        j = int(Imap[index])
        
        print j
            #print('found: ' + str(j))
        tmp = mat[:,j]
        row = np.array(tmp, dtype='float16')
    else:
     	print('not found: ' + str(index))

    return row






###################################### CNN Image Model #######################################################

def loadCNNModel():
    
    print('Making CNN model..')

    model = Sequential()
    model.add(core.Dense(1024, activation="tanh", input_dim=4096,trainable=False))

    model.summary()
    return model

######################################## Text Model #######################################################

def loadTextModel(vocab_size, MAX_SEQUENCE_LENGTH):

    textmodel= Sequential()
    textmodel.add(Embedding(input_dim=(vocab_size + 1), input_length=MAX_SEQUENCE_LENGTH, output_dim=512, mask_zero=True,trainable=False))
    textmodel.add(LSTM(output_dim=512, return_sequences=True,trainable=False))
    textmodel.add(LSTM(output_dim=512, return_sequences=False,trainable=False))
    #textmodel.add(Reshape((1,10,1024)))
    #textmodel.add(MaxPooling2D(pool_size=(10,1)))
    #textmodel.add(Convolution2D(8, 5, 5, activation='relu'))
    #textmodel.add(Convolution2D(16, 3, 3, activation='relu'))
    #textmodel.add(Convolution2D(32, 3, 3, activation='relu'))
    # textmodel.add(Flatten())w
    # textmodel.add(core.Dense(4096, activation="relu"))
    # textmodel.add(Dropout(0.5))
    textmodel.add(core.Dense(1024, activation="tanh",trainable=False))

    textmodel.summary()
    return textmodel

######################################### Merge Models #######################################################

def mergeModel(model, textmodel):

    merged = core.Merge([model, textmodel], mode='mul')

    return merged



print('Making Questions Vocab')
[vocab, vocab_size] = prepareQuestionVocab( questionsFile, threshQ )
print('Making Answers Vocab')
ans_vocab = prepareAnswerVocab( answersFile, threshA )
print('Prepare Image Map')
[Idmap,mat]=prepareImageData()
print('Making Model')
model = loadCNNModel()
textmodel = loadTextModel(vocab_size, MAX_SEQUENCE_LENGTH)
merged = mergeModel(model, textmodel)
finalmodel= Sequential()
finalmodel.add(merged)
#finalmodel.add(textmodel)
#finalmodel.add(model)
finalmodel.add(core.Dense(4096, activation="relu",trainable=False))
finalmodel.add(Dropout(0.5))
finalmodel.add(core.Dense(len(ans_vocab), activation="softmax",trainable=False))

print('Loading Wights')
finalmodel.load_weights(weightsFile)

#print ans_vocab



for i in range(30):
	image_id= unicode(raw_input("Give the image Id:"))
	#image_id=292181
	question = unicode(raw_input("Ask a question about the image:"))	
	#question='What is in the picture'
	print('Making Question Vector')
	test_question=np.array(prepareQuestionData(vocab,question)).reshape((1,10))
	print('Making Image Feature Vector')
	image_vector=np.array(ImageVector(int(image_id),Idmap,mat)).reshape((1,4096))
	print image_vector
	#print('Making Model')
	#model = loadCNNModel()
	#textmodel = loadTextModel(vocab_size, MAX_SEQUENCE_LENGTH)
	#merged = mergeModel(model, textmodel)

	#finalmodel= Sequential()
	#finalmodel.add(merged)
#finalmodel.add(textmodel)
#finalmodel.add(model)
	#finalmodel.add(core.Dense(4096, activation="relu",trainable=False))
	#finalmodel.add(Dropout(0.5))
	#finalmodel.add(core.Dense(len(ans_vocab), activation="softmax",trainable=False))

	#print('Loading Wights')
	#finalmodel.load_weights(weightsFile)
	
	print('Predicting')
	y_predict = finalmodel.predict([image_vector, test_question], verbose=0)
	#sort_y=sorted(y_predict,reverse=True)
	y_predict=np.array(y_predict[0],dtype='float16')
	yp=y_predict
	#print yp
	#index=np.argmax(y_predict)
	pred=heapq.nlargest(5, range(len(y_predict)), y_predict.take)
	#print index
	print('Guess: '+ prepareAnswerData(ans_vocab,pred[0]) +' '+ prepareAnswerData(ans_vocab,pred[1])+' ' +prepareAnswerData(ans_vocab,pred[2])+' ' +prepareAnswerData(ans_vocab,pred[3])+' ' +prepareAnswerData(ans_vocab,pred[4]) ) 
	#yp[index]=0
	#index=np.argmax(yp)
	
	#print('Second guess '+ prepareAnswerData(ans_vocab,pred[1]))
	#y_predict[index]=0
	#index=np.argmax(y_predict)
	
	#print('Third guess '+ prepareAnswerData(ans_vocab,pred[2]))

	#print('Fourth guess '+ prepareAnswerData(ans_vocab,pred[3]))
