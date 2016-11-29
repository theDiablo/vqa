from vqaTools.vqa import VQA
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os
import numpy as np

dataDir='../../VQA'
taskType='OpenEnded'
dataType='mscoco' # 'mscoco' for real and 'abstract_v002' for abstract
dataSubType='val2014'
annFile='%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
quesFile='%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType)
imgDir = '%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)

# initialize VQA api for QA annotations
vqa=VQA(annFile, quesFile)

qa = vqa.qa
qqa = vqa.qqa

question_keys = qqa.keys()

text_file_key = open((dataSubType + "/image_ids.txt"), "a")
text_file = open((dataSubType + "/questions.txt"), "a")
for key in question_keys:
	question = qqa[key]['question']
	question_tmp = question.split('?')
	question = str(question_tmp[0]).lower()
	
	dic = {'\'s':' \'s', 's\'':' \'s', 'n\'t':' n\'t', '\'re':' \'re', '\'d':' \'d', '(':'', ')':'', ',':'', '.':'', '"':'', '-':'', '\'ve':' \'ve', '\\s+':' '}
	for i, j in dic.iteritems():
        	question = question.replace(i, j)
    	
	question = question.replace("\b[\p{L}]{0,1}\b","")
	print question
	text_file.write(question+"\n")
	text_file_key.write(str(qqa[key]['image_id'])+"\n")
	
text_file.close()
text_file_key.close()
	
