from vqaTools.vqa import VQA
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os
import numpy as np

dataDir='../../VQA'
taskType='OpenEnded'
dataType='mscoco' # 'mscoco' for real and 'abstract_v002' for abstract
dataSubType='train2014'
annFile='%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
quesFile='%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType)
imgDir = '%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)

# initialize VQA api for QA annotations
vqa=VQA(annFile, quesFile)

qa = vqa.qa
qqa = vqa.qqa

answer_keys = qa.keys()

text_file = open((dataSubType+"/answers.txt"), "a")
for key in answer_keys:
	answers = qa[key]['answers']
	map = {}
	max = 0
	final_ans=''
	for single in answers:
		certain=str(single['answer_confidence'])
		#print single['answer']
		single_answer=str(single['answer'].encode("utf8"))
		#print single_answer
		dic = {'\'s':' \'s', 's\'':' \'s', 'n\'t':' n\'t', '\'re':' \'re', '\'d':' \'d', '(':'', ')':'', ',':'', '.':'', '"':'', '-':'', '\'ve':' \'ve', '\\s+':' '}
		for i, j in dic.iteritems():
                	single_answer = single_answer.replace(i, j)

		this_count=0
		if certain=='yes' or certain=='may_be':
			if map.has_key(single_answer):
				count=map[single_answer]
				map[single_answer]=count+1
				this_count=count+1
			else:
				map[single_answer]=1
				this_count=1
		if this_count>max:
			max=this_count
			final_ans=single_answer
	final_ans = str(final_ans).lower()
	text_file.write(final_ans+"\n")
	
text_file.close()
	
