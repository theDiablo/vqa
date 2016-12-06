## Resources

Resource Directory for preperation of data.

1. generate_answer_txt.py/generate_question_txt.py: 
	Follow instructions from https://github.com/VT-vision-lab/VQA and in tree/master/PythonHelperTools paste these files and run them copy the 'train201x' and 'val201x' folders into resources directory with following files answers.txt, image_ids.txt, questions.txt for each of training and validation data.

2. cnn-features(dir): 
	Download vgg16 features for each image in MS COCO with name vgg_feats.mat and create img_ids_map.txt for mapping image id with index in 'mat' file in this directory.

3. glove(dir): 
	Download 'glove.6B.300d.txt', glove pre-trained word embedding from http://nlp.stanford.edu/data/glove.6B.zip
