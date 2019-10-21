#!/usr/bin/env python

import pickle
import codecs
from tqdm import tqdm

data_dir = 'bank_dataset'
f=open(data_dir+'/id2word.pkl','rb')
id2word = pickle.load(f)

with codecs.open(data_dir+'/train.csv','r') as fi,codecs.open(data_dir+'/question.csv','r') as fii:
    train_lines = fi.readlines()
    ques_lines = fii.readlines()
    ques_dict = {}
    for i in tqdm(range(len(ques_lines))):
        if i == 0:
            continue
        line = ques_lines[i]
        line_lst = line.strip().split(',')
        ques_dict[line_lst[0]]=line_lst[1]
        #print(ques_dict)
        #input('wait')
    for i in range(len(train_lines)):
        if i == 0:
            continue
        line = train_lines[i].strip()
        line_lst = line.split(',')
        q1_id = line_lst[1]
        q2_id = line_lst[-1]
        q1_tokens_id = ques_dict[q1_id].split()
        q2_tokens_id = ques_dict[q2_id].split()

        q1_text = []
        for item in q1_tokens_id:
            q1_text.append(id2word[item])
        q2_text = []
        for item in q2_tokens_id:
            q2_text.append(id2word[item])
        
        print(''.join(q1_text))
        print(''.join(q2_text))
        print(line_lst[0])
        input('wait')
