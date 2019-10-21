#!/usr/bin/env python3
# 

import codecs
import re
import random
import numpy as np
from gensim.models import KeyedVectors
import pickle
import segment_hanlp as hanlp 
from tqdm import tqdm

a='origin-data/whole'
c='dataset/word_embed.txt'
d='dataset/question.csv'
e='dataset/train.csv'
id2word_dict='dataset/id2word.pkl'
word2id_dict='dataset/word2id.pkl'
#regular = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
regular = re.compile("[^\u4e00-\u9fa5^a-z^A-Z]")

general_embedding_dir = 'embedding/w2v_zh_100w.vec'
model = KeyedVectors.load(general_embedding_dir, mmap='r')

with codecs.open(a,'r') as fi,codecs.open(c,'w') as femb,codecs.open(d,'w') as fques,codecs.open(e,'w') as ftrain:
    lines_origin = fi.readlines()
    origin_lst = []
    trans_lst = []
    vocab_lst = []
    labels = []
    for i in tqdm(range(len(lines_origin))):
        line_lst = lines_origin[i].strip().split()
        #labels.append(line_lst[-1])#here is a bug,labels append should after judgement of null
        #number should be processed especially
	#null value
        ori_str_res = ' '.join(hanlp.get_segment(regular.sub('', line_lst[0].strip())))
        trans_str_res = ' '.join(hanlp.get_segment(regular.sub('', line_lst[1].strip())))
        if len(ori_str_res) == 0 or len(trans_str_res) == 0:
            continue
        else:
            labels.append(line_lst[-1])
            origin_lst.append(ori_str_res)
            trans_lst.append(trans_str_res)
	#print(line_lst)
	#print(origin_lst)
	#print(trans_lst)
	#print(labels)
	#input('haha')
	
            ori_sent_lst = ori_str_res.strip().split()
            trans_sent_lst = trans_str_res.strip().split()
            vocab_lst += ori_sent_lst
            vocab_lst += trans_sent_lst
    vocab_lst_f = list(set(vocab_lst))
    #print(vocab_lst_f)
    #input(vocab_lst_f)
    vocab_lst_f.sort(key=vocab_lst.index)
    id2word = {}#W0001:'hello'
    word2id = {}
    for vi in range(len(vocab_lst_f)):
        zero_num = len(str(len(vocab_lst_f)))-len(str(vi))
        em_ind = 'W'+str(0)*zero_num+str(vi)
        vocab = vocab_lst_f[vi]
        id2word[em_ind] = vocab
        word2id[vocab] = em_ind
    f = open(id2word_dict, 'wb')
    pickle.dump(id2word, f)
    f.close()
    f = open(word2id_dict, 'wb')
    pickle.dump(word2id, f)
    f.close()
    print('vocab len:%d' % len(word2id))
    print('pkl write done!')
    #print(list(vocab_dict.keys())[list(vocab_dict.values()).index('value')])#find key accordding to value
    #input('wait')
    #vocab_lst_f.sort(key=vocab_lst.index)
    #print(vocab_dict)
    #input('vocab_lst')
    origin_ids_lst = []
    trans_ids_lst = []
    length_distribution = []
    for i in tqdm(range(len(trans_lst))):
        origin_sent_id = [word2id[tmp] for tmp in origin_lst[i].strip().split()]
        trans_sent_id = [word2id[tmp] for tmp in trans_lst[i].strip().split()]
        origin_ids_lst.append(origin_sent_id)
        trans_ids_lst.append(trans_sent_id)
        length_distribution.append(len(origin_sent_id))
    #print(length_distribution)
    print('max sentence length:%d'%max(length_distribution))
    #for i in range(len(trans_lst),2*len(trans_lst)):
    #    flag = True
    #    while flag:
    #        rand_val = random.randint(len(trans_lst),2*len(trans_lst)-1)
    #        if rand_val != i:
    #            break
    #    zero_num = len(str(2*len(trans_ids_lst)))-len(str(rand_val))
    #    q_ind_neg = 'Q'+str(0)*zero_num+str(rand_val)
    #    neg_lst.append(q_ind_neg)
    #    #print(rand_val)
    #    #print(neg_lst)
    #    #input('wait')
    #neg_lst3 = []
    #for i in range(2*len(trans_lst),3*len(trans_lst)):
    #    flag = True
    #    while flag:
    #        rand_val = random.randint(2*len(trans_lst),3*len(trans_lst)-1)
    #        if rand_val != i:
    #            break
    #    zero_num = len(str(3*len(trans_ids_lst)))-len(str(rand_val))
    #    q_ind_neg = 'Q'+str(0)*zero_num+str(rand_val)
    #    neg_lst3.append(q_ind_neg)
    count = 0
    for em_ind in tqdm(id2word):
        item = id2word[em_ind]
        #print(item)
        try:
            embedding = model[item]
            #print(embedding)
            #input('model embedding')
        except KeyError as ke:
            count += 1
            embedding = np.random.random(300).tolist()
            #print(embedding)
            #input('random embedding')
        femb.write(em_ind+' '+' '.join([str(i) for i in embedding])+'\n')
    print('%d cannot find.'%count)
    q_inds_trans = []
    q_inds_origin = []
    length = len(trans_ids_lst)
    print('corpus length:%d'%length)
    fques.write('qid,words,chars'+'\n')
    for i in range(length):
        zero_num = len(str(2*length))-len(str(i))
        q_ind = 'Q'+str(0)*zero_num+str(i)
        q_inds_trans.append(q_ind)
        fques.write(q_ind+','+' '.join(trans_ids_lst[i])+','+' '.join(trans_ids_lst[i])+'\n')
    for i in range(length):
        tmp_index = i+length
        zero_num = len(str(2*length))-len(str(tmp_index))
        q_ind = 'Q'+str(0)*zero_num+str(tmp_index)
        q_inds_origin.append(q_ind)
        fques.write(q_ind+','+' '.join(origin_ids_lst[i])+','+' '.join(origin_ids_lst[i])+'\n')
    #for i in range(length):
    #    tmp_index = i+2*length
    #    zero_num = len(str(3*length))-len(str(tmp_index))
    #    q_ind = 'Q'+str(0)*zero_num+str(tmp_index)
    #    q_inds_trans3.append(q_ind)
    #    fques.write(q_ind+','+' '.join(trans_ids_lst3[i])+','+' '.join(trans_ids_lst3[i])+'\n')
    
    for i in range(length):
        ftrain.write(str(labels[i])+','+q_inds_origin[i]+','+q_inds_trans[i]+'\n')
    #for i in range(length):
    #    ftrain.write(str(1)+','+q_inds_origin[i]+','+q_inds_origin[i]+'\n')
    #for i in range(length):
    #    ftrain.write(str(1)+','+q_inds_trans[i]+','+q_inds_trans3[i]+'\n')
    #for i in range(length):
    #    ftrain.write(str(1)+','+q_inds_origin[i]+','+q_inds_trans3[i]+'\n')
    
    #for i in range(length):
    #    ftrain.write(str(0)+','+q_inds_trans[i]+','+neg_lst[i]+'\n')
    #for i in range(length):
    #    ftrain.write(str(0)+','+q_inds_trans[i]+','+neg_lst3[i]+'\n')
    #for i in range(length):
    #    ftrain.write(str(0)+','+neg_lst[i]+','+neg_lst3[i]+'\n')
    
    #for i in range(length):
    #    if i < length-1:
    #        ftrain.write(str(0)+','+q_inds_trans[i]+','+q_inds_trans[i+1]+'\n')
    #    else:
    #        ftrain.write(str(0)+','+q_inds_trans[i]+','+q_inds_trans[0]+'\n')
    #for i in range(length):
    #    if i < length-1:
    #        ftrain.write(str(0)+','+q_inds_origin[i]+','+q_inds_origin[i+1]+'\n')
    #    else:
    #        ftrain.write(str(0)+','+q_inds_origin[i]+','+q_inds_origin[0]+'\n')
    #for i in range(length):
    #    rand_v = random.randint(0,length)
    #    ftrain.write(str(0)+','+q_inds_origin[i]+','+q_inds_trans[rand_v]+'\n')

