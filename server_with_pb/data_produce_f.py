
import config
import jieba
import pickle
import numpy as np
import pandas as pd
import scipy as sp
from keras.preprocessing.sequence import pad_sequences

word2id_dir = 'data/word2id.pkl'
fpk = open(word2id_dir,'rb')
word2id = pickle.load(fpk)

def _to_ind(qid):
    return int(qid[1:])


def load_raw_question(df):
    df["words"] = df.words.str.split(" ")
    df["chars"] = df.chars.str.split(" ")
    Q = {}
    Q["words"] = df["words"].values
    Q["chars"] = df["chars"].values
    return Q


def load_question(params,text1,text2):
    #df = pd.read_csv(config.QUESTION_FILE)
    
    text1_ids_lst = []
    for tmp in list(jieba.cut(text1)):
        try:
            text1_ids_lst.append(word2id[tmp])
        except:
            text1_ids_lst.append('W'+config.MISSING_INDEX_WORD)
    text2_ids_lst = []
    for tmp in list(jieba.cut(text2)):
        try:
            text2_ids_lst.append(word2id[tmp])
        except:
            text2_ids_lst.append('W'+config.MISSING_INDEX_WORD)
    
    tmp_arr = [text1_ids_lst,text2_ids_lst]
    df = pd.DataFrame(columns=['words','chars'])
    df['words']=tmp_arr
    df['chars']=tmp_arr
    #print(list(jieba.cut(text1)))
    #print(list(jieba.cut(text2)))
    #print(df['words'])
    #input('df')
    df["words"] = df['words'].apply(lambda x: [_to_ind(z) for z in x])
    df["chars"] = df['chars'].apply(lambda x: [_to_ind(z) for z in x])
    Q = {}
    Q["seq_len_word"] = sp.minimum(df["words"].apply(len).values, params["max_seq_len_word"])
    Q["seq_len_char"] = sp.minimum(df["chars"].apply(len).values, params["max_seq_len_char"])
    Q["words"] = pad_sequences(df["words"],
                               maxlen=params["max_seq_len_word"],
                               padding=params["pad_sequences_padding"],
                               truncating=params["pad_sequences_truncating"],
                               value=config.PADDING_INDEX_WORD)
    Q["chars"] = pad_sequences(df["chars"],
                               maxlen=params["max_seq_len_char"],
                               padding=params["pad_sequences_padding"],
                               truncating=params["pad_sequences_truncating"],
                               value=config.PADDING_INDEX_CHAR)
    return Q


def load_train(df):
    #df = pd.read_csv(config.TRAIN_FILE)
    df["q1"] = df.q1.apply(_to_ind)
    df["q2"] = df.q2.apply(_to_ind)
    return df


def load_test(df):
    #df = pd.read_csv(config.TEST_FILE)
    df["q1"] = df.q1.apply(_to_ind)
    df["q2"] = df.q2.apply(_to_ind)
    df["label"] = np.zeros(df.shape[0])
    return df


def load_embedding_matrix(embedding_file):
    print("read embedding from: %s " %embedding_file)
    d = {}
    n = 0
    with open(embedding_file, "r") as f:
        line = f.readline()
        while line:
            n += 1
            w, v = line.strip().split(" ", 1)
            d[int(w[1:])] = v
            line = f.readline()
    dim = len(v.split(" "))

    # add two index for missing and padding
    emb_matrix = np.zeros((n+2, dim), dtype=float)
    for key ,val in d.items():
        v = np.asarray(val.split(" "), dtype=float)
        emb_matrix[key] = v
    emb_matrix = np.array(emb_matrix, dtype=np.float32)
    return emb_matrix


#init_embedding_matrix = {
#    "word": load_embedding_matrix(config.WORD_EMBEDDING_FILE),
#    "char": load_embedding_matrix(config.CHAR_EMBEDDING_FILE),
#}


def get_feed_dict(Q, params, construct_neg=False, training=False, symmetric=False):
    if not symmetric:
        #q1 = X["q1"][idx]
        #q2 = X["q2"][idx]
        q1 = 0
        q2 = 1
        feed_dict = {
            'seq_word_left': Q["words"][q1].reshape(1,params['max_seq_len_word']),
            'seq_word_right': Q["words"][q2].reshape(1,params['max_seq_len_word']),
            'seq_char_left': Q["chars"][q1].reshape(1,params['max_seq_len_word']),
            'seq_char_right': Q["chars"][q2].reshape(1,params['max_seq_len_word']),
            'seq_len_word_left': [Q["seq_len_word"][q1]],
            'seq_len_word_right': [Q["seq_len_word"][q2]],
            'seq_len_char_left': [Q["seq_len_char"][q1]],
            'seq_len_char_right': [Q["seq_len_char"][q2]],
            'labels': [0],
            'training': False,
        }
        if params["use_features"]:
            feed_dict.update({
                'features': X["features"][idx],
            })
    else:
        q1 = X["q1"][idx]
        q2 = X["q2"][idx]
        feed_dict = {
            'seq_word_left': np.vstack([Q["words"][q1],
                                           Q["words"][q2],
                                           ]),
            'seq_word_right': np.vstack([Q["words"][q2],
                                            Q["words"][q1],
                                            ]),
            'seq_char_left': np.vstack([Q["chars"][q1],
                                           Q["chars"][q2],
                                           ]),
            'seq_char_right': np.vstack([Q["chars"][q2],
                                            Q["chars"][q1],
                                            ]),
            'seq_len_word_left': np.hstack([Q["seq_len_word"][q1],
                                               Q["seq_len_word"][q2],
                                               ]),
            'seq_len_word_right': np.hstack([Q["seq_len_word"][q2],
                                                Q["seq_len_word"][q1],
                                                ]),
            'seq_len_char_left': np.hstack([Q["seq_len_char"][q1],
                                               Q["seq_len_char"][q2],
                                               ]),
            'seq_len_char_right': np.hstack([Q["seq_len_char"][q2],
                                                Q["seq_len_char"][q1],
                                                ]),
            'labels': np.hstack([X["label"][idx],
                                    X["label"][idx],
                                    ]),
            'training': training,
        }
        if params["use_features"]:
            feed_dict.update({
                'features': np.vstack([X["features"][idx],
                                          X["features"][idx],
                                          ]),
            })

    return feed_dict
def get_batch_index(seq, step):
    n = len(seq)
    res = []
    for i in range(0, n, step):
        res.append(seq[i:i + step])
    # last batch
    if len(res) * step < n:
        res.append(seq[len(res) * step:])
    return res
