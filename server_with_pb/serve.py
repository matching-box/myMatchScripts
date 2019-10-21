#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from tensorflow.contrib import predictor
import data_produce_f

params = {
        "construct_neg": False,
        "use_features": False,
        "threshold": 0.217277,
        "calibration": False,

        "max_seq_len_word": 63,
        "max_seq_len_char": 63,
        "pad_sequences_padding": "post",
        "pad_sequences_truncating": "post",

    }


if __name__ == '__main__':
    export_dir = 'pb'
    subdirs = [x for x in Path(export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    print(latest)
    predict_fn = predictor.from_saved_model(latest)
    #words = [w.encode() for w in LINE.split()]
    #nwords = len(words)
    query = data_produce_f.load_question(params,'我想办信用卡','信用卡怎么办理')
    feed_dict = data_produce_f.get_feed_dict(query,params)
    print(feed_dict)
    input('wait')
    predictions = predict_fn({'training':False,
        'labels':[0],
        'seq_word_left':feed_dict['seq_word_left'],
        'seq_word_right':feed_dict['seq_word_right'],
        'seq_char_left':feed_dict['seq_char_left'],
        'seq_char_right':feed_dict['seq_char_right'],
        'seq_len_word_left':feed_dict['seq_len_word_left'],
        'seq_len_word_right':feed_dict['seq_len_word_right'],
        'seq_len_char_left':feed_dict['seq_len_char_left'],
        'seq_len_char_right':feed_dict['seq_len_char_right']
        })
    #print(predictions['logits'])
    print(predictions['proba'])
