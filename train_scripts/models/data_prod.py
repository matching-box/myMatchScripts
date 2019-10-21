
import numpy as np

def _get_pos_neg_ind(label):
    length = len(label)
    pos_ind_tmp = np.where(label == 1)[0]
    inds = np.zeros((len(pos_ind_tmp) * length, 2), dtype=int)
    inds[:, 0] = np.tile(pos_ind_tmp, length)
    inds[:, 1] = list(range(length)) * len(pos_ind_tmp)
    mask = inds[:, 0] != inds[:, 1]
    pos_ind = inds[mask, 0]
    neg_ind = inds[mask, 1]
    return pos_ind, neg_ind


def get_feed_dict(X, idx, Q, params, construct_neg=False, training=False, symmetric=False):
    if training:
        if construct_neg:
            q1 = X["q1"][idx]
            q2 = X["q2"][idx]
            # for label=1 sample, construct negative sample within batch
            pos_ind, neg_ind = _get_pos_neg_ind(X["label"][idx])
            # original & symmetric
            feed_dict = {
                'seq_word_left': np.vstack([Q["words"][q1],
                                               Q["words"][X["q1"][idx[pos_ind]]],
                                               Q["words"][X["q1"][idx[neg_ind]]],
                                               Q["words"][q2],
                                               Q["words"][X["q2"][idx[neg_ind]]],
                                               Q["words"][X["q2"][idx[pos_ind]]]
                                               ]),
                'seq_word_right': np.vstack([Q["words"][q2],
                                                Q["words"][X["q2"][idx[neg_ind]]],
                                                Q["words"][X["q2"][idx[pos_ind]]],
                                                Q["words"][q1],
                                                Q["words"][X["q1"][idx[pos_ind]]],
                                                Q["words"][X["q1"][idx[neg_ind]]],
                                                ]),
                'seq_char_left': np.vstack([Q["chars"][q1],
                                               Q["chars"][X["q1"][idx[pos_ind]]],
                                               Q["chars"][X["q1"][idx[neg_ind]]],
                                               Q["chars"][q2],
                                               Q["chars"][X["q2"][idx[neg_ind]]],
                                               Q["chars"][X["q2"][idx[pos_ind]]]
                                               ]),
                'seq_char_right': np.vstack([Q["chars"][q2],
                                                Q["chars"][X["q2"][idx[neg_ind]]],
                                                Q["chars"][X["q2"][idx[pos_ind]]],
                                                Q["chars"][q1],
                                                Q["chars"][X["q1"][idx[pos_ind]]],
                                                Q["chars"][X["q1"][idx[neg_ind]]]
                                                ]),
                'labels': np.hstack([X["label"][idx],
                                        np.zeros(len(pos_ind)),
                                        np.zeros(len(pos_ind)),
                                        X["label"][idx],
                                        np.zeros(len(pos_ind)),
                                        np.zeros(len(pos_ind))
                                        ]),
                'training': training,
            }
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
    elif not symmetric:
        q1 = X["q1"][idx]
        q2 = X["q2"][idx]
        feed_dict = {
            'seq_word_left': Q["words"][q1],
            'seq_word_right': Q["words"][q2],
            'seq_char_left': Q["chars"][q1],
            'seq_char_right': Q["chars"][q2],
            'seq_len_word_left': Q["seq_len_word"][q1],
            'seq_len_word_right': Q["seq_len_word"][q2],
            'seq_len_char_left': Q["seq_len_char"][q1],
            'seq_len_char_right': Q["seq_len_char"][q2],
            'labels': X["label"][idx],
            'training': training,
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

