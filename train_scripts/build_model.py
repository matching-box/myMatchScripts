#!/usr/bin/env python3

import tensorflow as tf
import os

def build_save(model,MODEL_VERSION,save_dir):
    builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(save_dir, MODEL_VERSION))
    inputs = {
        "training":tf.saved_model.utils.build_tensor_info(model.training),
        "labels":tf.saved_model.utils.build_tensor_info(model.labels),
        "seq_word_left":tf.saved_model.utils.build_tensor_info(model.seq_word_left),
        "seq_word_right": tf.saved_model.utils.build_tensor_info(model.seq_word_right),
        "seq_char_left": tf.saved_model.utils.build_tensor_info(model.seq_char_left),
        "seq_char_right": tf.saved_model.utils.build_tensor_info(model.seq_char_right),
        "seq_len_word_left": tf.saved_model.utils.build_tensor_info(model.seq_len_word_left),
        "seq_len_word_right": tf.saved_model.utils.build_tensor_info(model.seq_len_word_right),
        "seq_len_char_left": tf.saved_model.utils.build_tensor_info(model.seq_len_char_left),
        "seq_len_char_right": tf.saved_model.utils.build_tensor_info(model.seq_len_char_right),
        "features": tf.saved_model.utils.build_tensor_info(model.features)
    }
    outputs = {"logits": tf.saved_model.utils.build_tensor_info(model.logits),
        "proba": tf.saved_model.utils.build_tensor_info(model.proba)
    }
    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=outputs,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    builder.add_meta_graph_and_variables(
        model.sess,
        [tf.saved_model.tag_constants.SERVING],
        {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature}
    )
    builder.save()
    print("Model Saved Succeed.")

#"features": tf.saved_model.utils.build_tensor_info(model.features)
