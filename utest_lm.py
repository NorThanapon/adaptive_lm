import argparse
import logging
import time
import os
import json
import random
import cPickle
# random.seed(1234)

import numpy as np
# np.random.seed(1234)
import tensorflow as tf
# tf.set_random_seed(1234)

import lm
import common_utils
import data_utils
from exp_utils import *

parser = common_utils.get_common_argparse()
args = parser.parse_args()
lm_opt = common_utils.Bunch.default_model_options()
lm_opt.update_from_ns(args)
logger = common_utils.get_logger(lm_opt.log_file_path)
logger.setLevel(logging.DEBUG)

dataset = ['train', 'valid']
lm_data, lm_vocab = load_datasets(lm_opt, dataset=dataset)
init_scale = lm_opt.init_scale
sess_config =tf.ConfigProto(log_device_placement=False,
                            device_count = {'GPU': 0})

with tf.Session(config=sess_config) as sess:
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    with tf.variable_scope('LM', reuse=None, initializer=initializer):
        lm_train = lm.MaxTargetLossLM(lm_opt)
        lm_train_op, lm_lr_var = lm.train_op(lm_train, lm_opt)
    with tf.variable_scope('LM', reuse=True, initializer=initializer):
        lm_valid = lm.MaxTargetLossLM(lm_opt, is_training=False)
    logger.debug('Trainable variables:')
    for v in tf.trainable_variables():
        logger.debug("- {} {} {}".format(v.name, v.get_shape(), v.device))
    logger.info('Initializing vairables...')
    sess.run(tf.global_variables_initializer())

    data_iter = lm_data['train']
    m = lm_train
    data_iter.init_batch(lm_opt.batch_size, lm_opt.num_steps)
    x, y, w, l, seq_len = data_iter.next_batch()
    feed_dict = {m.x: x, m.y: y, m.w: w, m.seq_len: seq_len}
    mask = np.zeros([320, 10000], dtype=np.int32)
    mask[:,:] = -100000
    mask[:,7] = 0
    feed_dict[m.local_logit_mask] = mask
    fetches = [m._all_logits, m.targets]
    res = sess.run(fetches, feed_dict=feed_dict)
