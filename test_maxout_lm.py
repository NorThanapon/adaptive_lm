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

def write_token_max_channel(ofp, channel_probs, y):
    max_channels = np.argmax(channel_probs, axis=2)
    targets = np.reshape(y, [-1])
    for i in range(len(targets)):
        ofp.write('{}\t{}\n'.format(targets[i], max_channels[i][targets[i]]))

def run_test_epoch(sess, m, data_iter, opt, mapper):
    """ train the model on the given data. """
    logger = logging.getLogger("exp")
    start_time = time.time()
    costs = 0.0
    num_words = 0
    state = []
    for c, h in m.initial_state:
        state.append((c.eval(), h.eval()))
    for step, (x, y, w, l, seq_len) in enumerate(data_iter.iterate_epoch(
        m.opt.batch_size, m.opt.num_steps)):
        sparse_mask_indices = mapper.create_sparse_indices(y)
        feed_dict = {m.x: x, m.y: y, m.w: w, m.seq_len: seq_len}
        fetches = [m.merged_loss, m.channel_probs]
        f_state_start = len(fetches)
        if opt.sen_independent and data_iter.is_new_sen():
            state = []
            for c, h in m.initial_state:
                state.append((c.eval(), h.eval()))
        for i, (c, h) in enumerate(m.initial_state):
            feed_dict[c], feed_dict[h] = state[i]
        for c, h in m.final_state:
            fetches.append(c)
            fetches.append(h)
        res = sess.run(fetches, feed_dict)
        cost = res[0]
        state_flat = res[f_state_start:]
        state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]
        b_num_words = np.sum(w)
        num_words += b_num_words
        costs += cost * b_num_words
        if opt.channel_log is not None:
            write_token_max_channel(opt.channel_log, res[1], y)
    return np.exp(costs / num_words), step

def main(lm_opt):
    model_prefix = ['best_lm']
    dataset = ['valid']
    lm_data, lm_vocab = load_datasets(lm_opt, dataset=dataset)
    mapper = data_utils.OneToManyMap.from_map_file(lm_opt.map_filepath)
    lm_opt.vocab_size = lm_vocab.vocab_size
    lm_opt.softmax_vocab_size = mapper.total_size
    lm_opt.vocab_segments = mapper.segments
    lm_opt.num_channels = mapper.max_num_values
    init_scale = lm_opt.init_scale
    sess_config = common_utils.get_tf_sess_config(lm_opt)
    with tf.Session(config=sess_config) as sess:
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope('LM', initializer=initializer):
            lm_valid = lm.MaxoutLogitsLM(lm_opt, is_training=False)
        logger.debug('Trainable variables:')
        for v in tf.trainable_variables():
            logger.debug("- {} {} {}".format(v.name, v.get_shape(), v.device))
        logger.info('Initializing vairables...')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        states = {}
        for p in model_prefix:
            states[p] = common_utils.get_initial_training_state()
        states, success = resume_many_states(lm_opt.output_dir, sess,
                                       saver, states, model_prefix)
        if not success:
            logger.warning('Fail to load the model...')
        lm_state = states[model_prefix[0]]
        lm_state.learning_rate = lm_opt.learning_rate

        logger.info('Start testing...')
        lm_ppl, _ = run_test_epoch(sess, lm_valid, lm_data[dataset[0]],
                                   lm_opt, mapper)
        logger.info('PPL: {}'.format(lm_ppl))

if __name__ == "__main__":
    global_time = time.time()
    parser = common_utils.get_common_argparse()
    parser.add_argument('--map_filepath', type=str,
                        default='experiments/multi-softmax/ptb/1to2.map.txt')
    parser.add_argument('--channel_log_file', type=str,
                        default='')
    args = parser.parse_args()
    lm_opt = common_utils.Bunch.default_model_options()
    lm_opt.update_from_ns(args)
    logger = common_utils.get_logger(lm_opt.log_file_path)
    if lm_opt.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info('Configurations:\n{}'.format(lm_opt.__repr__()))
    if lm_opt.channel_log_file != "":
        lm_opt.channel_log = open(
            os.path.join(lm_opt.output_dir, lm_opt.channel_log_file), 'w')
    main(lm_opt)
    logger.info('Total time: {}s'.format(time.time() - global_time))
