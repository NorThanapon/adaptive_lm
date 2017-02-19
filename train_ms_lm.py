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

def run_train_epoch(sess, m, data_iter, opt, mapper,
              train_op):
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
        feed_dict = {m.x: x, m.w: w, m.seq_len: seq_len,
                     m.sparse_logit_mask:sparse_mask_indices}
        fetches = [m.loss, train_op]
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
        if (step + 1) % opt.progress_steps == 0:
            logger.info("-- @{} perplexity: {} wps: {}".format(
                    step + 1, np.exp(costs / num_words),
                    num_words / (time.time() - start_time)))
    return np.exp(costs / num_words), step

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
        feed_dict = {m.x: x, m.y: y, m.w: w, m.seq_len: seq_len,
                     m.sparse_logit_mask:sparse_mask_indices}
        fetches = [m.output_probs]
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
        probs = res[0]
        merged_probs = mapper.reduce_sum(probs)
        cost = sess.run(m.merged_loss,
                               feed_dict={m.y: y, m.w: w,
                                          m.merged_probs:merged_probs})
        state_flat = res[f_state_start:]
        state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]
        b_num_words = np.sum(w)
        num_words += b_num_words
        costs += cost * b_num_words
    return np.exp(costs / num_words), step

def main(opt):
    model_prefix = ['latest_lm']
    dataset = ['train', 'valid']
    lm_data, lm_vocab = load_datasets(lm_opt, dataset=dataset)
    mapper = data_utils.OneToManyMap.from_map_file(lm_opt.map_filepath)
    lm_opt.vocab_size = lm_vocab.vocab_size
    lm_opt.softmax_vocab_size = mapper.total_size
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
        saver = tf.train.Saver()
        states = {}
        for p in model_prefix:
            states[p] = common_utils.get_initial_training_state()
        states, _ = resume_many_states(lm_opt.output_dir, sess,
                                       saver, states, model_prefix)
        lm_state = states[model_prefix[0]]
        lm_state.learning_rate = lm_opt.learning_rate

        logger.info('Start training loop:')
        logger.debug('\n' + common_utils.SUN_BRO())

        for epoch in range(lm_state.epoch, lm_opt.max_epochs):
            epoch_time = time.time()
            logger.info("========= Start epoch {} =========".format(epoch+1))
            sess.run(tf.assign(lm_lr_var, lm_state.learning_rate))
            logger.info("- Traning LM with learning rate {}...".format(
                lm_state.learning_rate))
            lm_train_ppl, _ = run_train_epoch(sess, lm_train, lm_data['train'],
                                              lm_opt, mapper, lm_train_op)
            logger.info('- Validating LM...')
            lm_valid_ppl, _ = run_test_epoch(sess, lm_valid, lm_data['valid'],
                                             lm_opt, mapper)
            logger.info('----------------------------------')
            logger.info('LM post epoch routine...')
            done_training = run_post_epoch(
                lm_train_ppl, lm_valid_ppl, lm_state, lm_opt,
                sess=sess, saver=saver,
                best_prefix="best_lm", latest_prefix="latest_lm")
            logger.info('- Epoch time: {}s'.format(time.time() - epoch_time))
            if done_training:
                break
        logger.info('Done training at epoch {}'.format(lm_state.epoch + 1))

if __name__ == "__main__":
    global_time = time.time()
    parser = common_utils.get_common_argparse()
    parser.add_argument('--map_filepath', type=str,
                        default='experiments/multi-softmax/ptb/1to2.map.txt')
    args = parser.parse_args()
    lm_opt = common_utils.Bunch.default_model_options()
    lm_opt.update_from_ns(args)
    logger = common_utils.get_logger(lm_opt.log_file_path)
    if lm_opt.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info('Configurations:\n{}'.format(lm_opt.__repr__()))
    main(lm_opt)
    logger.info('Total time: {}s'.format(time.time() - global_time))
