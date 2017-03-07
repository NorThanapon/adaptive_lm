import time
import os
import codecs

import numpy as np
import tensorflow as tf

from adaptive_lm.utils.data import Vocabulary
from adaptive_lm.utils import common as common_utils
from adaptive_lm.utils import run as run_utils
from adaptive_lm.utils import decode as decode_utils

def run(opt, exp_opt, logger):
    data_kwargs = exp_opt.get('data_kwargs', {})
    vocab = Vocabulary.from_vocab_file(
        os.path.join(opt.data_dir, opt.vocab_file))
    in_vocab = data_kwargs.get('x_vocab', vocab)
    out_vocab = data_kwargs.get('y_vocab', vocab)
    opt.input_vocab_size = in_vocab.vocab_size
    opt.output_vocab_size = out_vocab.vocab_size
    logger.debug('Staring session...')
    sess_config = common_utils.get_tf_sess_config(opt)
    with tf.Session(config=sess_config) as sess:
        _, test_model, _, _ = run_utils.create_model(
            opt, exp_opt)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        _, _ = run_utils.load_model_and_states(
            opt.experiment_dir, sess, saver, [exp_opt.best])
        logger.info('Decoding ...')
        batch = exp_opt.iterator_cls.get_empty_batch(
            opt.batch_size, opt.num_steps)
        decoder = decode_utils.Emb2SeqDecoder(
            sess, test_model, in_vocab, out_vocab, opt)
        defs, s = decoder.decode(batch, exp_opt.word_list)
    output_path = os.path.join(opt.experiment_dir, opt.output_file)
    with codecs.open(output_path, 'w', 'utf-8') as ofp:
        for i, d in enumerate(defs):
            if "</s>" in d:
                eos_idx = d.index("</s>") - 1
            else:
                eos_idx = len(d)
            ofp.write('{}\t{}\n'.format(
                ' '.join(d[0:eos_idx + 2]), np.log(s[i, 0:eos_idx]).mean()))
