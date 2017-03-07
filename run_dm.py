import argparse
import logging
import time
import os
import cPickle
import numpy as np

from adaptive_lm.models.rnnlm_helper import EmbDecoderRNNHelper
from adaptive_lm.models.basic_rnnlm import DecoderRNNLM
from adaptive_lm.utils import common as common_utils
from adaptive_lm.experiments import lm
from adaptive_lm.experiments import decode_dm
from adaptive_lm.utils.data import SenLabelIterator
from adaptive_lm.utils.data import Vocabulary

training_exp_opt = common_utils.LazyBunch(
    resume = 'latest_lm',
    best = 'best_lm',
    splits = ['train', 'valid'],
    run_split = 'valid',
    iterator_cls = SenLabelIterator,
    model_scope = 'DM',
    model_helper_cls = EmbDecoderRNNHelper,
    model_cls = DecoderRNNLM,
    build_train_fn = DecoderRNNLM.build_full_model_graph,
    build_test_fn = DecoderRNNLM.build_full_model_graph,
    init_variables = [],
    training = True,
    data_kwargs = common_utils.LazyBunch()
)

decoding_exp_opt = common_utils.LazyBunch(
    training_exp_opt,
    training = False)

if __name__ == '__main__':
    global_time = time.time()
    parser = common_utils.get_common_argparse()
    parser.add_argument('--emb_pickle_file', type=str,
                        default='emb.cpickle',
                        help='embedding cpickled file in data_dir')
    parser.add_argument('--output_vocab_file', type=str,
                        default='vocab_def.txt',
                        help='embedding cpickled file in data_dir')
    parser.add_argument('--tie_input_enc_emb', dest='tie_input_enc_emb',
                        action='store_true')
    parser.add_argument('--decoding', dest='decoding',
                        action='store_true')
    parser.add_argument('--sampling', dest='sampling',
                        action='store_true')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='number of samples for each word')
    parser.add_argument('--logit_temperature', type=float, default=1.0,
                        help='temperature for output logit')
    parser.add_argument('--max_len', type=int, default=30,
                        help='maximum length of decoding sequences')
    parser.add_argument('--target_word_path', type=str,
                        default='data/wordnet_all/shortlist/shortlist_valid.txt',
                        help='a path to a file containing list of words')
    parser.set_defaults(data_dir='data/wordnet_all/preprocess/',
                        state_size=300,
                        emb_size=300,
                        num_layers=2,
                        emb_keep_prob=0.75,
                        keep_prob=0.50,
                        sen_independent=True)
    args = parser.parse_args()
    opt = common_utils.update_opt(DecoderRNNLM.default_model_options(), parser)
    opt.input_emb_trainable = False
    common_utils.ensure_dir(opt.experiment_dir)
    if opt.save_config_file is not None:
        common_utils.save_config_file(opt)
    logger = common_utils.get_logger(os.path.join(opt.experiment_dir, opt.log_file))
    if opt.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info('Configurations:\n{}'.format(opt.toPretty()))
    logger.info('Loading output vocab...')
    out_vocab = Vocabulary.from_vocab_file(
        os.path.join(opt.data_dir, opt.output_vocab_file))
    training_exp_opt.data_kwargs.y_vocab = out_vocab
    opt.output_vocab_size = out_vocab.vocab_size
    if opt.decoding:
        opt.num_steps = 1
        with open(opt.target_word_path) as ifp:
            target_words = []
            for line in ifp:
                target_words.append(line.strip().split('\t')[0])
        decoding_exp_opt.word_list = target_words
        decode_dm.run(opt, decoding_exp_opt, logger)
    else:
        emb_path = os.path.join(opt.data_dir, opt.emb_pickle_file)
        logger.info('Loading embeddings from {}'.format(emb_path))
        with open(emb_path) as ifp:
            emb_values = cPickle.load(ifp)
            training_exp_opt.init_variables.append(
                ('{}/.*{}'.format('DM', 'emb'), emb_values))
        info = lm.run(opt, training_exp_opt, logger)
        logger.info('Perplexity: {}, Num tokens: {}'.format(
            np.exp(info.cost / info.num_words), info.num_words))
    logger.info('Total time: {}s'.format(time.time() - global_time))
