import argparse
import logging
import time
import os
import numpy as np
import cPickle

from adaptive_lm.models.rnnlm_helper import BasicRNNHelper
from adaptive_lm.models.basic_rnnlm import BasicRNNLM
from adaptive_lm.models.double_rnnlm import DoubleRNNLM
from adaptive_lm.utils import common as common_utils
from adaptive_lm.experiments import lm

def build_test_fn(m):
    nodes = DoubleRNNLM.build_full_model_graph(m)
    nodes.fetch.collect = common_utils.LazyBunch(
        target=nodes.targets.targets,
        weight=nodes.targets.weights,
        token_loss=nodes.losses.token_loss,
        transform_gates=nodes.transform_gates,
        rnn_top_outputs=nodes.outputs.rnn_top_outputs,
        rnn_outputs=nodes.outputs.rnn_outputs,)
    return nodes

training_exp_opt = common_utils.LazyBunch(
    resume = 'latest_lm',
    best = 'best_lm',
    splits = ['train', 'valid'],
    run_split = 'valid',
    iterator_cls = None, # use default option
    model_scope = 'LM',
    model_helper_cls = BasicRNNHelper,
    model_cls = DoubleRNNLM,
    build_train_fn = DoubleRNNLM.build_full_model_graph,
    build_test_fn = DoubleRNNLM.build_full_model_graph,
    training = True
)

testing_exp_opt = common_utils.LazyBunch(training_exp_opt,
    splits = ['valid'],
    run_split = 'valid',
    training = False
)

if __name__ == '__main__':
    global_time = time.time()
    parser = common_utils.get_common_argparse()
    parser.add_argument('--out_token_loss_file', type=str,
                        default=None,
                        help='output token loss to the file')
    args = parser.parse_args()
    opt = common_utils.update_opt(BasicRNNLM.default_model_options(), parser)
    common_utils.ensure_dir(opt.experiment_dir)
    if opt.save_config_file is not None:
        common_utils.save_config_file(opt)
    logger = common_utils.get_logger(os.path.join(opt.experiment_dir, opt.log_file))
    if opt.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info('Configurations:\n{}'.format(opt.toPretty()))
    if opt.training:
        info = lm.run(opt, training_exp_opt, logger)
    else:
        if opt.out_token_loss_file is not None:
            testing_exp_opt.build_test_fn = build_test_fn
            token_loss_path = os.path.join(opt.experiment_dir, opt.out_token_loss_file)
            token_loss_ofp = open(token_loss_path, 'w')
            output_data = []
            def write_token_loss(collect):
                tokens = np.reshape(collect.target, [-1])
                weights = np.reshape(collect.weight, [-1])
                losses = np.reshape(collect.token_loss, [-1])
                gates = np.reshape(collect.transform_gates, [-1, opt.state_size])
                rnn_outputs = np.reshape(collect.rnn_outputs, [-1, opt.state_size])
                top_outputs = np.reshape(collect.rnn_top_outputs, [-1, opt.state_size])
                for i in range(len(tokens)):
                    if weights[i] > 0:
                        token_loss_ofp.write("{}\t{}\t{}\n".format(
                            tokens[i], losses[i], gates[i].mean()))
                        output_data.append([tokens[i], rnn_outputs[i],
                                            top_outputs[i], gates[i]])
            testing_exp_opt.collect_fn = write_token_loss
        info = lm.run(opt, testing_exp_opt, logger)
        with open('tmp.cpickle', 'w') as ofp:
            cPickle.dump(output_data, ofp)
        if opt.out_token_loss_file is not None:
            token_loss_ofp.close()
    logger.info('Perplexity: {}, Num tokens: {}'.format(
        np.exp(info.cost / info.num_words), info.num_words))
    logger.info('Total time: {}s'.format(time.time() - global_time))
