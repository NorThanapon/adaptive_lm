import unittest
import tensorflow as tf
from adaptive_lm.models.basic_rnnlm import BasicRNNLM
from adaptive_lm.models.basic_rnnlm import DecoderRNNLM
from adaptive_lm.models.double_rnnlm import DoubleRNNLM

class RNNLMTest(unittest.TestCase):

    def assertStateSize(self, opt, states):
        if isinstance(states, dict):
            for k in states:
                self.assertStateSize(opt, states[k])
        else:
            for state in states:
                for substate in state:
                    self.assertEqual(substate.get_shape(),
                                     (opt.batch_size, opt.state_size))

    def assertPlaceholderSize(self, opt, var):
        self.assertEqual(var.get_shape(), (opt.batch_size, opt.num_steps))

    def test_smoke_basic_rnn(self):
        tf.reset_default_graph()
        opt = BasicRNNLM.default_model_options()
        opt.num_layers = 2
        opt.emb_size = 100
        opt.state_size = 100
        opt.vocab_size = 50
        m = BasicRNNLM(opt)
        inputs, init_state = m.initialize()
        self.assertPlaceholderSize(opt, inputs.inputs)
        self.assertEqual(inputs.seq_len.get_shape(), (opt.batch_size, ))
        self.assertStateSize(opt, init_state)
        outputs, final_state = m.forward()
        self.assertEqual(outputs.rnn_outputs.get_shape(),
                         (opt.batch_size, opt.num_steps, opt.state_size))
        targets, losses = m.loss()
        self.assertPlaceholderSize(opt, targets.targets)
        self.assertPlaceholderSize(opt, targets.weights)
        self.assertPlaceholderSize(opt, losses.token_loss)
        self.assertEqual(losses.loss.get_shape(), ())
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print(total_parameters)

    def test_smoke_decoder_rnn(self):
        tf.reset_default_graph()
        opt = BasicRNNLM.default_model_options()
        m = DecoderRNNLM(opt)
        inputs, init_state = m.initialize()
        self.assertPlaceholderSize(opt, inputs.inputs)
        self.assertPlaceholderSize(opt, inputs.enc_inputs)
        self.assertEqual(inputs.seq_len.get_shape(), (opt.batch_size, ))
        self.assertStateSize(opt, init_state)
        outputs, final_state = m.forward()
        self.assertEqual(outputs.rnn_outputs.get_shape(),
                         (opt.batch_size, opt.num_steps, opt.state_size))
        targets, losses = m.loss()
        self.assertPlaceholderSize(opt, targets.targets)
        self.assertPlaceholderSize(opt, targets.weights)
        self.assertPlaceholderSize(opt, losses.token_loss)
        self.assertEqual(losses.loss.get_shape(), ())

    def test_smoke_double_rnn(self):
        tf.reset_default_graph()
        opt = BasicRNNLM.default_model_options()
        opt.num_layers = 1
        opt.emb_size = 100
        opt.state_size = 88
        opt.vocab_size = 50
        m = DoubleRNNLM(opt)
        inputs, init_states = m.initialize()
        self.assertPlaceholderSize(opt, inputs.inputs)
        self.assertEqual(inputs.seq_len.get_shape(), (opt.batch_size, ))
        self.assertStateSize(opt, init_states)
        outputs, final_state = m.forward()
        self.assertEqual(outputs.rnn_outputs.get_shape(),
                         (opt.batch_size, opt.num_steps, opt.state_size))
        self.assertEqual(outputs.rnn_top_outputs.get_shape(),
                         (opt.batch_size, opt.num_steps, opt.state_size))
        targets, losses = m.loss()
        self.assertPlaceholderSize(opt, targets.targets)
        self.assertPlaceholderSize(opt, targets.weights)
        self.assertPlaceholderSize(opt, losses.token_loss)
        self.assertEqual(losses.loss.get_shape(), ())
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print(total_parameters)


if __name__ == '__main__':
    unittest.main()
