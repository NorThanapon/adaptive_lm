import numpy as np
import tensorflow as tf
import rnnlm
from adaptive_lm.utils.common import LazyBunch
from adaptive_lm.models.basic_rnnlm import BasicRNNLM
from adaptive_lm.models.rnnlm_helper import BasicRNNHelper

class DoubleRNNLM(BasicRNNLM):
    """A RNNLM with 2 recurrent stack."""
    def __init__(self, opt, cell=None, helper=None, cell_top=None):
        """Initialize BasicDecoder.

        Args:
            opt: a LazyBunch object.
            cell: (Optional) an instance of RNNCell. Default is BasicLSTMCell
            help: (Optional) an DecoderHelper
        """
        if helper is None:
            helper = BasicRNNHelper(opt)
        super(DoubleRNNLM, self).__init__(opt, cell, helper)
        self._cell_top = cell_top
        if self._cell_top is None:
            self._cell_top = rnnlm.get_rnn_cell(
                opt.state_size, opt.num_layers,
                opt.cell_type, opt.keep_prob)

    def initialize(self):
        inputs, self._initial_state = super(DoubleRNNLM, self).initialize()
        self._initial_state_top = self._cell_top.zero_state(
            self._opt.batch_size, tf.float32)
        self._initial_states = LazyBunch(
            word_state=self._initial_state,
            top_state=self._initial_state_top)
        return inputs, self._initial_states

    def _gated_update(self, carried, extra):
        carried_dim = int(carried.get_shape()[-1])
        extra_dim = int(extra.get_shape()[-1])
        full_size = carried_dim + extra_dim
        gate_w = tf.get_variable("gate_w", [full_size, carried_dim * 2])
        # _arr = np.zeros((full_size))
        # _arr[:] = -1
        # gate_b = tf.get_variable("gate_b", initializer=tf.constant(
        #     _arr, dtype=tf.float32))
        gate_b = tf.get_variable("gate_b", shape=[full_size], dtype=tf.float32)
        z = self.helper.fancy_matmul(tf.concat([carried, extra], -1),
                                     gate_w) + gate_b
        t = tf.sigmoid(tf.slice(z, [0,0,0], [-1, -1, carried_dim]))
        h = tf.tanh(tf.slice(z, [0,0, carried_dim], [-1, -1, -1]))
        self._transform_gate = t
        o = tf.multiply(h, t) + tf.multiply(carried, (1-t))
        if self._opt.keep_prob < 1.0:
            o = tf.nn.dropout(o, self._opt.keep_prob)
        return o

    def _attention_update(self, carried, extra):
        carried_dim = int(carried.get_shape()[-1])
        extra_dim = int(extra.get_shape()[-1])
        full_size = carried_dim + extra_dim
        gate_w = tf.get_variable("gate_w", [full_size, carried_dim + 1])
        # _bias_init = tf.constant(np.array([-1]), dtype=tf.float32)
        # gate_b = tf.get_variable("gate_b", initializer=_bias_init)
        gate_b = tf.get_variable("gate_b", [carried_dim + 1])
        z = self.helper.fancy_matmul(tf.concat([carried, extra], -1),
                                     gate_w) + gate_b
        t = tf.sigmoid(tf.slice(z, [0,0,0], [-1, -1, 1]))
        h = tf.tanh(tf.slice(z, [0,0, 1], [-1, -1, -1]))
        self._transform_gate = t
        o = tf.multiply(h, t) + tf.multiply(carried, (1-t))
        if self._opt.keep_prob < 1.0:
            o = tf.nn.dropout(o, self._opt.keep_prob)
        return o


    def forward(self):
        self._rnn_output, self._final_state = self.helper.unroll_rnn_cell(
            self._input_emb, self._seq_len,
            self._cell, self._initial_state)
        self._rnn_top_output, self._final_state_top = self.helper.unroll_rnn_cell(
            self._rnn_output, self._seq_len,
            self._cell_top, self._initial_state_top, scope="rnn_top")
        # self._rnn_top_output, self._final_state_top = self.helper.unroll_rnn_cell(
        #     self._input_emb, self._seq_len,
        #     self._cell_top, self._initial_state_top, scope="rnn_top")
        self._mixed_output = self._gated_update(self._rnn_output,
                                                    self._rnn_top_output)
        self._logit, self._temperature, self._prob = self.helper.create_output(
            self._mixed_output, self._emb)
        self._final_states = LazyBunch(word_state=self._final_state,
                                       top_state=self._final_state_top)
        outputs = LazyBunch(rnn_outputs=self._rnn_output,
                            rnn_top_outputs=self._rnn_top_output,
                            distributions=self._prob)
        return outputs, self._final_states

    @staticmethod
    def build_full_model_graph(m):
        nodes = BasicRNNLM.build_full_model_graph(m)
        nodes.transform_gates = m._transform_gate
        return nodes
