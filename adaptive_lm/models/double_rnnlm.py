import numpy as np
import tensorflow as tf
import rnnlm
from adaptive_lm.utils.common import LazyBunch
from adaptive_lm.models.basic_rnnlm import BasicRNNLM
from adaptive_lm.models.rnnlm_helper import BasicRNNHelper


class DoubleRNNLM(BasicRNNLM):
    """A RNNLM with 2 recurrent stack."""
    def __init__(self, opt, cell=None, helper=None, cell_top=None,
                 is_training=False):
        """Initialize BasicDecoder.

        Args:
            opt: a LazyBunch object.
            cell: (Optional) an instance of RNNCell. Default is BasicLSTMCell
            help: (Optional) an RNNHelper

        TODO:
            - Support more than 1 layer of each RNN.
        """
        self._opt = LazyBunch(opt)
        self._init_gate_bias = -2
        self._cell = cell
        if cell is None:
            # XXX: use manual dropout (only work with 1 layer)
            self._cell = rnnlm.get_rnn_cell(
                opt.state_size, opt.num_layers,
                opt.cell_type, 1.0)
        self._cell_top = cell_top
        if self._cell_top is None:
            self._cell_top = rnnlm.get_rnn_cell(
                opt.state_size, opt.num_layers, opt.cell_type, opt.keep_prob,
                is_training)
        self.helper = helper
        if self.helper is None:
            self.helper = BasicRNNHelper(opt)
        self.helper._model = self
        self.is_training = is_training

    def initialize(self):
        inputs, self._initial_state = super(DoubleRNNLM, self).initialize()
        self._initial_state_top = self._cell_top.zero_state(
            self._opt.batch_size, tf.float32)
        self._initial_states = LazyBunch(
            word_state=self._initial_state,
            top_state=self._initial_state_top)
        return inputs, self._initial_states

    def _gated_update(self, transform, extra, carried):
        dim = len(transform.get_shape())
        transform_dim = int(transform.get_shape()[-1])
        carried_dim = int(carried.get_shape()[-1])
        extra_dim = int(extra.get_shape()[-1])
        in_size = transform_dim + extra_dim
        out_size = carried_dim * 2
        gate_w = tf.get_variable("gate_w", [in_size, out_size])
        _arr = np.zeros((out_size))
        _arr[:] = self._init_gate_bias
        gate_b = tf.get_variable("gate_b", initializer=tf.constant(
            _arr, dtype=tf.float32))
        if dim == 3:
            z = self.helper.fancy_matmul(
                tf.concat([transform, extra], -1), gate_w) + gate_b
            t = tf.sigmoid(tf.slice(z, [0, 0, 0], [-1, -1, carried_dim]))
            h = tf.tanh(tf.slice(z, [0, 0, carried_dim], [-1, -1, -1]))
        else:
            z = tf.matmul(tf.concat([transform, extra], -1), gate_w) + gate_b
            t = tf.sigmoid(tf.slice(z, [0, 0], [-1, carried_dim]))
            h = tf.tanh(tf.slice(z, [0, carried_dim], [-1, -1]))
        self._transform_gate = t
        o = tf.multiply(h - carried, t) + carried
        self._final_rnn_output = o
        if self._opt.keep_prob < 1.0 and self.is_training:
            o = tf.nn.dropout(o, self._opt.keep_prob)
        return o

    # def _fake_gated_update(self, transform, extra, carried):
    #     carried_dim = int(carried.get_shape()[-1])
    #     self._transform_gate = tf.constant(np.ones((carried_dim)))
    #     self._final_rnn_output = extra
    #     return extra

    # def _fake_gated_update(self, transform, extra, carried):
    #     carried_dim = int(carried.get_shape()[-1])
    #     self._transform_gate = tf.constant(np.zeros((carried_dim)))
    #     o = extra
    #     self._final_rnn_output = o
    #     if self._opt.keep_prob < 1.0:
    #         o = tf.nn.dropout(o, self._opt.keep_prob)
    #     return o

    # def _attention_update(self, carried, extra):
    #     carried_dim = int(carried.get_shape()[-1])
    #     extra_dim = int(extra.get_shape()[-1])
    #     full_size = carried_dim + extra_dim
    #     gate_w = tf.get_variable("gate_w", [full_size, carried_dim + 1])
    #     # _bias_init = tf.constant(np.array([-1]), dtype=tf.float32)
    #     # gate_b = tf.get_variable("gate_b", initializer=_bias_init)
    #     gate_b = tf.get_variable("gate_b", [carried_dim + 1])
    #     z = self.helper.fancy_matmul(tf.concat([carried, extra], -1),
    #                                  gate_w) + gate_b
    #     t = tf.sigmoid(tf.slice(z, [0,0,0], [-1, -1, 1]))
    #     h = tf.tanh(tf.slice(z, [0,0, 1], [-1, -1, -1]))
    #     if self._opt.keep_prob < 1.0:
    #         h = tf.nn.dropout(
    #             h, keep_prob=self._opt.keep_prob, seed=self._dropout_seed)
    #     self._transform_gate = t
    #     o = tf.multiply(h, t) + tf.multiply(carried, (1-t))
    #     # if self._opt.keep_prob < 1.0:
    #     #     o = tf.nn.dropout(o, self._opt.keep_prob)
    #     return o

    def forward(self):
        _out, self._final_state = self.helper.unroll_rnn_cell(
            self._input_emb, self._seq_len,
            self._cell, self._initial_state)
        if isinstance(_out, list):
            _out, _ = self.helper._flat_rnn_outputs(_out)
        self._rnn_output = _out
        self._full_rnn_output = _out
        if self._opt.keep_prob < 1.0 and self.is_training:
            self._rnn_output = tf.nn.dropout(
                self._rnn_output, self._opt.keep_prob)
        _out, _state = self.helper.unroll_rnn_cell(
            self._rnn_output, self._seq_len,
            self._cell_top, self._initial_state_top, scope="rnn_top")
        if isinstance(_out, list):
            _out, _ = self.helper._flat_rnn_outputs(_out)
        self._rnn_top_output, self._final_state_top = _out, _state
        self._mixed_output = self._gated_update(
            self._rnn_output, self._rnn_top_output, self._full_rnn_output)
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
        nodes.rnn_top_outputs = m._rnn_top_output
        nodes.final_rnn_outputs = m._final_rnn_output
        return nodes
