import tensorflow as tf
import rnnlm
from adaptive_lm.utils.common import LazyBunch
from rnnlm_helper import BasicRNNHelper
from rnnlm_helper import EmbDecoderRNNHelper


class BasicRNNLM(rnnlm.RNNLM):
    """A basic RNNLM."""

    def __init__(self, opt, cell=None, helper=None, is_training=False):
        """Initialize BasicDecoder.

        Args:
            opt: a LazyBunch object.
            cell: (Optional) an instance of RNNCell. Default is BasicLSTMCell
            help: (Optional) an RNNHelper
        """
        self._opt = LazyBunch(opt)
        self._cell = cell
        if cell is None:
            self._cell = rnnlm.get_rnn_cell(
                opt.state_size, opt.num_layers, opt.cell_type, opt.keep_prob,
                is_training)
        self.helper = helper
        if self.helper is None:
            self.helper = BasicRNNHelper(opt)
        self.helper._model = self
        self.is_training = is_training

    def batch_size(self):
        return self._opt.batch_size

    def num_steps(self):
        return self._opt.num_steps

    def initialize(self):
        self._input, self._seq_len = self.helper.create_input_placeholder()
        self._emb, self._input_emb = self.helper.create_input_lookup(
            self._input)
        self._initial_state = self._cell.zero_state(
            self._opt.batch_size, tf.float32)
        inputs = LazyBunch(inputs=self._input, seq_len=self._seq_len)
        return inputs, self._initial_state

    def forward(self):
        self._rnn_output, self._final_state = self.helper.unroll_rnn_cell(
            self._input_emb, self._seq_len,
            self._cell, self._initial_state)
        self._logit, self._temperature, self._prob = self.helper.create_output(
            self._rnn_output, self._emb)
        outputs = LazyBunch(rnn_outputs=self._rnn_output,
                            distributions=self._prob)
        return outputs, self._final_state

    def loss(self):
        self._target, self._weight = self.helper.create_target_placeholder()
        self._token_loss, self._training_loss, self._mean_loss \
            = self.helper.create_xent_loss(
                self._logit, self._target, self._weight)
        target_holder = LazyBunch(targets=self._target, weights=self._weight)
        losses = LazyBunch(
            token_loss=self._token_loss,
            mean_loss=self._mean_loss,
            training_loss=self._training_loss)
        return target_holder, losses

    @staticmethod
    def build_full_model_graph(m):
        inputs, init_state = m.initialize()
        outputs, final_state = m.forward()
        targets, losses = m.loss()
        feed = LazyBunch(
            inputs=inputs.inputs,
            targets=targets.targets,
            weights=targets.weights,
            lengths=inputs.seq_len)
        fetch = LazyBunch(
            final_state=final_state,
            eval_loss=losses.mean_loss,
            training_loss=losses.training_loss
        )
        return LazyBunch(
            inputs=inputs, init_state=init_state, outputs=outputs,
            final_state=final_state, targets=targets, losses=losses,
            temperature=m._temperature, feed=feed, fetch=fetch)

    @staticmethod
    def default_model_options():
        return LazyBunch(
            batch_size=32,
            num_steps=10,
            num_layers=1,
            varied_len=False,
            emb_keep_prob=0.9,
            keep_prob=0.75,
            vocab_size=10000,
            emb_size=100,
            state_size=100,
            input_emb_trainable=True
        )


class AtomicDiscourseRNNLM(BasicRNNLM):

    def _compute_discourse_vector(self, rnn_output):
        state_size = self._opt.state_size
        discourse_size = self._opt.discourse_size
        dim = len(rnn_output.get_shape())
        self._discourse_emb_var = tf.get_variable(
            "discourse_w", [discourse_size, state_size])
        self._discourse_b = tf.get_variable(
            "discourse_b", [discourse_size])
        if dim == 2:
            alpha = tf.nn.relu(tf.matmul(rnn_output, self._discourse_emb_var,
                                         transpose_b=True) + self._discourse_b)
            # alpha = tf.nn.dropout(alpha, 0.75)
            top = 20
            topk = tf.nn.top_k(alpha, top)
            top_values = tf.reshape(
                tf.div(topk.values, tf.reduce_sum(
                    topk.values, axis=1, keep_dims=True)), [-1, top, 1])
            # top_values = tf.reshape(
            #     tf.nn.softmax(topk.values), [-1, top, 1])
            self.top_alpha = top_values
            top_dc = tf.gather(self._discourse_emb_var, topk.indices)
            o = tf.reduce_sum(tf.multiply(top_dc, top_values), axis=1)
            o = tf.layers.batch_normalization(o, training=self.is_training)
            if self._opt.keep_prob < 1.0 and self.is_training:
                o = tf.nn.dropout(o, self._opt.keep_prob)
            return o

    def forward(self):
        self._rnn_output, self._final_state = self.helper.unroll_rnn_cell(
            self._input_emb, self._seq_len,
            self._cell, self._initial_state)
        if isinstance(self._rnn_output, list):
            self._rnn_output, _ = self.helper._flat_rnn_outputs(
                self._rnn_output)
        self._discourse = self._compute_discourse_vector(self._rnn_output)
        self._logit, self._temperature, self._prob = self.helper.create_output(
            self._discourse, self._emb)
        outputs = LazyBunch(rnn_outputs=self._rnn_output,
                            distributions=self._prob)
        return outputs, self._final_state

    # def loss(self):
    #     target_holder, losses = super(AtomicDiscourseRNNLM, self).loss()
    #     l1_loss = tf.norm(self.top_alpha, ord=1) * 1e-2
    #     losses.training_loss = losses.training_loss + l1_loss
    #     return target_holder, losses


class DecoderRNNLM(BasicRNNLM):
    """A decoder RNNLM."""

    def __init__(self, opt, cell=None, helper=None, is_training=False):
        """Initialize BasicDecoder.

        Args:
            opt: a LazyBunch object.
            cell: (Optional) an instance of RNNCell. Default is BasicLSTMCell
            help: (Optional) an DecoderHelper
        """
        if helper is None:
            helper = EmbDecoderRNNHelper(opt)
        super(DecoderRNNLM, self).__init__(opt, cell, helper, is_training)

    def initialize(self):
        inputs, self._initial_state = super(DecoderRNNLM, self).initialize()
        self._enc_input = self.helper.create_enc_input_placeholder()
        inputs.enc_inputs = self._enc_input
        return inputs, self._initial_state

    def forward(self):
        self._rnn_output, self._final_state = self.helper.unroll_rnn_cell(
            self._input_emb, self._seq_len,
            self._cell, self._initial_state)
        self._enc_output = self.helper.create_encoder(
            self._enc_input, self._emb)
        mixed_output, _ = self.helper.create_enc_dec_mixer(
            self._enc_output, self._rnn_output)
        self._logit, self._temperature, self._prob = self.helper.create_output(
            mixed_output, self._emb)
        outputs = LazyBunch(rnn_outputs=self._rnn_output,
                            enc_outputs=self._enc_output,
                            distributions=self._prob)
        return outputs, self._final_state

    @staticmethod
    def build_full_model_graph(m):
        nodes = BasicRNNLM.build_full_model_graph(m)
        nodes.feed.enc_inputs = nodes.inputs.enc_inputs
        return nodes

    @staticmethod
    def default_model_options():
        opt = BasicRNNLM.default_model_options()
        opt.emb_size = 300
        opt.state_size = 300
        opt.emb_keep_prob = 0.75
        opt.keep_prob = 0.50
        opt.tie_input_enc_emb = True
        return opt
