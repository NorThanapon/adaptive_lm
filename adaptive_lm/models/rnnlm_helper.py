"""
TODO:
    - Fix class hiearachy
"""

import tensorflow as tf


class BasicRNNHelper(object):

    def __init__(self, opt):
        self.opt = opt

    def create_input_placeholder(self):
        """ Setup variables for network input (feeddict) """
        inputs = tf.placeholder(tf.int32,
                                [self.opt.batch_size, self.opt.num_steps],
                                name='inputs')
        seq_len = tf.placeholder(tf.int32,
                                 [self.opt.batch_size],
                                 name='seq_len')
        return inputs, seq_len

    def create_input_lookup(self, input):
        """ Create input embedding lookup """
        vocab_size = self.opt.get('input_vocab_size', self.opt.vocab_size)
        emb_var = tf.get_variable(
            "emb", [vocab_size, self.opt.emb_size],
            trainable=self.opt.input_emb_trainable)
        input_emb_var = tf.nn.embedding_lookup(emb_var, input)
        if self.opt.emb_keep_prob < 1.0:
            input_emb_var = tf.nn.dropout(input_emb_var,
                                          self.opt.emb_keep_prob)
        steps = int(input_emb_var.get_shape()[1])
        return emb_var, input_emb_var

    def unroll_rnn_cell(self, inputs, seq_len, cell,
                        initial_state, scope=None):
        """ Unroll RNNCell. """
        seq_len = None
        if self.opt.varied_len:
            seq_len = seq_len
        steps = int(inputs.get_shape()[1])
        # inputs = tf.unstack(inputs, num=steps, axis=1)
        # rnn_outputs, final_state = tf.contrib.rnn.static_rnn(
        #     cell, inputs, initial_state=initial_state,
        #     sequence_length=seq_len, scope=scope)
        # rnn_outputs = tf.stack([tf.reshape(_o, [self.opt.batch_size, 1, -1])
        #                         for _o in rnn_outputs], axis=1)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(
            cell, inputs, initial_state=initial_state,
            sequence_length=seq_len,
            scope=scope)
        return rnn_outputs, final_state

    def _flat_rnn_outputs(self, rnn_outputs):
        state_size = int(rnn_outputs[0].get_shape()[-1])
        return tf.reshape(tf.concat(rnn_outputs, 1),
                          [-1, state_size]), state_size

    def create_output(self, rnn_outputs, logit_weights=None):
        logits, temperature = self.create_output_logit(
            rnn_outputs, logit_weights)
        probs = tf.nn.softmax(logits)
        return logits, temperature, probs

    def fancy_matmul(self, mat3d, mat2d, transpose_2d=False):
        mat3d_dim = int(mat3d.get_shape()[-1])
        if transpose_2d:
            mat2d_dim = int(mat2d.get_shape()[0])
        else:
            mat2d_dim = int(mat2d.get_shape()[-1])
        output_shapes = tf.unstack(tf.shape(mat3d))
        output_shapes[-1] = mat2d_dim
        output_shape = tf.stack(output_shapes)
        flat_mat3d = tf.reshape(mat3d, [-1, mat3d_dim])
        outputs = tf.matmul(flat_mat3d, mat2d, transpose_b=transpose_2d)
        return tf.reshape(outputs, output_shape)

    def create_output_logit(self, features, logit_weights):
        """ Create softmax graph. """
        # features, softmax_size = self._flat_rnn_outputs(features)
        if self.opt.get('tie_input_output_emb', False):
            softmax_w = logit_weights
        else:
            softmax_size = features.get_shape()[-1]
            vocab_size = self.opt.get('output_vocab_size', self.opt.vocab_size)
            softmax_w = tf.get_variable("softmax_w",
                                        [vocab_size, softmax_size])
        softmax_b = tf.get_variable("softmax_b", softmax_w.get_shape()[0])
        logits = self.fancy_matmul(features, softmax_w, True) + softmax_b
        temperature = tf.placeholder_with_default(1.0, shape=None,
                                                  name="logit_temperature")
        # logits = tf.matmul(features, softmax_w, transpose_b=True) + softmax_b
        # return logits, temperature
        return logits / temperature, temperature

    def create_target_placeholder(self):
        """ create target placeholders """
        targets = tf.placeholder(tf.int32,
                                 [self.opt.batch_size, self.opt.num_steps],
                                 name='targets')
        weights = tf.placeholder(tf.float32,
                                 [self.opt.batch_size, self.opt.num_steps],
                                 name='weights')
        return targets, weights

    def create_xent_loss(self, logits, targets, weights):
        """ create cross entropy loss """
        # targets = tf.reshape(targets, [-1])
        # weights = tf.reshape(weights, [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=targets)
        sum_loss = tf.reduce_sum(tf.multiply(loss, weights))
        mean_loss = tf.div(sum_loss, tf.reduce_sum(weights) + 1e-12)
        return loss, mean_loss


class EmbDecoderRNNHelper(BasicRNNHelper):

    def __init__(self, opt):
        self.opt = opt

    def create_enc_input_placeholder(self):
        enc_inputs = tf.placeholder(
            tf.int32, [self.opt.batch_size, self.opt.num_steps],
            name='enc_inputs')
        return enc_inputs

    def create_encoder(self, enc_inputs, emb_var=None):
        if self.opt.get('tie_input_enc_emb', False):
            enc_emb = emb_var
        else:
            vocab_size = self.opt.get('enc_vocab_size', self.opt.vocab_size)
            emb_size = self.opt.get('enc_emb_size', self.opt.emb_size)
            trainable = self.opt.get(
                'enc_input_emb_trainable', self.opt.input_emb_trainable)
            enc_emb = tf.get_variable(
                "enc_emb", [vocab_size, emb_size], trainable=trainable)
        encoder = tf.nn.embedding_lookup(enc_emb, enc_inputs)
        if self.opt.emb_keep_prob < 1.0:
            encoder = tf.nn.dropout(encoder, self.opt.emb_keep_prob)
        # steps = int(encoder.get_shape()[1])
        # rearrange to fix rnn output
        # encoder = [tf.squeeze(_x, [1]) for _x in tf.split(encoder, steps, 1)]
        return encoder

    def create_enc_dec_mixer(self, enc_outputs, dec_outputs):
        """ Combine encoder and decoder into a feature for output
            Args:
                enc_outputs: A tensor batch x step x dim
                dec_outputs: A tensor batch x step x dim
            Returns:
                (feature tensor(batch*steps, hidden), hidden size)
        """
        enc_size = int(enc_outputs.get_shape()[-1])
        dec_size = int(dec_outputs.get_shape()[-1])
        full_size = enc_size + dec_size
        enc_dec = tf.concat([enc_outputs, dec_outputs], -1)
        zr_w = tf.get_variable("att_zr_w", [full_size, full_size])
        zr_b = tf.get_variable("att_zr_b", [full_size])
        zr = tf.sigmoid(self.fancy_matmul(enc_dec, zr_w) + zr_b)
        z = tf.slice(zr, [0, 0, 0], [-1, -1, dec_size],
                     name="att_z_gate")
        r = tf.slice(zr, [0, 0, dec_size], [-1, -1, -1],
                     name="att_r_gate")
        att_enc = tf.multiply(enc_outputs, r)
        att_enc_dec = tf.concat([att_enc, dec_outputs], -1)
        h_w = tf.get_variable("att_h_w",
                              [full_size, dec_size])
        h_b = tf.get_variable("att_h_b", [dec_size])
        h = tf.tanh(self.fancy_matmul(att_enc_dec, h_w) + h_b)
        outputs = tf.multiply((1-z), dec_outputs) + tf.multiply(z, h)
        return outputs, dec_size
