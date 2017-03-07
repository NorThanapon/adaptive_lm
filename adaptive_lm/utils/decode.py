from adaptive_lm.utils import run as run_utils
from adaptive_lm.utils import common as common_utils
import numpy as np

class Emb2SeqDecoder(object):

    def __init__(self, sess, model, in_vocab, out_vocab, opt):
        self.sess = sess
        self.model = model
        self.opt = opt
        self.in_vocab = in_vocab
        self.out_vocab = out_vocab

    def select_words(self, distributions):
        batch_size = distributions.shape[0]
        words = []
        scores = np.zeros((batch_size, ))
        choices = np.arange(distributions.shape[-1])
        for b in range(batch_size):
            if self.opt.sampling:
                idx = np.random.choice(choices, p=distributions[b, 0, :])
            else:
                idx = distributions[b, 0, :].argmax()
            words.append(self.out_vocab.i2w(idx))
            scores[b] = distributions[b, 0, idx]
        return words, scores

    def collect_definitions(self, inputs, definitions):
        for b in range(inputs.shape[0]):
            definitions[b].append(self.in_vocab.i2w(inputs[b, -1]))
        return definitions

    def put_words_to_batch(self, batch, words, set_enc_input=False):
        for b, w in enumerate(words):
            batch.inputs[b,0] = self.in_vocab.w2i(w)
            if set_enc_input:
                batch.enc_inputs[b,0] = self.in_vocab.w2i(w)
        return batch

    def seed_states(self, batch, seeds):
        zero_state = self.sess.run(self.model.init_state)
        state = zero_state
        num_batches = batch.inputs.shape[0]
        definitions = [[] for _ in range(num_batches)]
        for i, words in enumerate(seeds):
            self.put_words_to_batch(batch, words, i==0)
            self.collect_definitions(batch.inputs, definitions)
            feed_dict = run_utils.map_feeddict(batch, self.model.feed)
            feed_dict = run_utils.feed_state(
                feed_dict, self.model.init_state, state)
            feed_dict[self.model.temperature] = self.opt.logit_temperature
            result = self.sess.run(self.model.fetch, feed_dict)
            state = result.final_state
        return result, definitions

    def is_end(self, definitions):
        for b_def in definitions:
            if b_def[-1] != self.in_vocab.eos:
                return False
        return True

    def decode_batch(self, batch, seeds):
        self.model.fetch = common_utils.LazyBunch(
            final_state=self.model.fetch.final_state,
            distributions=self.model.outputs.distributions)
        result, definitions = self.seed_states(batch, seeds)
        scores = np.zeros((batch.inputs.shape[0], self.opt.max_len))
        for i in range(self.opt.max_len):
            out_words, s = self.select_words(result.distributions)
            scores[:, i] = s[:]
            self.put_words_to_batch(batch, out_words)
            self.collect_definitions(batch.inputs, definitions)
            if self.is_end(definitions):
                break
            feed_dict = run_utils.map_feeddict(batch, self.model.feed)
            feed_dict = run_utils.feed_state(
                feed_dict, self.model.init_state, result.final_state)
            feed_dict[self.model.temperature] = self.opt.logit_temperature
            result = self.sess.run(self.model.fetch, feed_dict)
        return definitions, scores

    def batch_words(self, words, batch_size):
        batches_of_words = []
        b_words = []
        for w in words:
            for _ in range(self.opt.num_samples):
                if len(b_words) == batch_size:
                    batches_of_words.append(b_words)
                    b_words = []
                b_words.append(w)
        if len(b_words) != 0:
            for _ in range(batch_size - len(b_words)):
                b_words.append(b_words[-1])
            batches_of_words.append(b_words)
        return batches_of_words

    def decode(self, batch, inputs):
        """
        Args:
            batch: a template batch
            inputs: a list of words
        """
        batch_size = batch.inputs.shape[0]
        batches_of_words = self.batch_words(inputs, batch_size)
        delimiters = ["<def>"] * batch_size
        definitions = []
        scores = np.zeros((len(batches_of_words)*batch_size, self.opt.max_len))
        for b, b_words in enumerate(batches_of_words):
            b_defs, b_scores = self.decode_batch(batch, [b_words, delimiters])
            definitions.extend(b_defs)
            scores[b*batch_size:(b+1)*batch_size, :] = b_scores
        return definitions, scores
