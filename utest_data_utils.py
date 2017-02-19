import data_utils
import exp_utils
import common_utils
import numpy as np
import cPickle
import logging

logger = common_utils.get_logger()
logger.setLevel(logging.DEBUG)
# with open('data/ptb_defs/wordnet/preprocess/t_features.pickle') as ifp:
#     idx, data = cPickle.load(ifp)
# vocab = data_utils.Vocabulary.from_vocab_file('data/ptb/preprocess/vocab.txt')
# loader = data_utils.TokenFeatureIterator(vocab, 'data/ptb/preprocess/valid.jsonl',
#                                          t_feature_idx=idx, t_features=data)
# loader.init_batch(2, 5)
# x,y,w,l,r = loader.next_batch()


# with open('data/ptb_defs/wordnet/preprocess/t_features.pickle') as ifp:
#     idx, data = cPickle.load(ifp)
# vocab = data_utils.Vocabulary.from_vocab_file('data/ptb_defs/preprocess/vocab.txt')
# loader = data_utils.DefIterator(vocab, 'data/ptb_defs/preprocess/train.jsonl',
#                                          t_feature_idx=idx, t_features=data)
# loader.init_batch(2)
# x,y,w,l,r = loader.next_batch()

# vocab2 = data_utils.Vocabulary.from_vocab_file('data/common_defs_v1.2/shortlist/common_defs_ptb_shortlist.txt')
####################################################
# Test DataIterator
####################################################

# vocab = data_utils.Vocabulary.from_vocab_file('data/ptb/preprocess/vocab.txt')
# vocab2 = data_utils.Vocabulary.from_vocab_file('data/common_defs_v1.2/wordnet/shortlist/shortlist_all_ptb.txt')
# loader = data_utils.DataIterator(vocab, 'data/ptb/preprocess/valid.jsonl',
#                                  shuffle_data=False, y_vocab=vocab2)
# vocab3 = vocab
# vocab = vocab2
# tokens = 0
# batch = 32
# rho = 30
# data = [[] for _ in range(batch)]
# for step, (x, y, w, l, r) in enumerate(loader.iterate_epoch(batch, rho)):
#     tokens += w.sum()
#     for i in range(batch):
#         for j in range(rho):
#             if y[i, j] != vocab.eos_id and w[i, j] == 1:
#                 data[i].append(vocab.i2w(y[i,j]))
#             elif y[i, j] == vocab.eos_id and w[i, j] == 1:
#                 data[i].append("\n")
# print('Num tokens from iterator: {} (should be {})'.format(
#     int(tokens), len(loader._data) - 1))
# with open('tmp_valid.txt', 'w') as ofp:
#     for i in range(batch):
#         ofp.write(' ')
#         ofp.write(' '.join(data[i]))

####################################################
# Test SentenceIterator
####################################################

# vocab = data_utils.Vocabulary.from_vocab_file('data/ptb-100/drop_0/preprocess/vocab.txt')
# loader = data_utils.SentenceIterator(vocab, 'data/ptb/preprocess/valid.jsonl',
#                                      shuffle_data=False)
# sen = 0
# tokens = 0
# batch = 32
# rho = 10
# data = [[] for _ in range(batch)]
# for step, (x, y, w, l, r) in enumerate(loader.iterate_epoch(batch, rho)):
#     if loader._new_sentence_set:
#         sen += len(r.nonzero()[0])
#     tokens += w.sum()
#     for i in range(batch):
#         for j in range(rho):
#             if y[i, j] != vocab.eos_id and w[i, j] == 1:
#                 data[i].append(vocab.i2w(y[i,j]))
#             elif y[i, j] == vocab.eos_id and w[i, j] == 1:
#                 data[i].append("\n")
# print('Num sentences from iterator: {} (should be {})'.format(
#     sen, len(loader._sen_idx)))
# print('Num tokens from iterator: {} (should be {})'.format(
#     int(tokens), len(loader._data) - len(loader._sen_idx) - 1))
# with open('tmp_valid.txt', 'w') as ofp:
#     for i in range(batch):
#         ofp.write(' ')
#         ofp.write(' '.join(data[i]))

####################################################
# Test SenLabelIterator
####################################################

# import codecs
# vocab = data_utils.Vocabulary.from_vocab_file('data/common_defs_v1.2/preprocess/vocab.txt')
# loader = data_utils.SenLabelIterator(vocab, 'data/common_defs_v1.2/preprocess/valid.jsonl',
#                                      shuffle_data=False)
# sen = 0
# tokens = 0
# batch = 1
# rho = 30
# loader.init_batch(batch, rho)
# data = [[] for _ in range(batch)]
# labels = [[] for _ in range(batch)]
# for step, (x, y, w, l, r) in enumerate(loader.iterate_epoch(batch, rho)):
#     if loader._new_sentence_set:
#         sen += len(r.nonzero()[0])
#         for i in range(batch):
#             data[i].append([])
#             labels[i].append(vocab.i2w(l[0,0]))
#     tokens += w.sum()
#     for i in range(batch):
#         for j in range(rho):
#             if y[i, j] != vocab.eos_id and w[i, j] == 1:
#                 data[i][-1].append(vocab.i2w(y[i,j]))
#             elif y[i, j] == vocab.eos_id and w[i, j] == 1:
#                 data[i][-1].append("\n")
# print('Num sentences from iterator: {} (should be {})'.format(
#     sen, len(loader._sen_idx)))
# print('Num tokens from iterator: {} (should be {})'.format(
#     int(tokens), len(loader._data) - 2*len(loader._sen_idx) - 1))
# with codecs.open('tmp_valid.txt', 'w', 'utf-8') as ofp:
#     for i in range(batch):
#         for j, d in enumerate(data[i]):
#             ofp.write(u'{}\t{}'.format(labels[i][j], ' '.join(d)))
# loader.init_batch(batch, rho)

# vocab = data_utils.Vocabulary.from_vocab_file('data/common_defs_v1.2/preprocess/vocab.txt')
# loader = data_utils.DefIterator(vocab, 'data/common_defs_v1.2/preprocess/valid.jsonl')#, x_vocab=vocab2)
# loader.init_batch(batch)

####################################################
# Test exp_utils.load_datasets()
####################################################
# vocab = data_utils.Vocabulary.from_vocab_file('data/common_defs_v1.2/preprocess/vocab.txt')
# opt = common_utils.Bunch.default_model_options()
# parser = common_utils.get_common_argparse()
# args = parser.parse_args()
# opt.update_from_ns(args)
# data, vocab = exp_utils.load_datasets(opt, x_vocab=vocab)

####################################################
# Test OneToManyMap
####################################################

m = data_utils.OneToManyMap.from_map_file('experiments/multi-softmax/ptb/1to2.map.txt')
y = np.array([[15]])
mask = m.create_map_mask(y)
for i in m._map[15]:
    print(mask[0,i])
