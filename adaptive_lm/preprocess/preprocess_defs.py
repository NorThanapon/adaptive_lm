import os
import argparse
import json
import operator
import nltk

# sos_symbol = '<s>'
eos_symbol = '</s>'
unk_symbol = '<unk>'
def_symbol = '<def>'

parser = argparse.ArgumentParser()
parser.add_argument("text_dir",
                    help="Definition directory.")

parser.add_argument("--restrict_vocab", help="restrict vocab file path",
                    default=None)
parser.add_argument("--source_indices", help="source of the definitions",
                    default='2', type=str)
parser.add_argument("--stopword_file", help="stopword file path",
                    default='stopwords.txt')
parser.add_argument("--bow_vocab_size", help="Number of vocab in BOW features",
                    default=100)
parser.add_argument("--max_def_len",
                    help="remove definitions that is longer than this number.",
                    default=100)

parser.add_argument('--only_train', dest='only_train', action='store_true')
parser.set_defaults(only_train=False)

args = parser.parse_args()
args.source_indices = [int(sidx) for sidx in args.source_indices.split(',')]


def get_source(parts, indices):
    source = []
    for i in indices:
        source.append(parts[i])
    return u'.'.join(source)


limit_vocab = None
if args.restrict_vocab is not None:
    limit_vocab = set()
    with open(args.restrict_vocab) as ifp:
        for line in ifp:
            limit_vocab.add(line.strip())

splits = ('train', 'valid', 'test')
if args.only_train:
    splits = ('train',)
ofps = {}
for s in splits:
    ofps[s] = open(os.path.join(args.text_dir,
                                'preprocess/{}.jsonl'.format(s)), 'w')
w_count = {
    # sos_symbol:0,
    eos_symbol: 0,
    unk_symbol: 0,
    def_symbol: 0,
}
w_def_count = w_count.copy()
w_low_count = w_count.copy()
args.max_def_len = int(args.max_def_len)
for s in splits:
    f = os.path.join(args.text_dir, '{}.tsv'.format(s))
    pf = os.path.join(args.text_dir, '{}.txt'.format(s))
    with open(f) as ifp, open(pf, 'w') as pfp:
        for line in ifp:
            parts = line.lower().strip().split('\t')
            def_tokens = nltk.word_tokenize(parts[-1])
            if limit_vocab is not None:
                # ignore word not in vocab
                if parts[0] not in limit_vocab:
                    continue
                for i in range(len(def_tokens)):
                    if def_tokens[i] not in limit_vocab:
                        # replace with unk token
                        def_tokens[i] = unk_symbol
            if len(def_tokens) > args.max_def_len:
                continue
            for t in def_tokens:
                w_def_count[t] = w_def_count.get(t, 0) + 1
            parts[-1] = ' '.join(def_tokens)
            data = {'meta': {'word': parts[0], 'pos': parts[1],
                             'src': get_source(parts, args.source_indices)},
                    'key': parts[0],
                    'lines': [' '.join([parts[0], def_symbol, parts[-1]])]}
            # w_count[sos_symbol] += 1
            w_count[eos_symbol] += 1
            # w_low_count[sos_symbol] += 1
            w_low_count[eos_symbol] += 1
            w_def_count[def_symbol] += 1
            w_def_count[eos_symbol] += 1
            for token in data['lines'][0].split():
                w_count[token] = w_count.get(token, 0) + 1
                l_token = token.lower()
                w_low_count[l_token] = w_low_count.get(l_token, 0) + 1
            ofp = ofps[s]
            json.dump(obj=data, fp=ofp)
            ofp.write('\n')
            pfp.write('\t'.join(parts))
            pfp.write('\n')
for ofp in ofps:
    ofps[ofp].close()

w_count = sorted(w_count.items(), key=operator.itemgetter(1), reverse=True)

vocab_filepath = os.path.join(args.text_dir, 'preprocess/vocab.txt')
with open(vocab_filepath, 'w') as ofp:
    for w in w_count:
        ofp.write('{}\t{}\n'.format(w[0], w[1]))

w_def_count = sorted(w_def_count.items(),
                     key=operator.itemgetter(1), reverse=True)
vocab_filepath = os.path.join(args.text_dir, 'preprocess/vocab_def.txt')
with open(vocab_filepath, 'w') as ofp:
    for w in w_def_count:
        ofp.write('{}\t{}\n'.format(w[0], w[1]))

w_low_count = sorted(w_low_count.items(),
                     key=operator.itemgetter(1), reverse=True)
bow_vocab_size = 0
stopwords = set()
with open(args.stopword_file) as ifp:
    for line in ifp:
        stopwords.add(line.strip())

bow_vocab_filepath = os.path.join(args.text_dir, 'preprocess/bow_vocab.txt')
with open(bow_vocab_filepath, 'w') as ofp:
    for w in w_low_count:
        if bow_vocab_size >= args.bow_vocab_size:
            break
        if w[0] in stopwords:
            continue
        ofp.write('{}\t{}\n'.format(w[0], w[1]))
        bow_vocab_size += 1
