import sys
import os

vocab_path = sys.argv[1]
stopword_path = sys.argv[2]
function_word_path = sys.argv[3]
num_senses = int(sys.argv[4])
output_path = sys.argv[5]

freq_limit = 10
char_limit = 2

no_good_words = set()
with open(stopword_path) as ifp:
    for line in ifp:
        no_good_words.add(line.strip())
with open(function_word_path) as ifp:
    for line in ifp:
        no_good_words.add(line.strip())

vocab = []
with open(vocab_path) as ifp:
    for line in ifp:
        p = line.strip().split()
        word = p[0]
        freq = int(p[1])
        vocab.append((word, freq))

next_index = 0
sense_map = []
for i, (word, freq) in enumerate(vocab):
    n = num_senses
    if word in no_good_words or len(word) <= char_limit or freq <= freq_limit:
        n = 1
    sense_map.append(range(next_index, next_index+n))
    next_index += n

with open(output_path, 'w') as ofp:
    for s in sense_map:
        ofp.write(' '.join([str(i) for i in s]))
        ofp.write('\n')
