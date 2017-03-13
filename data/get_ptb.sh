#!/bin/bash

echo "[1/3] Downloading data..."
wget http://websail-fe.cs.northwestern.edu/downloads/cached/simple-examples.tgz
echo "[2/3] Extracting files..."
tar -xf simple-examples.tgz
rm simple-examples.tgz
mkdir ptb
mv simple-examples/data/ptb.test.txt ptb/test.txt
mv simple-examples/data/ptb.train.txt ptb/train.txt
mv simple-examples/data/ptb.valid.txt ptb/valid.txt
mkdir ptb-char
mv simple-examples/data/ptb.char.test.txt ptb-char/test.txt
mv simple-examples/data/ptb.char.train.txt ptb-char/train.txt
mv simple-examples/data/ptb.char.valid.txt ptb-char/valid.txt
rm -r simple-examples
echo "[3/3] Preprocessing text files..."
mkdir ptb/preprocess
mkdir ptb-char/preprocess
python ../adaptive_lm/preprocess/preprocess_text.py ptb/
python ../adaptive_lm/preprocess/preprocess_text.py ptb-char/ --no_unk
