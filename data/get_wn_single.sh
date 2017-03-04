#!/bin/bash
echo "[1/2] Downloading data..."
wget http://websail-fe.cs.northwestern.edu/downloads/dictdef/wordnet_single.tar.gz
tar -xf wordnet_single.tar.gz
rm wordnet_single.tar.gz
mv first_sense_splits wordnet_single
cd wordnet_single
mkdir -p preprocess
mkdir -p shortlist
SPLITS="train valid test"
for SPLIT in $SPLITS; do
  mv $SPLIT".txt" $SPLIT".tsv"
  awk -F '\t' '{print $1}' $SPLIT".tsv" | sort | uniq > "shortlist/shortlist_"$SPLIT".txt"
  cat "shortlist/shortlist_"$SPLIT".txt" >> "tmp.txt"
done
mv "tmp.txt" "shortlist/shortlist_all.txt"
cd ../
echo "[2/2] Preprocessing..."
python ../adaptive_lm/preprocess/preprocess_defs.py wordnet_single
