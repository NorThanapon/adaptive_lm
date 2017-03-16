#!/bin/bash
echo "[1/2] Downloading data..."
wget http://websail-fe.cs.northwestern.edu/downloads/dictdef/common_wordnet_defs.tar.gz
tar -xf common_wordnet_defs.tar.gz
rm common_wordnet_defs.tar.gz
cd common_wordnet_defs
mkdir -p first_senses
mkdir -p all_senses
SPLITS="train valid test"
for SPLIT in $SPLITS; do
  mv $SPLIT".txt" all_senses/$SPLIT".tsv"
  grep -P '\t1\t' all_senses/$SPLIT".tsv" > first_senses/$SPLIT".tsv"
done
DATA_NAMES="first_senses all_senses"
echo "[2/2] Preprocessing..."
for N in $DATA_NAMES; do
  cd $N
  mkdir -p preprocess
  mkdir -p shortlist
  SPLITS="train valid test"
  for SPLIT in $SPLITS; do
    awk -F '\t' '{print $1}' $SPLIT".tsv" | sort | uniq > "shortlist/shortlist_"$SPLIT".txt"
    cat "shortlist/shortlist_"$SPLIT".txt" >> "tmp.txt"
  done
  mv "tmp.txt" "shortlist/shortlist_all.txt"
  cd ../../
  python ../adaptive_lm/preprocess/preprocess_defs.py common_wordnet_defs/$N --source_indices 1,2,3
  cd common_wordnet_defs
done
