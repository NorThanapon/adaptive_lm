#!/bin/bash
echo "[1/2] Downloading data..."
DATA_NAMES="single all"
for N in $DATA_NAMES; do
  wget http://websail-fe.cs.northwestern.edu/downloads/dictdef/wordnet_$N.tar.gz
  tar -xf wordnet_$N.tar.gz
  rm wordnet_$N.tar.gz
done
mv first_sense_splits wordnet_single
mv all_splits wordnet_all
echo "[2/2] Preprocessing..."
for N in $DATA_NAMES; do
  cd wordnet_$N
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
  python ../adaptive_lm/preprocess/preprocess_defs.py wordnet_$N --source_index 3
done
