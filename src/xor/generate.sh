#!/usr/bin/env sh

rm -r xor_lmdb_test
rm -r xor_lmdb_train
../../build/src/xor/generate-random-xor-training-data --backend=lmdb --split=1 --shuffle=true 1000 xor_lmdb
