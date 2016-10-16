#!/usr/bin/env sh

rm -r shape_lmdb_test
rm -r shape_lmdb_train
../../build/src/shape/generate-random-shape-training-data --backend=lmdb --split=1 --shuffle=true --balance=true shape_lmdb
