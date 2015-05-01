#!/usr/bin/env sh

../../build/src/xor/generate-random-xor-training-data --backend=lmdb --split=1 --shuffle=true 1000 xor_lmdb
