#!/usr/bin/env sh

../../build/src/shape/generate-random-shape-training-data --backend=lmdb --split=1 --shuffle=true --balance=true shape_lmdb
