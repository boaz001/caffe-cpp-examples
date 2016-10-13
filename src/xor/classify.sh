#!/usr/bin/env sh

../../build/src/xor/classify-xor deploy.prototxt snapshot_iter_500.caffemodel 0 0
../../build/src/xor/classify-xor deploy.prototxt snapshot_iter_500.caffemodel 0 1
../../build/src/xor/classify-xor deploy.prototxt snapshot_iter_500.caffemodel 1 0
../../build/src/xor/classify-xor deploy.prototxt snapshot_iter_500.caffemodel 1 1
