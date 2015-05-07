# caffe-cpp-examples [![Build Status](https://travis-ci.org/boaz001/caffe-cpp-examples.svg?branch=master)](https://travis-ci.org/boaz001/caffe-cpp-examples)
Examples of how to use the Caffe framework with C++

## Requires
    cmake >= 2.8.8
    Caffe
    OpenCV >= 2.4
Note: Since there is no pre-build Caffe package (yet) Caffe must be built with cmake and ```make install```'ed. If in doubt; [This](https://github.com/BVLC/caffe/pull/1667) is what I used.

## Install
    git clone git@github.com:boaz001/caffe-cpp-examples.git
    cd caffe-cpp-examples
    mkdir build
    cd build
    cmake ..
    make

Now you can navigate to a example in the ```src``` directory and follow the instructions in each README.md

## Notes
The source of some examples are derived from the caffe tools.
It was never my intent to write super efficient code but rather some small simple examples of how to use Caffe in C++.

## License
MIT (see LICENSE for details)
