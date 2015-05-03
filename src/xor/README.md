# xor
Train a network with random generated XOR data and classify with the trained model

## Generate training data
    ./generate.sh

This default generate script has generated 1000 random generate XOR samples and put those in two LMDB databases ```xor_lmdb_train``` and ```xor_lmdb_test``` equally divided so 500 samples in each.

Possible values for the samples are the XOR function, which looks like:

| In1 | In2 | Out |
|-----|-----|-----|
|  0  |  0  |  0  |
|  0  |  1  |  1  |
|  1  |  0  |  1  |
|  1  |  1  |  0  |

Where Out is the same as the label

## Train the network
    ./train.sh

This default train script uses the settings from the solver.prototxt to train the network defined by train-test.prototxt
In the end it should give a accuracy of 1 and the loss should minimize to a really low number 0.0...

## Classify some data with the trained model
    ./classify.sh

This default classification script classifies all four possible Input combinations (see table above) using the trained model and the network from deploy.prototxt and shows the values and if it was classified correctly (all should be GOOD of course).
