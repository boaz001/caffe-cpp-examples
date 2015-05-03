# shape
Train a network with random generated shapes (squares and circles) data and classify with the trained model

## Generate training data
    ./generate.sh

This default generate script has generated 1000+ random samples and put those in two LMDB databases ```shape_lmdb_train``` and ```shape_lmdb_test``` equally divided. The samples are a square kernel of all shapes and around the shapes.
Each shape is labeled, 1 for squares, 2 for circles and 0 for background.

## Train the network
    ./train.sh

This default train script uses the settings from the solver.prototxt to train the network defined by train-test.prototxt
In the end it should give a accuracy of 1 and the loss should minimize to a really low number 0.0...

## Classify some data with the trained model
    ./classify.sh

This default classification script classifies a random generated image with squares and circles, using the trained model and the network from deploy.prototxt and shows the classification result image and the classification error percentage.
