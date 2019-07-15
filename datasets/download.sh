#!/bin/bash

mkdir ./mnist
wget -P ./mnist/ http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -P ./mnist/ http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget -P ./mnist/ http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -P ./mnist/ http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gzip -d -f ./mnist/*.gz
