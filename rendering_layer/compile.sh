#!/bin/bash

# Note: To successfully compile this lib, the following dependencies must be satisfied:
# 1. python==3.5 (We recommand using the Anaconda3.4.11 for installing Python3)
# 2. TensorFlow==1.2.0 (with cuDNN5.1 and CUDA8.0)
# 3. gcc/g++ == 4.8
# Make sure the cuda path is corectly set in the ops.py

rm -rf ./ops_src/*.so
rm -rf ./ops_src/*.o

python ops.py

# echo "The rendering layer has been compiled successfully....."
