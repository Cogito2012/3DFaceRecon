#!/bin/bash

sudo wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
sudo tar -xzvf vgg_16_2016_08_28.tar.gz
sudo mv vgg_16.ckpt vgg16.ckpt

sudo wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
sudo tar -xzvf resnet_v1_101_2016_08_28.tar.gz
sudo mv resnet_v1_101.ckpt res101.ckpt

sudo rm -rf ./*.tar.gz