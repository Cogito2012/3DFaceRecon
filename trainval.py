import tensorflow as tf
from nets.network import FaceRecNet
import time
import os
import logging
import argparse
ROOT_PATH = os.path.dirname(__file__)


def train():
    if p.ckpt_file is not None:
        checkpoints_dir = os.path.join(ROOT_PATH, p.output_path, "checkpoints", p.ckpt_file)
    else:
        checkpoints_dir = os.path.join(ROOT_PATH, p.output_path, "checkpoints")
        if os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

    graph = tf.Graph()
    with graph.as_default():
        # Set the random seed for tensorflow
        tf.set_random_seed(12345)

        grayimg_placeholder = tf.placeholder(dtype=tf.float32, shape=[p.image_size, p.image_size], name='im_gray')
        pncc_placeholder = tf.placeholder(dtype=tf.float32, shape=[p.image_size, p.image_size, 3], name='im_pncc')
        normal_placeholder = tf.placeholder(dtype=tf.float32, shape=[p.image_size, p.image_size, 3], name='im_normal')
        # initialize SalCNN Model
        face_recnet = FaceRecNet(
            im_gray=grayimg_placeholder,
            im_pncc=pncc_placeholder,
            im_normal=None,
            nIter=p.nIter,
            batch_size=p.batch_size,
            im_size=p.image_size
        )
        face_recnet.build()

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig, graph=graph) as sess:
        # TODO
        pass

def test():
    '''TODO'''
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/VGGFace/vgg_face_dataset', help='The directory of training data.')
    parser.add_argument('--phase', default='train', choices=['train', 'test'], help=['The phase of running.'])
    parser.add_argument('--nIter', type=int, default=4, help='The number of iteration for CoarseNet.')
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs for training process.')
    parser.add_argument('--image_size', type=int, default=200, help='The input image size.')
    parser.add_argument('--batch_size', type=int, default=64, help='The batchsize in the training and evaluation.')
    parser.add_argument('--base_lr', type=float, default=1e-3, help='The base learning rate.')
    parser.add_argument('--ckpt_file', default=None, help='The weights file in the snapshot directory for training recovery.')
    parser.add_argument('--output_path', default='./output', help='The path to save output results.')
    p = parser.parse_args()

    if p.phase == 'train':
        train()
    if p.phase == 'test':
        test()