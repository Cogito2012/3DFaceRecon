import tensorflow as tf
from nets.network import FaceRecNet
import time
import os
import logging
import argparse
import utils.parser_3dmm as parser_3dmm
ROOT_PATH = os.path.dirname(__file__)




def train():
    if p.ckpt_file is not None:
        checkpoints_dir = os.path.join(ROOT_PATH, p.output_path, "checkpoints", p.ckpt_file)
    else:
        checkpoints_dir = os.path.join(ROOT_PATH, p.output_path, "checkpoints")
        if os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

    # read basic params from 3dmm facial model
    modeldata_3dmm = parser_3dmm.read_3dmm_model()
    ndim_params = modeldata_3dmm['ndim_pose'] + modeldata_3dmm['ndim_shape'] + modeldata_3dmm['ndim_exp']

    graph = tf.Graph()
    with graph.as_default():
        # Set the random seed for tensorflow
        tf.set_random_seed(12345)

        grayimg_placeholder = tf.placeholder(dtype=tf.float32, shape=[p.batch_size, p.image_size, p.image_size, 1], name='im_gray')
        labels_placeholder = tf.placeholder(dtype=tf.float32, shape=[p.batch_size, ndim_params], name='im_gray')
        # initialize SalCNN Model
        face_recnet = FaceRecNet(
            im_gray=grayimg_placeholder,
            params_label=labels_placeholder,
            modeldata_3dmm=modeldata_3dmm,
            nIter=p.nIter,
            batch_size=p.batch_size,
            im_size=p.image_size,
            weight_decay=1e-4
        )
        pred_depth_map = face_recnet.build()
        loss = face_recnet.get_loss()



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