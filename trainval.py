import tensorflow as tf
from nets.network import FaceRecNet
import time
import os
import logging
import argparse
import utils.parser_3dmm as parser_3dmm
import utils.listfile_reader as file_reader
import utils.data_process as data_process
import scipy.io as sio
import numpy as np

ROOT_PATH = os.path.dirname(__file__)

def generator(batch_size, img_size, phase='train'):
    dataset_path = os.path.join(ROOT_PATH, 'data', 'vggface')
    if phase == 'train' or phase == 'val':
        image_files, label_files = file_reader.read_listfile_trainval(dataset_path, 'train_list.txt')
    else:
        image_files, label_files = file_reader.read_listfile_test(dataset_path, 'val_list')

    counter = 0
    while True:
        yield data_process.prepare_input_image(image_files[counter:counter + batch_size], batch_size, img_size), \
              data_process.prepare_input_label(label_files[counter:counter + batch_size], batch_size)
        counter = (counter + batch_size) % len(image_files)


def train():
    if p.ckpt_file is not None:
        checkpoints_dir = os.path.join(ROOT_PATH, p.output_path, "checkpoints", p.ckpt_file)
    else:
        checkpoints_dir = os.path.join(ROOT_PATH, p.output_path, "checkpoints")
        if os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

    # read basic params from 3dmm facial model
    mesh_data_path = os.path.join(ROOT_PATH, '3dmm')
    mesh_data_3dmm = parser_3dmm.read_3dmm_model(mesh_data_path)
    ndim_params = mesh_data_3dmm['ndim_pose'] + mesh_data_3dmm['ndim_shape'] + mesh_data_3dmm['ndim_exp']

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
            mesh_data=mesh_data_3dmm,
            nIter=p.nIter,
            batch_size=p.batch_size,
            im_size=p.image_size,
            weight_decay=1e-4
        )
        pred_depth_map = face_recnet.build()
        loss = face_recnet.get_loss()

    print('===============================')

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
