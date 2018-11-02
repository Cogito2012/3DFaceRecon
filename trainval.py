import tensorflow as tf
from nets.network import FaceRecNet
import time
import os, sys
import logging
import argparse
import utils.parser_3dmm as parser_3dmm
import utils.listfile_reader as file_reader
import utils.data_process as data_process
import scipy.io as sio
import numpy as np
from utils.timer import Timer
from tensorflow.python import debug as tf_debug

ROOT_PATH = os.path.dirname(__file__)

def trainval_generator(batch_size, img_size, label_dim, phase='train'):
    '''The data generator to fetch a batch of images and labels from disk

    :param batch_size:
    :param img_size:
    :param label_dim:
    :param phase:
    :return:
    '''
    dataset_path = os.path.join(ROOT_PATH, 'data', 'vggface')
    if phase == 'train':
        image_files, label_files = file_reader.read_listfile_trainval(dataset_path, 'train_list.txt')
    elif phase == 'val':
        image_files, label_files = file_reader.read_listfile_trainval(dataset_path, 'val_list.txt')
    else:
        raise NotImplementedError

    counter = 0
    while True:
        yield data_process.prepare_input_image(image_files[counter:counter + batch_size], batch_size, img_size), \
              data_process.prepare_input_label(label_files[counter:counter + batch_size], batch_size, label_dim)
        counter = (counter + batch_size) % len(image_files)


def test_generator(batch_size, img_size):
    '''The data generator to fetch a batch of test images from disk

    :param batch_size:
    :param img_size:
    :return:
    '''
    dataset_path = os.path.join(ROOT_PATH, 'data', 'vggface')
    image_files = file_reader.read_listfile_test(dataset_path, 'test_list.txt')

    counter = 0
    while True:
        yield data_process.prepare_input_image(image_files[counter:counter + batch_size], batch_size, img_size)
        counter = (counter + batch_size) % len(image_files)


def train_model(sess):
    # checkpoint files dir
    checkpoints_dir = os.path.join(ROOT_PATH, p.output_path, "checkpoints")
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    ckpt_prefix = 'FaceReconNet'
    # tensorboard dirs
    tb_train_dir = os.path.join(ROOT_PATH, p.output_path, 'tensorboard', 'train')
    tb_val_dir = os.path.join(ROOT_PATH, p.output_path, 'tensorboard', 'val')
    if not os.path.exists(tb_train_dir):
        os.makedirs(tb_train_dir)
    if not os.path.exists(tb_val_dir):
        os.makedirs(tb_val_dir)

    # read basic params from 3dmm facial model
    mesh_data_path = os.path.join(ROOT_PATH, '3dmm')
    mesh_data_3dmm = parser_3dmm.read_3dmm_model(mesh_data_path)
    ndim_params = mesh_data_3dmm['ndim_pose'] + mesh_data_3dmm['ndim_shape'] + mesh_data_3dmm['ndim_exp']

    with sess.graph.as_default():
        # Set the random seed for tensorflow
        tf.set_random_seed(12345)

        grayimg_placeholder = tf.placeholder(dtype=tf.float32, shape=[p.batch_size, p.image_size, p.image_size, 1], name='im_gray')
        labels_placeholder = tf.placeholder(dtype=tf.float32, shape=[p.batch_size, 1, 1, ndim_params], name='im_gray')
        # initialize Face Reconstruction Model
        face_recnet = FaceRecNet(
            im_gray=grayimg_placeholder,
            params_label=labels_placeholder,
            mesh_data=mesh_data_3dmm,
            nIter=p.nIter,
            batch_size=p.batch_size,
            im_size=p.image_size,
            weight_decay=1e-4
        )
        # build up computational graph
        pred_depth_map = face_recnet.build()

        # construct loss
        losses = face_recnet.get_loss()

        # add summaries
        summary_op, summary_op_val = face_recnet.add_summaries()
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=p.base_lr)
        # construct optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=p.base_lr, name='Adam')
        train_op = optimizer.minimize(losses['total_loss'])
        # gvs = optimizer.compute_gradients(loss)
        # train_op = optimizer.apply_gradients(gvs)

        # prepare saver and writer
        tf_saver = tf.train.Saver(max_to_keep=100000)
        # Write the train and validation information to tensorboard
        tf_train_writer = tf.summary.FileWriter(tb_train_dir, sess.graph)
        tf_val_writer = tf.summary.FileWriter(tb_val_dir)

        # construct a data generator
        img_size = [p.image_size, p.image_size]
        traindata_generator = trainval_generator(p.batch_size, img_size, ndim_params, phase='train')
        valdata_generator = trainval_generator(p.batch_size, img_size, ndim_params, phase='val')

        # initialize all models
        last_snapshot_iter, np_paths, ss_paths = face_recnet.initialize_models(sess)

        timer = Timer()
        iter = 0 # We always start with the initial time point
        while iter < p.max_iters + 1:
            timer.tic()
            # Get training data, one batch at a time
            train_images, train_labels = next(traindata_generator)
            # train step
            feed_dict = {grayimg_placeholder: train_images,
                         labels_placeholder: train_labels}
            pose_loss, geometry_loss, total_loss, summary, _ = sess.run(
                [losses['pose_loss'], losses['geometry_loss'],
                 # losses['fidelity_loss'], losses['smoothness_loss'],
                 losses['total_loss'],
                 summary_op, train_op],
                feed_dict=feed_dict)
            # pose_loss, geometry_loss, fidelity_loss, smoothness_loss, total_loss, summary, _ = sess.run(
            #     [losses['pose_loss'], losses['geometry_loss'],
            #      losses['fidelity_loss'], losses['smoothness_loss'],
            #      losses['total_loss'],
            #      summary_op, train_op],
            #     feed_dict=feed_dict)
            # add train summaries
            tf_train_writer.add_summary(summary, float(iter))

            # validatation
            val_images, val_labels = next(valdata_generator)
            feed_dict = {grayimg_placeholder: val_images,
                         labels_placeholder: val_labels}
            total_loss_val, summary_val = sess.run([losses['total_loss'], summary_op_val], feed_dict=feed_dict)
            # add val summaries
            tf_val_writer.add_summary(summary_val, float(iter))
            timer.toc()

            # Display training information
            if iter % (p.display) == 0:
                print('--------------------------- iter: %d / %d, total loss: %.6f ---------------------------' % (
                iter, p.max_iters, total_loss))
                print(' --- loss_pose: %.6f,                --- loss_geometry: %.6f\n'
                      # ' --- loss_fidelity: %.6f             --- loss_smoothness: %.6f\n'
                      ' --- loss_total (train/val): %.6f / (%.6f)'
                      % (pose_loss, geometry_loss,
                         # fidelity_loss, smoothness_loss,
                         total_loss, total_loss_val))
                print(' --- speed: {:.3f}s / iter'.format(timer.average_time))

            # Snapshotting
            if iter % p.snapshot_iters == 0 and iter > 0:
                last_snapshot_iter = iter
                ss_path, np_path = face_recnet.snapshot(sess, tf_saver, checkpoints_dir, iter, prefix=ckpt_prefix)
                np_paths.append(np_path)
                ss_paths.append(ss_path)

                # Remove the old snapshots if there are too many
                if len(np_paths) > 5:
                    face_recnet.remove_snapshot(np_paths, ss_paths ,keep=5)
            iter += 1

        if last_snapshot_iter != iter - 1:
            face_recnet.snapshot(sess, tf_saver, checkpoints_dir, iter - 1, prefix=ckpt_prefix)
        tf_train_writer.close()
        tf_val_writer.close()

def train():

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        train_model(sess)

def test():
    '''TODO'''
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/VGGFace/vgg_face_dataset', help='The directory of training data.')
    parser.add_argument('--phase', default='train', choices=['train', 'test'], help=['The phase of running.'])
    parser.add_argument('--nIter', type=int, default=4, help='The number of iteration for CoarseNet.')
    parser.add_argument('--max_iters', type=int, default=70000, help='The number of iterations for training process.')
    parser.add_argument('--image_size', type=int, default=200, help='The input image size.')
    parser.add_argument('--batch_size', type=int, default=4, help='The batchsize in the training and evaluation.')
    parser.add_argument('--base_lr', type=float, default=1e-3, help='The base learning rate.')
    parser.add_argument('--ckpt_file', default=None, help='The weights file in the snapshot directory for training recovery.')
    parser.add_argument('--display', type=int, default=10, help='The display intervals for training.')
    parser.add_argument('--snapshot_iters', type=int, default=1000, help='The number of iterations to snapshot trained model.')
    parser.add_argument('--output_path', default='./output', help='The path to save output results.')
    p = parser.parse_args()

    if p.phase == 'train':
        train()
    if p.phase == 'test':
        test()
