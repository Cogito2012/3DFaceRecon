import tensorflow as tf
from nets.network import FaceRecNet
import os
import argparse
import utils.parser_3dmm as parser_3dmm
from utils.data_process import trainval_generator, test_generator
from utils.timer import Timer
import cv2

ROOT_PATH = os.path.dirname(__file__)



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
        labels_placeholder = tf.placeholder(dtype=tf.float32, shape=[p.batch_size, 1, 1, ndim_params], name='params_label')
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
        face_recnet.build()

        # construct loss
        losses = face_recnet.get_loss()

        # add summaries
        summary_op, summary_op_val = face_recnet.add_summaries()

        # construct optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=p.base_lr, name='Adam')
        train_op = optimizer.minimize(losses['total_loss'])

        # prepare saver and writer
        tf_saver = tf.train.Saver(max_to_keep=100000)
        # Write the train and validation information to tensorboard
        tf_train_writer = tf.summary.FileWriter(tb_train_dir, sess.graph)
        tf_val_writer = tf.summary.FileWriter(tb_val_dir)

        # construct a data generator
        img_size = [p.image_size, p.image_size]
        traindata_generator = trainval_generator(p.batch_size, img_size, ndim_params,
                                                 dataset=p.dataset, img_mean=face_recnet.gray_mean, phase='train')
        valdata_generator = trainval_generator(p.batch_size, img_size, ndim_params,
                                                 dataset=p.dataset, img_mean=face_recnet.gray_mean, phase='val')
        # initialize all models
        last_snapshot_iter, ss_paths = face_recnet.initialize_models(sess)

        timer = Timer()
        iter = 0 # We always start with the initial time point
        while iter < p.max_iters + 1:
            timer.tic()
            # Get training data, one batch at a time
            train_images, train_labels = next(traindata_generator)
            # train step
            feed_dict = {grayimg_placeholder: train_images,
                         labels_placeholder: train_labels}
            pose_loss, geometry_loss, sh_loss, fidelity_loss, smoothness_loss, total_loss, summary, _ = sess.run(
                [losses['pose_loss'], losses['geometry_loss'],
                 losses['spherical_harmonics_loss'], losses['fidelity_loss'], losses['smoothness_loss'],
                 losses['total_loss'],
                 summary_op, train_op],
                feed_dict=feed_dict)
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
                      ' --- loss_spherical_harmonics: %.6f, --- loss_fidelity: %.6f, --- loss_smoothness: %.6f\n'
                      ' --- loss_total (train/val): %.6f / (%.6f)'
                      % (pose_loss, geometry_loss,
                         sh_loss, fidelity_loss, smoothness_loss,
                         total_loss, total_loss_val))
                print(' --- speed: {:.3f}s / iter'.format(timer.average_time))

            # Snapshotting
            if iter % p.snapshot_iters == 0 and iter > 0:
                last_snapshot_iter = iter
                ss_path = face_recnet.snapshot(sess, tf_saver, checkpoints_dir, iter, prefix=ckpt_prefix)
                ss_paths.append(ss_path)

                # Remove the old snapshots if there are too many
                if len(ss_paths) > 5:
                    face_recnet.remove_snapshot(ss_paths ,keep=5)
            iter += 1

        if last_snapshot_iter != iter - 1:
            face_recnet.snapshot(sess, tf_saver, checkpoints_dir, iter - 1, prefix=ckpt_prefix)
        tf_train_writer.close()
        tf_val_writer.close()


def get_weight_file(model_wieghts):
    ''' Get the weight file from specific filename or from the checkpoint dirs if not specified

    :param model_wieghts: filename str or None
    :return:
    '''
    if p.model_wieghts:
        checkpoints_file = os.path.join(ROOT_PATH, model_wieghts)
        if not os.path.exists(checkpoints_file):
            raise FileNotFoundError
    else:
        checkpoints_dir = os.path.join(ROOT_PATH, p.output_path, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            raise FileNotFoundError
        else:
            ckpt_files = os.listdir(checkpoints_dir)
            ckpt_files.sort()
            checkpoints_file = os.path.join(checkpoints_dir, ckpt_files[-1])
    return checkpoints_file


def eval_model(sess):
    # checkpoint files dir
    checkpoints_file = get_weight_file(p.model_wieghts)

    # prepare evaluation result directory
    eval_dir = os.path.join(ROOT_PATH, p.output_path, 'evaluation')
    eval_depth_dir = os.path.join(eval_dir, 'depth')
    if not os.path.exists(eval_depth_dir):
        os.makedirs(eval_depth_dir)

    # read basic params from 3dmm facial model
    mesh_data_path = os.path.join(ROOT_PATH, '3dmm')
    mesh_data_3dmm = parser_3dmm.read_3dmm_model(mesh_data_path)

    with sess.graph.as_default():
        # the place holder for single image input
        grayimg_placeholder = tf.placeholder(dtype=tf.float32, shape=[p.batch_size, p.image_size, p.image_size, 1], name='im_gray')

        # initialize Face Reconstruction Model
        face_recnet = FaceRecNet(
            im_gray=grayimg_placeholder,
            mesh_data=mesh_data_3dmm,
            nIter=p.nIter,
            batch_size=p.batch_size,
            im_size=p.image_size
        )
        # build up computational graph
        face_recnet.build(is_training=False)

        print(('Loading model check point from {:s}').format(checkpoints_file))
        saver = tf.train.Saver()
        saver.restore(sess, checkpoints_file)
        print('Loaded.')

        img_size = [p.image_size, p.image_size]
        testdata_generator = test_generator(p.batch_size, img_size, dataset=p.dataset, img_mean=face_recnet.gray_mean)
        batch_id = 0
        timer = Timer()
        while True:
            try:
                timer.tic()
                test_images, image_files = next(testdata_generator)
                pred_depth = sess.run(face_recnet.pred_depth_map, feed_dict={grayimg_placeholder: test_images})
                depth_images = pred_depth * 255.0
                # prepare output results
                for filename, depth_im in zip(image_files, depth_images):
                    obj_id = os.path.splitext(filename)[0].split('/')[-2]
                    file_id = os.path.splitext(filename)[0].split('/')[-1]
                    result_depth_file = os.path.join(eval_depth_dir, obj_id + '_' + file_id + '.jpg')
                    cv2.imwrite(result_depth_file, depth_im)
                    # TODO: Load ground truth for quantitivity evaluation of 3D facial depth
                    #
                batch_id += 1
                timer.toc()
                print('process test data batch: %d, speed: %.3f s per batch.'% (batch_id, timer.average_time))
            except StopIteration:
                break

        print('Done!')




def trainval(phase):
    '''Entry function for training or evaluation

    :param phase:
    :return:
    '''
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        if phase == 'train':
            train_model(sess)
        if phase == 'test':
            eval_model(sess)
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='vggface', help='The dataset name or folder name of training data.')
    parser.add_argument('--phase', default='train', choices=['train', 'test'], help=['The phase of running.'])
    parser.add_argument('--nIter', type=int, default=4, help='The number of iteration for CoarseNet.')
    parser.add_argument('--max_iters', type=int, default=70000, help='The number of iterations for training process.')
    parser.add_argument('--image_size', type=int, default=200, help='The input image size.')
    parser.add_argument('--batch_size', type=int, default=4, help='The batchsize in the training and evaluation.')
    parser.add_argument('--base_lr', type=float, default=1e-3, help='The base learning rate.')
    parser.add_argument('--model_wieghts', default=None, help='The weights file in the snapshot directory for evaluation.')
    parser.add_argument('--display', type=int, default=10, help='The display intervals for training.')
    parser.add_argument('--snapshot_iters', type=int, default=1000, help='The number of iterations to snapshot trained model.')
    parser.add_argument('--output_path', default='./output', help='The path to save output results.')
    p = parser.parse_args()

    trainval(p.phase)
