import scipy.io as sio
import os
import numpy as np
import tensorflow as tf
from math import cos, sin
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block


def parse_3dmm_files(morphable_model, model_infofile, model_expfile, vertex_codefile):
    ''' Parse params and data from 3DMM model files

    :param morphable_model:
    :param model_infofile:
    :param model_expfile:
    :param vertex_codefile:
    :return:
    '''
    assert os.path.exists(morphable_model), 'File %s does not exist!' % (morphable_model)
    data = sio.loadmat(morphable_model)
    shapePC = data['shapePC']
    sigma_shape = data['shapeEV']
    shapeMU = data['shapeMU']

    assert os.path.exists(model_expfile), 'File %s does not exist!' % (model_expfile)
    data = sio.loadmat(model_expfile)
    mu_exp = data['mu_exp']
    pc_exp = data['w_exp']
    sigma_exp = data['sigma_exp']

    assert os.path.exists(model_infofile), 'File %s does not exist!' % (model_infofile)
    data = sio.loadmat(model_infofile)
    tri = data['tri']
    trimIndex = data['trimIndex']

    assert os.path.exists(vertex_codefile), 'File %s does not exist!' % (vertex_codefile)
    data = sio.loadmat(vertex_codefile)
    vertex_code = data['vertex_code']

    # prepare useful params
    trimIndex1 = np.hstack((np.hstack((3*trimIndex-2, 3*trimIndex-1)), 3*trimIndex))  # N x 3
    trimIndex1 = np.reshape(np.transpose(trimIndex1), [np.shape(trimIndex)[0]*3, 1])  # 3N x 1

    mu_shape = np.expand_dims(np.squeeze(shapeMU[trimIndex1]), axis=1)
    pc_shape = np.squeeze(shapePC[trimIndex1,:])

    mu = mu_shape + mu_exp
    return vertex_code, tri, mu, pc_shape, pc_exp


class FaceRecNet:
    def __init__(self, im_gray=None, im_pncc=None, im_normal=None, nIter=4, batch_size=64, im_size=200, weight_decay=1e-4):
        self.im_gray = im_gray
        self.im_pncc = im_pncc
        self.im_normal = im_normal
        self.nIter = nIter
        self.batch_size = batch_size
        self.im_size = im_size
        self.weight_decay = weight_decay

        # exterior data and model params
        morphable_model = '../3dmm/01_MorphableModel.mat'
        model_infofile = '../3dmm/model_info.mat'
        model_expfile = '../3dmm/Model_Expression.mat'
        vertex_codefile = '../3dmm/vertex_code.mat'
        self.vertex_code, self.tri, self.mu, self.pc_shape, self.pc_exp = \
            parse_3dmm_files(morphable_model, model_infofile, model_expfile, vertex_codefile)
        self.ndim_shape = np.shape(self.pc_shape)[1]
        self.ndim_exp = np.shape(self.pc_exp)[1]
        self.ndim_pose = 7
        self.ndim = self.ndim_pose + self.ndim_shape + self.ndim_exp

        # self.vertex_code = tf.constant(vertex_code, dtype=tf.float32)
        # self.tri = tf.constant(tri, dtype=tf.int32)
        # self.mu = tf.constant(mu, dtype=tf.float32)
        # self.pc_shape = tf.constant(pc_shape, dtype=tf.float32)
        # self.pc_exp = tf.constant(pc_exp, dtype=tf.float32)

        # construct geometry and pose parameters, they will be updated in the training
        geo_params = tf.zeros((self.batch_size, self.ndim_shape + self.ndim_exp), dtype=tf.float32)
        init_pose = tf.constant([0, 0, 0, self.im_size / 2.0, self.im_size / 2.0, 0, 0.001], dtype=tf.float32)
        pose_params = tf.tile(tf.expand_dims(init_pose, axis=0), [batch_size, 1])
        self.pred_param = tf.stack([pose_params, geo_params], axis=0, name='pred_params')  # pose + shape + expression


    def build(self, is_training=True):
        '''To create the architechture of the proposed method

        :param is_training:
        :return:
        '''
        # handle most of the regularizers here
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],  # conv, deconv, fc
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                            weights_initializer=slim.variance_scaling_initializer(),
                            biases_regularizer=tf.no_regularizer,
                            biases_initializer=tf.constant_initializer(0.0),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            trainable=is_training):
            self.build_coarse_net(is_training)

            self.rendering_layer(is_training)

            self.build_fine_net(is_training)


    def build_coarse_net(self, is_training=True):
        '''

        :param is_training:
        :return:
        '''
        for n in range(self.nIter):
            scope_name = 'Input_Rendering_iter%d'%(n)
            with tf.variable_scope(scope_name, scope_name):
                # transforming the predicted params into rendered PNCC
                # not end-to-end way by py_func
                # pncc_batch, maskimg_batch = tf.py_func(self.zbuffer_rendering, [self.pred_param], [tf.float32, tf.float32])
                pncc_batch, maskimg_batch = self.input_rendering_layer(self.pred_params)

            scope_name = 'CoarseNet_iter_iter%d'%(n)
            with tf.variable_scope(scope_name, scope_name):
                # shape: [batch_size, height, width, channels], channels = 4
                net_input = tf.concat([maskimg_batch, pncc_batch], axis=3, name='input_data')
                # CoarseNet
                net_conv = slim.conv2d(net_input, 32, [7, 7], stride=2, scope='conv1')
                res_blocks = [resnet_v1_block('block1', base_depth=32, num_units=2, stride=1),
                              resnet_v1_block('block2', base_depth=64, num_units=2, stride=2),
                              resnet_v1_block('block3', base_depth=128, num_units=2, stride=2),
                              resnet_v1_block('block4', base_depth=256, num_units=2, stride=2)]
                net_conv, _ = resnet_v1.resnet_v1(net_conv, res_blocks,
                                                  global_pool=False, include_root_block=False, reuse=False, scope='conv_blocks')
                net_conv = slim.conv2d(net_conv, self.ndim, [3, 3], scope='conv2')
                net_pool = tf.reduce_mean(net_conv, [1, 2], keep_dims=False, name='pooling') # Global average pooling. 13x13
                self.pred_params = slim.fully_connected(net_pool, self.ndim,
                                     weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001),
                                     activation_fn=None, scope='pred_params')

    def input_rendering_layer(self, pred_params):
        # parse geometry and pose parameters from prediction
        pose_params = tf.slice(pred_params, [0, 0], [-1, self.ndim_pose], name='pred_pose_params')
        shape_params = tf.slice(pred_params, [0, self.ndim_pose], [-1, self.ndim_shape], name='pred_shape_params')
        exp_params = tf.slice(pred_params, [0, self.ndim_pose + self.ndim_shape], [-1, self.ndim_exp], name='pred_exp_params')
        # pose parameters
        phi, gamma, theta, t3d, f = self.parse_pose_params(pose_params)
        R = tf.map_fn(self.rotation_maxtrix, np.stack([phi, gamma, theta], axis=0))  # Nx3x3


    def parse_pose_params(self, pose_params):
        ''' Get the params with physical meanings
        :param pose_params: batchsize x 7
        :return: phi, gamma, theta, t3d(3x1), f
        '''
        phi = tf.slice(pose_params, [0, 0], [-1, 1], name='pred_phi')
        gamma = tf.slice(pose_params, [0, 1], [-1, 1], name='pred_gamma')
        theta = tf.slice(pose_params, [0, 2], [-1, 1], name='pred_theta')
        t3d = tf.slice(pose_params, [0, 3], [-1, 3], name='pred_t3d')
        f = tf.slice(pose_params, [0, 6], [-1, 1], name='pred_f')
        return phi, gamma, theta, t3d, f


    def rotation_matrix(self, angles):
        '''get rotation matrix from three rotation angles(degree). right-handed.

        :param angles: phi, gamma, theta angles (1, 3)
                phi: pitch. positive for looking up.
                gamma: yaw. positive for looking right.
                theta: roll. positive for tilting head left.
        :return: rotation matrix (3, 3).
        '''
        phi, gamma, theta = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
        # phi
        R_pitch = np.array([[1, 0, 0],
                       [0, cos(phi), sin(phi)],
                       [0, -sin(phi), cos(phi)]])
        # gamma
        R_yaw = np.array([[cos(gamma), 0, -sin(gamma)],
                       [0, 1, 0],
                       [sin(gamma), 0, cos(gamma)]])
        # theta
        R_roll = np.array([[cos(theta), sin(theta), 0],
                       [-sin(theta), cos(theta), 0],
                       [0, 0, 1]])

        R = R_roll.dot(R_yaw.dot(R_pitch))
        return R.astype(np.float32)

if __name__ == '__main__':
    # exterior data and model params
    morphable_model = '../3dmm/01_MorphableModel.mat'
    model_infofile = '../3dmm/model_info.mat'
    model_expfile = '../3dmm/Model_Expression.mat'
    vertex_codefile = '../3dmm/vertex_code.mat'
    vetex_code, tri, mu, pc_shape, pc_exp = \
        parse_3dmm_files(morphable_model, model_infofile, model_expfile, vertex_codefile)

    print 'ss'