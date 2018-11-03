from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from math import cos, sin
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block


from rendering_layer.ops import render_depth
import os

class FaceRecNet:
    def __init__(self, im_gray=None, params_label=None, mesh_data=None, nIter=4, batch_size=64, im_size=200, weight_decay=1e-4):
        self.im_gray = im_gray
        self.params_label = params_label
        self.nIter = nIter
        self.batch_size = batch_size
        self.im_size = im_size
        self.weight_decay = weight_decay
        self.ndim_tex = 10
        self.lambda_pose = 1.0  # 1e-3
        self.lambda_geo = 1e-6   #
        self.lambda_sh = 1e-3  # 1.0 in paper
        self.lambda_f = 100  # 5e-2 in paper
        self.lambda_sm = 1e-5  # 1.0 in paper
        self.gray_mean = 126.064
        self.pretrained_resnet_v1 = './pretrained/res101.ckpt'
        self.pretrained_vgg16 = './pretrained/vgg16.ckpt'
        self.net_scopes = {'coarse_net': 'resnet_v1_101',
                           'fine_net': 'vgg_16'}

        # mesh data
        self.vertex_code = mesh_data['vertex']
        self.tri = mesh_data['tri']
        self.mu = tf.constant(mesh_data['mu'], dtype=tf.float32)  # (3*Nvert, 1)
        self.pc_shape = tf.constant(mesh_data['pc_shape'], dtype=tf.float32)  # (3*Nvert, ndim_shape)
        self.pc_exp = tf.constant(mesh_data['pc_exp'], dtype=tf.float32)      # (3*Nvert, ndim_exp)
        self.mu_tex = tf.constant(mesh_data['mu_tex'], dtype=tf.float32)  # (3, Nvert)
        self.pc_tex = tf.constant(mesh_data['pc_tex'][:, 0:self.ndim_tex], dtype=tf.float32)  # (3*Nvert, ndim_tex)
        self.param_tex = tf.constant(mesh_data['param_tex'][0:self.ndim_tex, :])
        # self.lighting = tf.constant([0.2, 0, 0.98], dtype=tf.float32) # the first-order spherical harmonics coefficients

        # mesh info
        self.ndim_shape = mesh_data['ndim_shape']
        self.ndim_exp = mesh_data['ndim_exp']
        self.ndim_pose = mesh_data['ndim_pose']
        self.ndim = self.ndim_pose + self.ndim_shape + self.ndim_exp
        self.nvert = int(np.floor(np.shape(self.vertex_code)[0] / 3.0))

        # construct geometry and pose parameters, they will be updated in the training
        geo_params = tf.zeros((self.batch_size, self.ndim_shape + self.ndim_exp), dtype=tf.float32)
        init_pose = tf.constant([0, 0, 0, self.im_size / 2.0, self.im_size / 2.0, 0, 0.001], dtype=tf.float32)
        pose_params = tf.tile(tf.expand_dims(init_pose, axis=0), [batch_size, 1])
        init_pred_params = tf.concat([pose_params, geo_params], axis=1, name='pred_params')  # pose + shape + expression (B, d)
        self.pred_params = tf.expand_dims(tf.expand_dims(init_pred_params, axis=1), axis=1)  # (B, 1, 1 ,d)

        self.losses = {}
        self.scalar_summaries = {}
        self.image_summaries = {}
        self.histogram_summaries = {}
        self.tf_print = {}

    def build(self, is_training=True):
        '''To create the architechture of the proposed method

        :param is_training:
        :return:
        '''
        # handle most of the regularizers here
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],  # conv, deconv, fc
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            biases_regularizer=tf.no_regularizer,
                            biases_initializer=tf.constant_initializer(0.0),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            trainable=is_training):
            # CoarseNet
            self.build_coarse_net()
            if is_training:
                self.histogram_summaries.update({'pred_params': self.pred_params})

            # Rendering Layer
            self.depth_rendering_layer()
            if is_training:
                self.image_summaries.update({'pncc_map': self.pncc_batch * 255.0,
                                            'mask_im': self.maskimg_batch,
                                            'normal_map': (self.normal_batch + 1.0) / 2.0 * 255.0,
                                            'coarse_depth': self.depth_to_im(self.coarse_depth_map)})
            # FineNet
            self.build_fine_net()
            if is_training:
                self.image_summaries.update({'fine_depth': self.pred_depth_map})



    def build_coarse_net(self):
        '''

        :param is_training:
        :return:
        '''
        for n in range(self.nIter):
            scope_name = 'Input_Rendering_iter%d'%(n)
            with tf.variable_scope(scope_name, scope_name):
                # transforming the predicted params into vertices
                vertices_proj = self.vertices_transform(self.pred_params)
                # rendered PNCC, normal map and Masked Image
                pncc_batch, normal_batch, maskimg_batch, _ = \
                    self.rendering_layer(vertices_proj, self.tri, self.vertex_code)

            scope_name = 'CoarseNet_iter_iter%d'%(n)
            # scope_name = self.net_scopes['coarse_net']
            with tf.variable_scope(scope_name, scope_name):
                # shape: [batch_size, height, width, channels], channels = 7
                net_input = tf.concat([maskimg_batch, pncc_batch, normal_batch], axis=3, name='input_data')
                # CoarseNet
                net_conv = slim.conv2d(net_input, 64, [7, 7], stride=2, scope='conv_in')
                res_blocks = [resnet_v1_block('block1', base_depth=64, num_units=2, stride=1),  # in paper: 32
                              resnet_v1_block('block2', base_depth=128, num_units=2, stride=2), # in paper: 64
                              resnet_v1_block('block3', base_depth=256, num_units=2, stride=2), # in paper: 128
                              resnet_v1_block('block4', base_depth=512, num_units=2, stride=2)] # in papar: 256
                net_conv, _ = resnet_v1.resnet_v1(net_conv, res_blocks,
                                                  global_pool=False, include_root_block=False, reuse=False)
                net_conv = slim.conv2d(net_conv, self.ndim, [3, 3], scope='conv_out')
                net_pool = tf.reduce_mean(net_conv, [1, 2], keep_dims=True, name='pooling') # Global average pooling. 13x13
                self.pred_params = slim.fully_connected(net_pool, self.ndim,
                                     weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001),
                                     normalizer_fn=None, activation_fn=None, scope='fc')


    def vertices_transform(self, pred_params):
        # parse geometry and pose parameters from prediction
        pred_params = tf.squeeze(pred_params, [1, 2])  # (B, d)
        pose_params = tf.slice(pred_params, [0, 0], [-1, self.ndim_pose], name='pred_pose_params')  # (B, ndim_pose)
        shape_params = tf.slice(pred_params, [0, self.ndim_pose], [-1, self.ndim_shape], name='pred_shape_params')  # (B, ndim_shape)
        exp_params = tf.slice(pred_params, [0, self.ndim_pose + self.ndim_shape], [-1, self.ndim_exp], name='pred_exp_params') # (B, ndim_exp)
        # pose parameters
        phi, gamma, theta, t3d, f = self.parse_pose_params(pose_params)
        # R here is not computed by TF API, but numpy.
        # TODO: pure TF implementation to calculate R.
        R = tf.py_func(self.rotation_matrix_batch, [tf.concat([phi, gamma, theta], axis=1)], tf.float32)
        R.set_shape([self.batch_size, 3, 3])

        shapes = tf.transpose(tf.matmul(self.pc_shape, shape_params, transpose_b=True), name='pred_shape') # (B, 3*N)
        shapes = tf.transpose(tf.reshape(shapes, [self.batch_size, 3, -1]), [0, 2, 1], name='pred_shape_reshape')  # (B, N, 3)
        expressions = tf.transpose(tf.matmul(self.pc_exp, exp_params, transpose_b=True), name='pred_exp') # (B, 3*N)
        expressions = tf.transpose(tf.reshape(expressions, [self.batch_size, 3, -1]), [0, 2, 1], name='pred_exp_reshape')  # (B, N, 3)
        mu_reshape = tf.transpose(tf.reshape(self.mu, [3, -1]), [1, 0])
        mu_expand = tf.tile(tf.expand_dims(mu_reshape, axis=0), [self.batch_size, 1, 1], name='mu_expand')
        vertex = mu_expand + shapes + expressions

        # vertex_proj = tf.map_fn(projection_equation, (f, R, vertex, t3d), dtype=tf.float32)
        # Projection
        f_expand = tf.tile(tf.expand_dims(f, axis=2), [1, 3, 3])  # (B, 3, 3)
        t3d_expand = tf.tile(tf.expand_dims(t3d, axis=2), [1, 1, tf.shape(vertex)[1]])  # (B, 3, N)
        vertex_proj = tf.matmul(f_expand * R, tf.transpose(vertex, [0, 2, 1])) + t3d_expand
        # flip vertices along y-axis.
        vertex_proj = tf.concat([tf.slice(vertex_proj, [0, 0, 0], [-1, 1, -1]),
                                 self.im_size - tf.slice(vertex_proj, [0, 1, 0], [-1, 1, -1]) - 1,
                                 tf.slice(vertex_proj, [0, 2, 0], [-1, 1, -1])], axis=1)  #(B, 3, N)

        return vertex_proj


    def rendering_layer(self, vertex_proj, triangles, colors):

        #prepare vertices, triangles, and texture       
        tf_vertex = tf.to_float(vertex_proj, name='vertices')  # (B, 3, N)
        tf_triangles = tf.constant(triangles, tf.float32, name='triangles')  # (3, N)
        tf_texture = tf.tile(tf.expand_dims(tf.constant(colors, tf.float32), axis=0), [self.batch_size, 1, 1], name='texture')  # (B, 3, N)
        tf_image = tf.tile(self.im_gray, [1, 1, 1, 3], name='image')
        # call TF rendering layer
        tf_depth, tf_tex, tf_normal, tf_tri_ind = render_depth(ver=tf_vertex, tri=tf_triangles, texture=tf_texture, image=tf_image)
        
        # 1. pncc result
        pncc_batch = tf.clip_by_value(tf_tex, 1e-6, 1.0)

        # 2. normal map result
        flip_ind = tf_normal[:, :, :, 2] < 0
        tf_normal = tf.where(tf.tile(tf.expand_dims(flip_ind, axis=3), [1, 1, 1, 3]), -1.0 * tf_normal, tf_normal)  # flip the opposite normal vectors
        mag = tf.reduce_sum(tf.square(tf_normal), axis=-1)
        mag = tf.where(mag > 1e-6, mag, tf.zeros_like(mag) + 1.0)  # mag[mag==0.0] = 1.0
        normalimg_batch = tf_normal / tf.expand_dims(tf.sqrt(mag) + 1e-6, axis=3)  # normalization with magnitude, range (-1, 1)

        # 3. masked image result
        mask = tf.clip_by_value(tf_depth, 1e-6, 1.0)  # (B, H, W, 1)
        maskimg_batch = mask * self.im_gray

        # # 4. depth image result
        depthimg_batch = tf.maximum(tf_depth, 1e-6)

        return pncc_batch, normalimg_batch, maskimg_batch, depthimg_batch


    def depth_to_im(self, depth_map):
        '''Map depth data to image for visualization

        :param depth_map:
        :return:
        '''
        depth_map = tf.squeeze(depth_map, axis=3)  # (B, H, W)
        def foreground_normalization(depth_map):
            foreground = tf.gather_nd(depth_map, tf.where(depth_map > 0.0))
            return (depth_map - tf.reduce_min(foreground)) / (tf.reduce_max(foreground) - tf.reduce_min(foreground) + 1e-6)
        depth_norm = tf.map_fn(foreground_normalization, depth_map, dtype=tf.float32)  # -a, 1
        depth_img = tf.maximum(depth_norm, 1e-6) * 255.0 # 0-1

        depth_img = tf.expand_dims(depth_img, axis=3)  # (B, H, W, 1)
        return depth_img


    def projection_equation(self, f, R, vertex, t3d):
        '''The equation of 3D projection onto the plane above the image

        :param f: The scale factor, (1,)
        :param R: The rotation matrix, (3, 3)
        :param vertex: The 3D mesh vertices, (N, 3)
        :param t3d: The 3D translation, (1, 3)
        :return: projected vetices, (N, 3)
        '''
        f_expand = tf.tile(tf.expand_dims(f, axis=1), [3, 3])
        t3d_expand = tf.tile(tf.expand_dims(t3d, axis=1), [3, tf.shape(vertex)[1]])
        projected_vertices = tf.matmul(f_expand * R, vertex, transpose_b=True) + t3d_expand
        return tf.transpose(projected_vertices)


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
        # phi, gamma, theta = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
        phi, gamma, theta = angles
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
        R = np.dot(np.dot(R_pitch, R_yaw), R_roll)
        # R = R_roll.dot(R_yaw.dot(R_pitch))
        return R.astype(np.float32)

    def rotation_matrix_batch(self, angles_batch):
        batch_size = np.shape(angles_batch)[0]
        R_batch = np.zeros([batch_size, 3, 3], dtype=np.float32)
        for i in range(batch_size):
            R_batch[i, :, :] = self.rotation_matrix(angles_batch[i])
        return R_batch


    def depth_rendering_layer(self):
        # Rendering Layer
        scope_name = 'Rendering_Layer'
        with tf.variable_scope(scope_name, scope_name):
            # transforming the predicted params into depth image
            self.vertices_proj = self.vertices_transform(self.pred_params)
            self.pncc_batch, self.normal_batch, self.maskimg_batch, self.coarse_depth_map = \
                self.rendering_layer(self.vertices_proj, self.tri, self.vertex_code)
        return self.coarse_depth_map


    def build_fine_net(self):

        scope_name = 'FineNet'
        with tf.variable_scope(scope_name, scope_name):
            # shape: [batch_size, height, width, channels], channels = 4
            net_input = tf.concat([self.im_gray, self.coarse_depth_map], axis=3, name='input_data')  # (B, H, W, 2)
            # FineNet
            net_conv1 = slim.repeat(net_input, 2, slim.conv2d, 64, [3, 3], scope='conv1_1')
            net_pool1 = slim.max_pool2d(net_conv1, [2, 2], scope='pool1')
            net_conv2 = slim.repeat(net_pool1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net_pool2 = slim.max_pool2d(net_conv2, [2, 2], scope='pool2')
            net_conv3 = slim.repeat(net_pool2, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            # upconvolutional layers
            net_upconv2 = slim.conv2d_transpose(net_conv2, 128, [2, 2], stride=2, scope='upconv2')
            net_upconv3 = slim.repeat(net_conv3, 2, slim.conv2d_transpose, 256, [2, 2], stride=2, scope='upconv3')
            net_conv = tf.concat([net_conv1, net_upconv2, net_upconv3], axis=3, name='hypercolumn_feat')
            # Linear regressor through 1x1 conv
            net_conv = slim.conv2d(net_conv, 50, [1, 1], scope='conv4_1')
            net_conv = slim.conv2d(net_conv, 50, [1, 1], scope='conv4_2')
            net_conv = slim.conv2d(net_conv, 10, [1, 1], scope='conv4_3')
            self.pred_depth_map = slim.conv2d(net_conv, 1, [1, 1], activation_fn=None, normalizer_fn=None, scope='pred_depth_map')

        return self.pred_depth_map


    def get_loss(self):
        ''' Construct loss functions

        :return:
        '''
        with tf.variable_scope('Loss'):
            # pose loss (MSE)
            pose_pred = tf.slice(tf.squeeze(self.pred_params, [1, 2]), [0, 0], [-1, self.ndim_pose], name='pose_pred')  # (batchsize, ndim_pose)
            pose_label = tf.slice(tf.squeeze(self.params_label, [1, 2]), [0, 0], [-1, self.ndim_pose], name='pose_label')
            loss_pose = tf.losses.mean_squared_error(pose_label, pose_pred, scope='pose_loss')
            self.losses['pose_loss'] = loss_pose

            # geometry loss (GMSE)
            geometry_pred = tf.slice(tf.squeeze(self.pred_params, [1, 2]), [0, self.ndim_pose], [-1, self.ndim_shape + self.ndim_exp], name='geometry_pred')  # (batchsize, ndim_shape + ndim_exp)
            geometry_label = tf.slice(tf.squeeze(self.params_label, [1, 2]), [0, self.ndim_pose], [-1, self.ndim_shape + self.ndim_exp], name='geometry_label')
            geometry_basis = tf.concat([self.pc_shape, self.pc_exp], axis=1, name='geometry_basis')
            loss_geometry = tf.losses.mean_squared_error(tf.matmul(geometry_basis, geometry_label, transpose_b=True),
                                                         tf.matmul(geometry_basis, geometry_pred, transpose_b=True),
                                                         scope='geometry_loss')
            self.losses['geometry_loss'] = loss_geometry

            # Shape from Shading (SfS) loss
            intensity_recover = self.get_spherical_harmonics_model()
            loss_sh = tf.losses.mean_squared_error(self.im_gray, intensity_recover, scope='spherical_harmonics_loss')
            self.losses['spherical_harmonics_loss'] = loss_sh

            # fidelity loss of depth map
            loss_fidelity = tf.losses.mean_squared_error(self.coarse_depth_map, self.pred_depth_map, scope='fidelity_loss')
            # loss_fidelity = tf.reduce_mean(tf.square(self.coarse_depth_map - self.pred_depth_map), name='fidelity_loss')
            self.losses['fidelity_loss'] = loss_fidelity

            # smoothness loss of depth map
            filtered_depth = tf.map_fn(lambda x: self.laplace_transform(x), tf.squeeze(self.pred_depth_map, axis=3), dtype=tf.float32, name='laplacian_smoothing')
            loss_smooth = tf.contrib.layers.l1_regularizer(1.0)(filtered_depth)
            self.losses['smoothness_loss'] = loss_smooth

            # final loss function
            loss = self.lambda_pose * loss_pose + self.lambda_geo * loss_geometry + self.lambda_sh * loss_sh + self.lambda_f * loss_fidelity + self.lambda_sm * loss_smooth
            # loss = self.lambda_pose * loss_pose + self.lambda_geo * loss_geometry + self.lambda_f * loss_fidelity + self.lambda_sm * loss_smooth
            self.losses['total_loss'] = loss

            self.scalar_summaries.update(self.losses)
            return self.losses

    
    def laplace_transform(self, x):
        """Compute the 2D laplacian of an array"""
        laplace_k = np.asarray([[0.5, 1.0, 0.5],
                                [1.0, -6., 1.0],
                                [0.5, 1.0, 0.5]])
        laplace_k = laplace_k.reshape(list(laplace_k.shape) + [1,1])
        laplace_k = tf.constant(laplace_k, dtype=np.float32, name='laplace_kernel')

        x = tf.expand_dims(tf.expand_dims(x, 0), -1)
        y = tf.nn.depthwise_conv2d(x, laplace_k, [1, 1, 1, 1], padding='SAME')
        
        return y[0, :, :, 0]

    def compute_abedo_image(self, vertices, triangles, abedos):
        ''' transform the abedos into abedo image format

        :param vertices:
        :param triangles:
        :param abedos:  (3, N)
        :return:
        '''
        # prepare vertices, triangles, and texture
        tf_vertex = tf.to_float(vertices, name='vertices')  # (B, 3, N)
        tf_triangles = tf.constant(triangles, tf.float32, name='triangles')  # (3, N)
        tf_texture = tf.tile(tf.expand_dims(abedos, axis=0), [self.batch_size, 1, 1], name='texture')  # (B, 3, N)
        tf_image = tf.tile(self.im_gray, [1, 1, 1, 3], name='image')
        # call TF rendering layer
        _, tf_abedo, tf_normal, _ = render_depth(ver=tf_vertex, tri=tf_triangles, texture=tf_texture, image=tf_image)

        abedos_image = tf.reduce_mean(tf.maximum(tf_abedo, 1e-6), axis=-1, keep_dims=True)  # (B, H, W, 1)

        flip_ind = tf_normal[:, :, :, 2] < 0
        tf_normal = tf.where(tf.tile(tf.expand_dims(flip_ind, axis=3), [1, 1, 1, 3]), -1.0 * tf_normal, tf_normal)  # flip the opposite normal vectors
        mag = tf.reduce_sum(tf.square(tf_normal), axis=-1)
        mag = tf.where(mag > 1e-6, mag, tf.zeros_like(mag) + 1.0)  # mag[mag==0.0] = 1.0
        normal_map = tf_normal / tf.expand_dims(tf.sqrt(mag) + 1e-6, axis=3)  # normalization with magnitude, range (-1, 1)
        return abedos_image, normal_map


    def get_spherical_harmonics_model(self):

        # transform the (3, N) mu_tex into image shape: (B, H, W, 1)
        abedo_image, normal_map = self.compute_abedo_image(self.vertices_proj, self.tri, self.mu_tex)  # (B, H, W, 1)
        abedo_image = tf.transpose(abedo_image, [1, 2, 3, 0], name='abedos')  # (H, W, 1, B)
        Yz0 = tf.transpose(normal_map, [1, 2, 3, 0], name='homo_normals')  # (H, W, 3, B)  from z0
        # here we use the least square estimation (LSE) with data of batchsize to get recovered lighting:
        # LSE equation: l = (Y x Y_T).inv() x Y x (I./mu_tex)_T
        # TODO: ERROR occured here due to invertible nec. Instead, we use moore-penrose inverse.
        # Yz0_nec_inv = tf.matrix_inverse(tf.matmul(Yz0, tf.transpose(Yz0, [0, 1, 3, 2])))  # (H, W, 3, 3)
        Yz0_nec = tf.matmul(Yz0, tf.transpose(Yz0, [0, 1, 3, 2]))
        Yz0_nec_inv = tf.py_func(np.linalg.pinv, [Yz0_nec], tf.float32)
        I = tf.transpose(self.im_gray, [1, 2, 3, 0])  # (H, W, 1, B)
        lighting_lse = tf.matmul(tf.matmul(Yz0_nec_inv, Yz0),
                                 tf.transpose(I / (abedo_image + 1.0), [0, 1, 3, 2]))  # (H, W, 3, 1)

        '''  # The least square estimation (LSE) for alpha
        # transform the (3N, 10) mu_tex into image shape: (B, H, W, 1)
        PC = tf.expand_dims(self.pc_tex, axis=-1)  # (3*N, 10, 1)
        PC_nec_inv = tf.matrix_inverse(tf.matmul(PC, tf.transpose(PC, [0, 2, 1])))  # (3N, 10, 10)
        diff = I / (tf.matmul(tf.transpose(lighting_lse, [0, 1, 3, 2]), Yz0)) - abedo_image  # (H, W, 1, B)
        # The following step cannot be fulfilled!!
        # Since the result of PC_nec_inv x PC takes the size (3N, 10, 1),
        # which cannot be producted with diff with the size (H, W, 1, B) unless we can inversely transform the diff into (3N, 1, B) space...
        alpha = tf.matmul(tf.matmul(PC_nec_inv, PC), diff)
        '''

        # instead, we use the same average alpha for all batch data, that's to say, the same abedo coefficient
        texture_new = tf.matmul(self.pc_tex, self.param_tex)  # (3*N, 1)
        texture_new = tf.reshape(texture_new, [3, -1])  # (3, N)
        texture_new = self.mu_tex + texture_new
        # Here the self.vertices_proj should be updated according to predicted fine depth
        # But, how to inversely convert predicted fine depth into new vertices, in order to get new normals ??
        # If we still use the self.vertices_proj, the SfS loss will make constraints on coarse depth rather than fine depth!!!
        abedo_image_new, normal_map_new = self.compute_abedo_image(self.vertices_proj, self.tri,
                                                                   texture_new)  # (B, H, W, 1)
        abedo_image_new = tf.transpose(abedo_image_new, [1, 2, 3, 0], name='abedos_new')  # (H, W, 1, B)
        Yz = tf.transpose(normal_map_new, [1, 2, 3, 0], name='homo_normals')  # (H, W, 3, B)  from z
        intensity_recover = abedo_image_new * tf.matmul(tf.transpose(lighting_lse, [0, 1, 3, 2]),
                                                        Yz)  # Eqn (8), (H, W, 1, B)
        intensity_recover = tf.transpose(intensity_recover, [3, 0, 1, 2])  # reformat the shape with (B, H, W, 1)

        return intensity_recover


    def add_summaries(self):
        ''' add all summaries into tensorflow summaries

        :return:
        '''
        val_summaries = []
        with tf.device("/cpu:0"):
            # add summaries of input image
            image = self.im_gray + self.gray_mean
            image = tf.reverse(image, axis=[-1])
            val_summaries.append(tf.summary.image('input image', image))
            # add summaries of predicted vars
            for key, var in self.scalar_summaries.items():
                val_summaries.append(tf.summary.scalar(key, var))
            for key, var in self.image_summaries.items():
                val_summaries.append(tf.summary.image(key, var))
            # for key, var in self.histogram_summaries.items():
            #     tf.summary.histogram(var.op.name, var)

        summary_op = tf.summary.merge_all()
        summary_op_val = tf.summary.merge(val_summaries)
        return summary_op, summary_op_val


    def initialize_models(self, sess):
        # Fresh train directly from ImageNet weights
        print('Loading initial model weights from {:s} and {:s}'.format(self.pretrained_resnet_v1, self.pretrained_vgg16))
        variables = tf.global_variables()
        # Initialize all variables first
        sess.run(tf.variables_initializer(variables, name='init'))
        sess.run(tf.local_variables_initializer())

        variables_to_restore_resv1, variables_to_restore_vgg16 = self.get_variables_in_ckpt(variables, self.pretrained_resnet_v1, self.pretrained_vgg16)

        # Get the variables to restore Res50
        restorer = tf.train.Saver(variables_to_restore_resv1)
        restorer.restore(sess, self.pretrained_resnet_v1)
        # Get the variables to restore VGG16
        restorer = tf.train.Saver(variables_to_restore_vgg16)
        restorer.restore(sess, self.pretrained_vgg16)
        print('Loaded.')

        # Initial file lists are empty
        ss_paths = []
        last_snapshot_iter = 0
        return last_snapshot_iter, ss_paths


    def get_variables_in_ckpt(self, variables, pretrained_resv1, pretrained_vgg16):
        try:
            reader = tf.train.NewCheckpointReader(pretrained_resv1)
            var_keep_dic_resv1 = reader.get_variable_to_shape_map()
            reader = tf.train.NewCheckpointReader(pretrained_vgg16)
            var_keep_dic_vgg16 = reader.get_variable_to_shape_map()
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")

        variables_to_restore_resv1 = {}
        for v in variables:
            vname = v.name.split(':')[0]
            if ('conv_in' in vname) or ('conv_out' in vname):
                continue
            core_vname1, core_vname2 = self.get_core_var_names(vname, 'resnet_v1_101')
            if core_vname1 in var_keep_dic_resv1:
                print('Variables restored: %s' % v.name)
                variables_to_restore_resv1.update({core_vname1: v})
            elif core_vname2 in var_keep_dic_resv1:
                print('Variables restored: %s' % v.name)
                variables_to_restore_resv1.update({core_vname2: v})
            else:
                continue

        variables_to_restore_vgg16 = {}
        for v in variables:
            vname = v.name.split(':')[0]
            core_vname1, core_vname2 = self.get_core_var_names(vname, 'vgg_16')
            if core_vname1 in var_keep_dic_vgg16:
                print('Variables restored: %s' % v.name)
                variables_to_restore_vgg16.update({core_vname1: v})
            elif core_vname2 in var_keep_dic_vgg16:
                print('Variables restored: %s' % v.name)
                variables_to_restore_vgg16.update({core_vname2: v})
            else:
                continue

        return variables_to_restore_resv1, variables_to_restore_vgg16


    def get_core_var_names(self, vname, src_model):
        if '/' in vname:
            # at least one '/'
            p1 = vname.index('/')
            core_vname1 = src_model + '/' + vname[p1 + 1:]
            if '/' in vname[p1 + 1:]:
                # at least two '/'
                p2 = vname[p1 + 1:].index('/')
                core_vname2 = src_model + '/' + vname[p1 + 1:][p2 + 1:]
            else:
                core_vname2 = core_vname1
        else:
            # no '/'
            core_vname1 = 'resnet_v1_101/' + vname
            core_vname2 = core_vname1
        return core_vname1, core_vname2



    def snapshot(self, sess, tf_saver, checkpoints_dir, iter, prefix='FaceRecoNet'):
        # Store the model snapshot
        filename = prefix + '_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(checkpoints_dir, filename)
        tf_saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))
        return filename


    def remove_snapshot(self, ss_paths, keep=5):
        to_remove = len(ss_paths) -keep
        for c in range(to_remove):
            filename = ss_paths[0]
            os.remove(str(filename))
            ss_paths.remove(filename)

        to_remove = len(ss_paths) - keep
        for c in range(to_remove):
            sfile = ss_paths[0]
            # To make the code compatible to earlier versions of Tensorflow,
            # where the naming tradition for checkpoints are different
            if os.path.exists(str(sfile)):
                os.remove(str(sfile))
            else:
                os.remove(str(sfile + '.data-00000-of-00001'))
                os.remove(str(sfile + '.index'))
            sfile_meta = sfile + '.meta'
            os.remove(str(sfile_meta))
            ss_paths.remove(sfile)