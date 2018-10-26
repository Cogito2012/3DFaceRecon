import numpy as np
import tensorflow as tf
from math import cos, sin
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
import mesh_render.mesh_core_cython as mesh_core_cython



class FaceRecNet:
    def __init__(self, im_gray=None, params_label=None, modeldata_3dmm=None, nIter=4, batch_size=64, im_size=200, weight_decay=1e-4):
        self.im_gray = im_gray
        self.params_label = params_label
        self.modeldata_3dmm = modeldata_3dmm
        self.nIter = nIter
        self.batch_size = batch_size
        self.im_size = im_size
        self.weight_decay = weight_decay
        self.depth_map = None

        self.vertex_code = modeldata_3dmm['vertex']
        self.tri = modeldata_3dmm['tri']
        self.mu = tf.constant(modeldata_3dmm['mu'], dtype=tf.float32)  # (3*Nvert, 1)
        self.pc_shape = tf.constant(modeldata_3dmm['pc_shape'], dtype=tf.float32)  # (3*Nvert, ndim_shape)
        self.pc_exp = tf.constant(modeldata_3dmm['pc_exp'], dtype=tf.float32)      # (3*Nvert, ndim_exp)
        self.ndim_shape = modeldata_3dmm['ndim_shape']
        self.ndim_exp = modeldata_3dmm['ndim_exp']
        self.ndim_pose = modeldata_3dmm['ndim_pose']
        self.ndim = self.ndim_pose + self.ndim_shape + self.ndim_exp

        # construct geometry and pose parameters, they will be updated in the training
        geo_params = tf.zeros((self.batch_size, self.ndim_shape + self.ndim_exp), dtype=tf.float32)
        init_pose = tf.constant([0, 0, 0, self.im_size / 2.0, self.im_size / 2.0, 0, 0.001], dtype=tf.float32)
        pose_params = tf.tile(tf.expand_dims(init_pose, axis=0), [batch_size, 1])
        self.pred_params = tf.concat([pose_params, geo_params], axis=1, name='pred_params')  # pose + shape + expression

        self.summaries = {}


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
            self.summaries.update({'pncc': self.pncc_batch,
                                   'maskimg': self.maskimg_batch,
                                   'pred_params': self.pred_params})
            # Rendering Layer
            self.depth_rendering_layer()
            self.summaries.update({'coarse_depth': self.coarse_depth_map})

            # FineNet
            self.build_fine_net()
            self.summaries.update({'fine_depth': self.pred_depth_map})

        return self.pred_depth_map



    def build_coarse_net(self):
        '''

        :param is_training:
        :return:
        '''
        for n in range(self.nIter):
            scope_name = 'Input_Rendering_iter%d'%(n)
            with tf.variable_scope(scope_name, scope_name):
                # transforming the predicted params into rendered PNCC and Masked Image
                self.pncc_batch, self.maskimg_batch = self.input_rendering_layer(self.pred_params)

            scope_name = 'CoarseNet_iter_iter%d'%(n)
            with tf.variable_scope(scope_name, scope_name):
                # shape: [batch_size, height, width, channels], channels = 4
                net_input = tf.concat([self.maskimg_batch, self.pncc_batch], axis=3, name='input_data')
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
        pose_params = tf.slice(pred_params, [0, 0], [-1, self.ndim_pose], name='pred_pose_params')  # (batchsize, ndim_pose)
        shape_params = tf.slice(pred_params, [0, self.ndim_pose], [-1, self.ndim_shape], name='pred_shape_params')  # (batchsize, ndim_shape)
        exp_params = tf.slice(pred_params, [0, self.ndim_pose + self.ndim_shape], [-1, self.ndim_exp], name='pred_exp_params') # (batchsize, ndim_exp)
        # pose parameters
        phi, gamma, theta, t3d, f = self.parse_pose_params(pose_params)
        # R here is not computed by TF API, but numpy.
        # TODO: pure TF implementation to calculate R.
        R = tf.py_func(self.rotation_matrix_batch, [tf.concat([phi, gamma, theta], axis=1)], tf.float32)
        R.set_shape([self.batch_size, 3, 3])

        shapes = tf.transpose(tf.matmul(self.pc_shape, shape_params, transpose_b=True), name='pred_shape') # (batchsize, 3*Nvert)
        shapes = tf.reshape(shapes, [self.batch_size, -1, 3], name='pred_shape_reshape')
        expressions = tf.transpose(tf.matmul(self.pc_exp, exp_params, transpose_b=True), name='pred_exp') # (batchsize, 3*Nvert)
        expressions = tf.reshape(expressions, [self.batch_size, -1, 3], name='pred_exp_reshape')
        mu_expand = tf.tile(tf.expand_dims(tf.reshape(self.mu, [-1, 3]), axis=0), [self.batch_size, 1, 1], name='mu_expand')
        vertex = mu_expand + shapes + expressions

        # vertex_proj = tf.map_fn(projection_equation, (f, R, vertex, t3d), dtype=tf.float32)
        # Projection
        f_expand = tf.tile(tf.expand_dims(f, axis=2), [1, 3, 3])  # (B, 3, 3)
        t3d_expand = tf.tile(tf.expand_dims(t3d, axis=2), [1, 1, tf.shape(vertex)[1]])  # (B, 3, N)
        vertex_proj = tf.matmul(f_expand * R, tf.transpose(vertex, [0, 2, 1])) + t3d_expand
        vertex_proj = tf.transpose(vertex_proj, [0, 2, 1])
        # flip vertices along y-axis.
        vertex_proj = tf.concat([tf.slice(vertex_proj, [0, 0, 0], [-1, -1, 1]),
                                 self.im_size - tf.slice(vertex_proj, [0, 0, 1], [-1, -1, 1]) - 1,
                                 tf.slice(vertex_proj, [0, 0, 2], [-1, -1, 1])], axis=2)

        # start rendering with z_buffer renderer
        pncc_batch, maskimg_batch = tf.py_func(self.zbuffer_rendering_py, [vertex_proj], [tf.float32, tf.float32])
        pncc_batch.set_shape([self.batch_size, self.im_size, self.im_size, 3])
        maskimg_batch.set_shape([self.batch_size, self.im_size, self.im_size, 1])

        return pncc_batch, maskimg_batch


    def zbuffer_rendering_py(self, vertices_proj):
        pncc_batch = np.zeros((self.batch_size, self.im_size, self.im_size, 3), dtype=np.float32)
        maskimg_batch = np.zeros((self.batch_size, self.im_size, self.im_size, 1), dtype=np.float32)
        for i in range(self.batch_size):
            code_map = pncc_batch[i, :, :, :].copy()
            depth_buffer = np.zeros([self.im_size, self.im_size], dtype=np.float32, order='C') - 999999.
            # We use the Cython package based on C++ implementation
            mesh_core_cython.render_colors_core(
                code_map, vertices_proj, self.tri, self.vertex_code,
                depth_buffer,
                vertices_proj.shape[0], self.tri.shape[0],
                self.im_size, self.im_size, 3)
            # binarization for masking
            mask = np.minimum(np.maximum(depth_buffer, 0.0), 1.0)
            maskimg = tf.expand_dims(mask, axis=2) * self.im_gray[i]
            pncc_batch[i, :, :, :] = code_map
            maskimg_batch[i, :, :, :] = maskimg

        return pncc_batch, maskimg_batch



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

        R = R_roll.dot(R_yaw.dot(R_pitch))
        return R.astype(np.float32)

    def rotation_matrix_batch(self, angles_batch):
        batch_size = np.shape(angles_batch)[0]
        R_batch = np.zeros([batch_size, 3, 3], dtype=np.float32)
        for i in range(batch_size):
            R_batch[i, :, :] = self.rotation_matrix(angles_batch[i])
        return R_batch


    def depth_rendering_layer(self):
        # TODO
        self.coarse_depth_map = tf.zeros((self.batch_size, self.im_size, self.im_size, 1), dtype=tf.float32)


    def build_fine_net(self):

        scope_name = 'FineNet'
        with tf.variable_scope(scope_name, scope_name):
            # shape: [batch_size, height, width, channels], channels = 4
            net_input = tf.concat([self.im_gray, self.coarse_depth_map], axis=3, name='input_data')
            # FineNet
            net_conv1 = slim.repeat(net_input, 2, slim.conv2d, 64, [3, 3], scope='conv1')
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
            self.pred_depth_map = slim.conv2d(net_conv, 1, [1, 1], scope='pred_depth_map')

        return self.pred_depth_map


    def get_loss(self):
        #
        with tf.variable_scope('Loss'):
            # pose loss (MSE)
            pose_pred = tf.slice(self.pred_params, [0, 0], [-1, self.ndim_pose], name='pose_pred')  # (batchsize, ndim_pose)
            pose_label = tf.slice(self.params_label, [0, 0], [-1, self.ndim_pose], name='pose_label')
            loss_pose = tf.losses.mean_squared_error(pose_label, pose_pred, scope='pose_loss')
            self.summaries.update({'pose_loss': loss_pose})

            # geometry loss (GMSE)
            geometry_pred = tf.slice(self.pred_params, [0, self.ndim_pose], [-1, self.ndim_shape + self.ndim_exp], name='geometry_pred')  # (batchsize, ndim_shape + ndim_exp)
            geometry_label = tf.slice(self.params_label, [0, self.ndim_pose], [-1, self.ndim_shape + self.ndim_exp], name='geometry_label')
            geometry_basis = tf.concat([self.pc_shape, self.pc_exp], axis=0, name='geometry_basis')
            loss_geometry = tf.losses.mean_squared_error(tf.matmul(geometry_basis, geometry_label, transpose_b=True),
                                                         tf.matmul(geometry_basis, geometry_pred, transpose_b=True),
                                                         scope='geometry_loss')
            self.summaries.update({'geometry_loss': loss_geometry})

            # Shape from Shading (SfS) loss
            # TODO