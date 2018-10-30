import numpy as np
import tensorflow as tf
from math import cos, sin
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block

from rendering_layer.ops import render_depth

class FaceRecNet:
    def __init__(self, im_gray=None, params_label=None, mesh_data=None, nIter=4, batch_size=64, im_size=200, weight_decay=1e-4):
        self.im_gray = im_gray
        self.params_label = params_label
        self.nIter = nIter
        self.batch_size = batch_size
        self.im_size = im_size
        self.weight_decay = weight_decay
        self.ndim_tex = 10
        self.lambda_sh = 1.0
        self.lambda_f = 5e-3
        self.lambda_sm = 1.0

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
            self.summaries.update({
                                   'pred_params': self.pred_params})
            # Rendering Layer
            self.depth_rendering_layer()
            self.summaries.update({'pncc_map': self.pncc_batch,
                                   'mask_im': self.maskimg_batch,
                                   'normal_map': self.normal_batch,
                                   'coarse_depth': self.coarse_depth_map})

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
                # transforming the predicted params into vertices
                vertices_proj = self.vertices_transform(self.pred_params)
                # rendered PNCC, normal map and Masked Image
                pncc_batch, normal_batch, maskimg_batch, _ = \
                    self.rendering_layer(vertices_proj, self.tri, self.vertex_code)

            scope_name = 'CoarseNet_iter_iter%d'%(n)
            with tf.variable_scope(scope_name, scope_name):
                # shape: [batch_size, height, width, channels], channels = 7
                net_input = tf.concat([maskimg_batch, pncc_batch, normal_batch], axis=3, name='input_data')
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


    def vertices_transform(self, pred_params):
        # parse geometry and pose parameters from prediction
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
        # call TF rendering layer
        tf_depth, tf_tex, tf_normal, tf_tri_ind = render_depth(ver=tf_vertex, tri=tf_triangles, texture=tf_texture, image=self.im_gray)
        
        # 1. pncc result
        pncc_batch = tf.clip_by_value(tf_tex, 0.0, 1.0)

        # 2. normal map result
        mag = tf.reduce_sum(tf.square(tf_normal), axis=-1) + 1.0  # here we add 1 to avoid meaningless division for normalization
        normal_map = tf.maximum(tf_normal, 0.0)  # if v <0, v == 0
        normal_map = normal_map / tf.tile(tf.expand_dims(tf.sqrt(mag), axis=3), [1, 1, 1, 3])  # normalize vertices normals
        normal_batch = tf.map_fn(lambda x: (x - tf.reduce_min(x, axis=[0, 1])) /
                                           (tf.reduce_max(x, axis=[0, 1]) - tf.reduce_min(x, axis=[0, 1])), normal_map, dtype=tf.float32)
        normalimg_batch = tf.map_fn(lambda x: tf.maximum(x, 0.0) * 255.0, normal_batch, dtype=tf.float32)

        # 3. masked image result
        mask = tf.minimum(tf.maximum(tf_depth, 0.0), 1.0)  # (B, H, W, 1)
        maskimg_batch = mask * self.im_gray

        # 4. depth image result
        tf_depth = tf.squeeze(tf_depth)  # (B, H, W)
        foreground = tf.map_fn(lambda x: tf.gather_nd(x, tf.where(x > 0.0)), tf_depth, dtype=tf.float32)
        tf_depth_norm = tf.map_fn(lambda x: (x[0] - tf.reduce_min(x[1])) / (tf.reduce_max(x[1]) - tf.reduce_min(x[1])), (tf_depth, foreground), dtype=tf.float32)
        depthimg_batch = tf.map_fn(lambda x: tf.maximum(x, 0.0)*255.0, tf_depth_norm, dtype=tf.float32)
        depthimg_batch = tf.expand_dims(depthimg_batch, axis=3)  # (B, H, W, 1)

        return pncc_batch, normalimg_batch, maskimg_batch, depthimg_batch


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
        ''' Construct loss functions

        :return:
        '''
        with tf.variable_scope('Loss'):
            # pose loss (MSE)
            pose_pred = tf.slice(self.pred_params, [0, 0], [-1, self.ndim_pose], name='pose_pred')  # (batchsize, ndim_pose)
            pose_label = tf.slice(self.params_label, [0, 0], [-1, self.ndim_pose], name='pose_label')
            loss_pose = tf.losses.mean_squared_error(pose_label, pose_pred, scope='pose_loss')
            self.summaries.update({'pose_loss': loss_pose})

            # geometry loss (GMSE)
            geometry_pred = tf.slice(self.pred_params, [0, self.ndim_pose], [-1, self.ndim_shape + self.ndim_exp], name='geometry_pred')  # (batchsize, ndim_shape + ndim_exp)
            geometry_label = tf.slice(self.params_label, [0, self.ndim_pose], [-1, self.ndim_shape + self.ndim_exp], name='geometry_label')
            geometry_basis = tf.concat([self.pc_shape, self.pc_exp], axis=1, name='geometry_basis')
            loss_geometry = tf.losses.mean_squared_error(tf.matmul(geometry_basis, geometry_label, transpose_b=True),
                                                         tf.matmul(geometry_basis, geometry_pred, transpose_b=True),
                                                         scope='geometry_loss')
            self.summaries.update({'geometry_loss': loss_geometry})

            # Shape from Shading (SfS) loss
            intensity_recover = self.get_spherical_harmonics_model()
            loss_sh = tf.losses.mean_squared_error(self.im_gray, intensity_recover, scope='spherical_harmonics_loss')
            self.summaries.update({'spherical_harmonics_loss': loss_sh})

            # fidelity loss of depth map
            loss_fidelity = tf.losses.mean_squared_error(self.coarse_depth_map, self.pred_depth_map, scope='fidelity_loss')
            self.summaries.update({'fidelity_loss': loss_fidelity})

            # smoothness loss of depth map
            filtered_depth = tf.map_fn(lambda x: self.laplace_transform(x), tf.squeeze(self.pred_depth_map), dtype=tf.float32, name='laplacian_smoothing')
            loss_smooth = tf.contrib.layers.l1_regularizer(1.0)(filtered_depth)
            self.summaries.update({'smoothness_loss': loss_smooth})

            # final loss function
            loss = self.lambda_sh * loss_sh + self.lambda_f * loss_fidelity + self.lambda_sm * loss_smooth
            self.summaries.update({'total_loss', loss})

            return loss

    
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
        # call TF rendering layer
        _, tf_abedo, tf_normal, _ = render_depth(ver=tf_vertex, tri=tf_triangles, texture=tf_texture, image=self.im_gray)

        abedos_image = tf.reduce_mean(tf.maximum(tf_abedo, 0.0), axis=-1, keep_dims=True)  # (B, H, W, 1)
        mag = tf.reduce_sum(tf.square(tf_normal), axis=-1) + 1.0  # here we add 1 to avoid meaningless division for normalization
        normal_map = tf.maximum(tf_normal, 0.0)  # if v <0, v == 0
        normal_map = normal_map / tf.tile(tf.expand_dims(tf.sqrt(mag), axis=3), [1, 1, 1, 3])  # normalize vertices normals

        return abedos_image, normal_map


    def get_spherical_harmonics_model(self):

        # transform the (3, N) mu_tex into image shape: (B, H, W, 1)
        abedo_image, normal_map = self.compute_abedo_image(self.vertices_proj, self.tri, self.mu_tex)  # (B, H, W, 1)
        abedo_image = tf.transpose(abedo_image, [1, 2, 3, 0], name='abedos')  # (H, W, 1, B)
        Yz0 = tf.transpose(normal_map, [1, 2, 3, 0], name='homo_normals')  # (H, W, 3, B)  from z0
        # here we use the least square estimation (LSE) with data of batchsize to get recovered lighting:
        # LSE equation: l = (Y x Y_T).inv() x Y x (I./mu_tex)_T
        Yz0_nec_inv = tf.matrix_inverse(tf.matmul(Yz0, tf.transpose(Yz0, [0, 1, 3, 2])))  # (H, W, 3, 3)
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