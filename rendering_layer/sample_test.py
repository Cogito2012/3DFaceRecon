import numpy as np
import os #, cv2
ROOT_PATH = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(ROOT_PATH, '../utils'))

from scipy.misc import imread, imsave, imshow, imresize, imsave
import parser_3dmm as parser_3dmm
from numpy.random import rand
from math import sin, cos
import scipy.io as sio

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
from ops import render_depth
import numpy as np

ROOT_PATH = os.path.dirname(__file__)

def get_random_params(im_size, num_shape_param, num_exp_param, beta=1.0):

    phi = (-75 + 120 * rand()) * np.pi/180  # [-75, 45]
    gamma = (-90 + 180 * rand()) * np.pi/180  # [-90, 90]
    theta = (-30 + 60 * rand()) * np.pi/180  # [-20, 20]
    focal_factor = rand() * 1e-3
    t3d = np.vstack([rand(2, 1) * (60), np.array([[0.0]], dtype=np.float32)])
    pose_param_rand = np.vstack([np.array([[phi], [gamma], [theta]], dtype=np.float32), t3d,
                                 np.array([[focal_factor]], dtype=np.float32)])
    pose_param_base = np.reshape(np.array([0, 0, 0, im_size/2, im_size/2, 0, 0.001], dtype=np.float32), [7, 1])
    pose_param = beta * pose_param_base + (1 - beta) * pose_param_rand
    shape_param = rand(num_shape_param, 1) * 1e04
    exp_param = -1.5 + 3 * rand(num_exp_param, 1)
    #shape_param = np.zeros([num_shape_param, 1], dtype=np.float32)
    #exp_param = np.zeros([num_exp_param, 1], dtype=np.float32)
    return pose_param, shape_param, exp_param

def parse_pose_params(pose_param):
    phi = pose_param[0, 0]
    gamma = pose_param[1, 0]
    theta = pose_param[2, 0]
    t3d = pose_param[3:6, 0]
    f = pose_param[6, 0]
    return phi, gamma, theta, t3d, f

def rotation_matrix(angles):
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


def main():
    model_3dmm_path = os.path.join(ROOT_PATH, '../3dmm')
    modeldata_3dmm = parser_3dmm.read_3dmm_model(model_3dmm_path)
    vertex_code = modeldata_3dmm['vertex']
    tri = modeldata_3dmm['tri']
    mu = modeldata_3dmm['mu']  # (3*Nvert, 1)
    pc_shape = modeldata_3dmm['pc_shape']  # (3*Nvert, ndim_shape)
    pc_exp = modeldata_3dmm['pc_exp']  # (3*Nvert, ndim_exp)
    ndim_shape = modeldata_3dmm['ndim_shape']
    ndim_exp = modeldata_3dmm['ndim_exp']
    ndim_pose = modeldata_3dmm['ndim_pose']
    nvert = int(np.floor(np.shape(mu)[0] / 3.0))
    im_size = 200
    im = imread(os.path.join(ROOT_PATH, '../samples/face_images/Ana_Ivanovic', '00000045.jpg'))
    im = np.resize(im, [im_size, im_size, 3])

    pose_param, shape_param, exp_param = get_random_params(im_size, ndim_shape, ndim_exp, beta=0.8)
    phi, gamma, theta, t3d, f = parse_pose_params(pose_param)
    R = rotation_matrix([phi, gamma, theta])

    shapes = pc_shape.dot(shape_param)
    shapes = np.transpose(np.reshape(shapes, [3, nvert]), [1, 0])  # (nvert, 3)
    expressions = pc_exp.dot(exp_param)
    expressions = np.transpose(np.reshape(expressions, [3, nvert]), [1, 0])  # (nvert, 3)
    mu = np.transpose(np.reshape(mu, [nvert, 3]), [1, 0])  #(3, nvert)
    vertex = mu + np.transpose(shapes + expressions) 
    
    vertex_proj = np.dot(f * R, vertex) + np.tile(np.expand_dims(t3d, axis=1), [1, nvert])
    vertex_proj[1, :] = im_size - vertex_proj[1, :]

    vertex_proj = vertex_proj.astype(np.float32)
    tri = tri.astype(np.float32)
    texture = vertex_code.astype(np.float32)

    with tf.device("/device:GPU:0"):
        tf_vertex = tf.Variable(tf.constant(np.expand_dims(vertex_proj,axis=0)))
        tf_triangles = tf.Variable(tf.constant(tri))
        tf_texture = tf.Variable(tf.constant(np.expand_dims(texture,axis=0)))
        tf_image =tf.Variable(tf.constant(np.expand_dims(im.astype(np.float32)/255.0,axis=0)))

        tf_depth,tf_tex, tf_tri_ind  =render_depth(ver=tf_vertex,tri=tf_triangles,texture = tf_texture,image=tf_image)

        # must init the vertex and tri before depth.eval()
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_g)
            sess.run(init_l)

            tf_depth_value,tf_tex_value, tf_tri_ind_value = sess.run([tf_depth, tf_tex, tf_tri_ind])

            tf_vertex_value = tf_vertex.eval()
            tf_tri_value = tf_triangles.eval()
            tf_image_value = tf_image.eval()
            
            # code_map (PNCC), depth_buffer (depth image)
            depth_buffer = tf_depth_value[0,:,:,0]
            code_map = tf_tex_value[0,:,:,:]

    # binarization for masking
    mask = np.minimum(np.maximum(depth_buffer, 0.0), 1.0)
    maskimg = np.tile(np.expand_dims(mask, axis=2), [1, 1, 3]) * im
    #depth image
    #depthimg = (depth_buffer - np.min(depth_buffer))/(np.max(depth_buffer) - np.min(depth_buffer))*255.0
    ind = np.where(depth_buffer > 0.0)
    depthimg = (depth_buffer - np.min(depth_buffer[ind]))/(np.max(depth_buffer[ind]) - np.min(depth_buffer[ind]))
    depthimg = np.maximum(depthimg, 0.0)*255.0
    
    imsave('test_pncc_tf_gpu.png',np.round(np.clip(code_map,0.0,1.0) * 255.0).astype(np.uint8))
    imsave('test_maskimg_tf_gpu.png',np.round(maskimg).astype(np.uint8))
    imsave('test_depthimg_tf_gpu.png', np.round(depthimg).astype(np.uint8))

if __name__ == '__main__':
    main()
