import scipy.io as sio
import os
import numpy as np


def parse_3dmm_files(model_shapefile, model_expfile, vertex_codefile):
    '''Parse params and data from 3DMM model files

    :param model_shapefile:
    :param model_expfile:
    :param vertex_codefile:
    :return:
    '''
    assert os.path.exists(model_shapefile), 'File %s does not exist!' % (model_shapefile)
    data = sio.loadmat(model_shapefile)
    mu_shape = data['mu_shape']
    pc_shape = data['w']
    tri = data['tri']

    assert os.path.exists(model_expfile), 'File %s does not exist!' % (model_expfile)
    data = sio.loadmat(model_expfile)
    mu_exp = data['mu_exp']
    pc_exp = data['w_exp']

    assert os.path.exists(vertex_codefile), 'File %s does not exist!' % (vertex_codefile)
    data = sio.loadmat(vertex_codefile)
    vertex_code = data['vertex_code']

    mu = mu_shape + mu_exp
    return vertex_code, tri, mu, pc_shape, pc_exp


def read_3dmm_model(model_path):
    ''' Parse the 3dmm parameters data

    :return: model_params
    '''
    # exterior data and model params
    model_shapefile = os.path.join(model_path, 'Model_Shape.mat')
    model_expfile = os.path.join(model_path, 'Model_Expression.mat')
    vertex_codefile = os.path.join(model_path, 'vertex_code.mat')
    vertex_code, tri, mu, pc_shape, pc_exp = parse_3dmm_files(model_shapefile, model_expfile, vertex_codefile)
    ndim_shape = np.shape(pc_shape)[1]
    ndim_exp = np.shape(pc_exp)[1]
    ndim_pose = 7
    model_params = {'vertex': vertex_code,
                    'tri': tri,
                    'mu': mu,
                    'pc_shape': pc_shape,
                    'pc_exp': pc_exp,
                    'ndim_shape': ndim_shape,
                    'ndim_exp': ndim_exp,
                    'ndim_pose': ndim_pose}
    return model_params
