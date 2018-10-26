import scipy.io as sio
import os
import numpy as np


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


def read_3dmm_model():
    ''' Parse the 3dmm parameters data

    :return: model_params
    '''
    # exterior data and model params
    morphable_model = './3dmm/01_MorphableModel.mat'
    model_infofile = './3dmm/model_info.mat'
    model_expfile = './3dmm/Model_Expression.mat'
    vertex_codefile = './3dmm/vertex_code.mat'
    vertex_code, tri, mu, pc_shape, pc_exp = \
        parse_3dmm_files(morphable_model, model_infofile, model_expfile, vertex_codefile)
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
