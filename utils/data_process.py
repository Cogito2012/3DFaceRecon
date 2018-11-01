import numpy as np
import cv2
import os

def prepare_input_image(image_files, batch_size, img_size):
    ''' prepare the input image with 4-D tensor shape

    :param image_files: a batch of image files
    :param batch_size: batch size
    :param img_size: image size, [h, w]
    :return: 4-D data, [batchsize, h, w, 1]
    '''
    assert len(image_files) == batch_size
    input_image = np.zeros([batch_size, img_size[0], img_size[1], 1])
    for i in range(batch_size):
        if not os.path.exists(image_files[i]):
            raise FileNotFoundError
        try:
            # read image with BGR order
            im = cv2.imread(image_files[i])
        except:
            raise IOError
        if len(im.shape) == 2: # gray image already
            im = cv2.resize(im, img_size)
            input_image[i, :, :, 0] = im
        elif len(im.shape) == 3:
            # R: 0.3, G: 0.59, B: 0.11
            im_gray = 0.3*im[:, :, 2] + 0.59*im[:, :, 1] + 0.11*im[:, :, 0]
            input_image[i, :, :, 0] = im_gray
        else:
            raise IOError
    return input_image



def prepare_input_label(label_files, batch_size, label_dim):
    '''prepare the input label with 2-D tensor shape

    :param label_files: a batch of label files
    :param batch_size: batch size
    :param label_dim: label dim (235,)
    :return: 2-D data, (batchsize, label_dim)
    '''
    assert len(label_files) == batch_size
    input_label = np.zeros([batch_size, 1, 1, label_dim])
    for i in range(batch_size):
        if not os.path.exists(label_files[i]):
            raise FileNotFoundError
        try:
            # read image with BGR order
            labels = np.loadtxt(label_files[i])
        except:
            raise IOError
        if labels.shape[0] == label_dim: # gray image already
            input_label[i, 0, 0, :] = labels
        else:
            raise IOError
    return input_label