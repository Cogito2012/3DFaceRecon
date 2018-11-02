import numpy as np
import cv2
import os
import utils.listfile_reader as file_reader
ROOT_PATH = os.path.join(os.path.dirname(__file__), '..')

def prepare_input_image(image_files, batch_size, img_size, img_mean=127.0):
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
            im = cv2.resize(im, img_size)  # in BGR order
            input_image[i, :, :, 0] = im
        elif len(im.shape) == 3:
            # R: 0.3, G: 0.59, B: 0.11
            im_gray = 0.3*im[:, :, 2] + 0.59*im[:, :, 1] + 0.11*im[:, :, 0]
            input_image[i, :, :, 0] = im_gray - img_mean
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


def trainval_generator(batch_size, img_size, label_dim, dataset=None, img_mean=127.0, phase='train'):
    '''The data generator to fetch a batch of images and labels from disk

    :param batch_size:
    :param img_size:
    :param label_dim:
    :param phase:
    :return:
    '''
    dataset_path = os.path.join(ROOT_PATH, 'data', dataset)
    if phase == 'train':
        image_files, label_files = file_reader.read_listfile_trainval(dataset_path, 'train_list.txt')
    elif phase == 'val':
        image_files, label_files = file_reader.read_listfile_trainval(dataset_path, 'val_list.txt')
    else:
        raise NotImplementedError

    counter = 0
    while True:
        yield prepare_input_image(image_files[counter:counter + batch_size], batch_size, img_size, img_mean), \
              prepare_input_label(label_files[counter:counter + batch_size], batch_size, label_dim)
        counter = (counter + batch_size) % len(image_files)


def test_generator(batch_size, img_size, dataset=None, img_mean=127.0):
    '''The data generator to fetch a batch of test images from disk

    :param batch_size:
    :param img_size:
    :return:
    '''
    dataset_path = os.path.join(ROOT_PATH, 'data', dataset)
    image_files = file_reader.read_listfile_test(dataset_path, 'test_list.txt')

    counter = 0
    while True:
        yield prepare_input_image(image_files[counter:counter + batch_size], batch_size, img_size, img_mean), \
              image_files[counter:counter + batch_size]
        counter = (counter + batch_size) % len(image_files)


def data_generator_test():
    '''Validate the data generators of trainval and test

    :return:
    '''
    batch_size = 2
    img_size = [200, 200]
    ndim_params = 235
    img_mean = 126.064
    traindata_generator = trainval_generator(batch_size, img_size, ndim_params,
                                             dataset='vggface', img_mean=img_mean, phase='train')
    # Get training data, one batch at a time
    # [Note: for python2.7 users, use the traindata_generator.next()]
    train_images, train_labels = next(traindata_generator)
    for i, img_in in enumerate(train_images):
        img_out = img_in + img_mean
        cv2.imwrite('test_%d_correct.png'%i, img_out)

    testdata_generator = test_generator(batch_size, img_size, dataset='vggface', img_mean=img_mean)
    test_images, image_files = next(testdata_generator)
    for filename in image_files:
        print(filename)
        obj_id = os.path.splitext(filename)[0].split('/')[-2]
        file_id = os.path.splitext(filename)[0].split('/')[-1]
        print(obj_id)
        print(file_id)


if __name__ == '__main__':
    data_generator_test()