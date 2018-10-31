import os, cv2
import numpy as np
from utils.listfile_reader import read_listfile_test

ROOT_PATH = os.path.dirname(__file__)

def compute_mean(filelists, img_size=[200, 200]):
    mean_image = np.zeros([img_size[0], img_size[1]], dtype=np.float32)
    num = 0
    for filepath in enumerate(filelists):
        if not os.path.exists(filepath[1]):
            raise FileNotFoundError
        try:
            # read image with BGR order
            im = cv2.imread(filepath[1])
        except:
            raise IOError
        if len(im.shape) == 2: # gray image already
            im_gray = cv2.resize(im, img_size)
        elif len(im.shape) == 3:
            # R: 0.3, G: 0.59, B: 0.11
            im_gray = 0.3*im[:, :, 2] + 0.59*im[:, :, 1] + 0.11*im[:, :, 0]
        else:
            raise IOError
        num += 1
        mean_image += im_gray
        if num % 1000:
            print('processed %d images.'%(num))
    mean_image /= num
    mean_values = np.mean(mean_image)
    return mean_values, mean_image


if __name__ == '__main__':

    dataset_path = os.path.join(ROOT_PATH , '../data', 'vggface')
    train_filelists = read_listfile_test(dataset_path, 'train_list.txt')

    mean_val, mean_im = compute_mean(train_filelists, img_size=[200, 200])

    cv2.imwrite(os.path.join(dataset_path, 'mean_im.png'), mean_im)
    f = open(os.path.join(dataset_path, 'mean_val_%.3f'%(mean_val)), 'w')
    f.close()