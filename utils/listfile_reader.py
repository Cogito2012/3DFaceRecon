import os


def read_listfile_trainval(dataset_path, filename):
    '''File reader for list in train and val

    :param dataset_path:
    :param filename:
    :return: imagefiles and label files
    '''
    listfile = os.path.join(dataset_path, filename)
    f = open(listfile)
    strline = f.readline()
    imgfile_results = []
    labelfile_results = []
    while strline:
        strline = strline.strip('.txt\n')
        if strline == '':
            break
        imgfile_abspath = os.path.join(dataset_path, 'face_images', strline + '.jpg')
        imgfile_results.append(imgfile_abspath)
        labelfile_abspath = os.path.join(dataset_path, 'labels', strline + '.txt')
        labelfile_results.append(labelfile_abspath)
        # read the next line
        strline = f.readline()
    f.close()
    return imgfile_results, labelfile_results


def read_listfile_test(dataset_path, filename):
    '''File reader for list in test

    :param dataset_path:
    :param filename:
    :return: imagefiles and label files
    '''
    listfile = os.path.join(dataset_path, filename)
    f = open(listfile)
    strline = f.readline()
    imgfile_results = []
    while strline:
        strline = strline.strip('.txt\n')
        if strline == '':
            break
        imgfile_abspath = os.path.join(dataset_path, 'face_images', strline + '.jpg')
        imgfile_results.append(imgfile_abspath)
        # read the next line
        strline = f.readline()
    f.close()
    return imgfile_results