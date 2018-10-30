import os
import numpy as np

ROOT_PATH = os.path.dirname(__file__)

# input data path
dataset_path = os.path.join(ROOT_PATH, '../data/vggface')
listfile = os.path.join(dataset_path, 'filelist.txt')
# output file lists
listfile_train = os.path.join(dataset_path, 'train_list.txt')
listfile_val = os.path.join(dataset_path, 'val_list.txt')
listfile_test = os.path.join(dataset_path, 'test_list.txt')

ratio_train = 0.6
ratio_val = 0.2
np.random.seed(1234)

f = open(listfile)
strline = f.readline()
num = 0
all_list_data = []
while strline:
    strline = strline.strip('\n')
    all_list_data.append(strline)
    num += 1
    if strline == '':
        break
    if num % 1000 == 0:
        print('get samples num: %d'%(num))
    strline = f.readline()
f.close()
print('Total samples: %d'%(num))

assert (len(all_list_data) == num), 'Invalid listfile!'
# random shuffling the dataset
np.random.shuffle(all_list_data)

# split the list file into train val and test
p1 = int(np.floor(num*ratio_train))
p2 = p1 + int(np.floor(num*ratio_val))
train_list = all_list_data[0:p1]
val_list = all_list_data[p1:p2]
test_list = all_list_data[p2:]

# write result list file
f = open(listfile_train, 'w')
f.write('\n'.join(train_list))
f.close()

f = open(listfile_val, 'w')
f.write('\n'.join(val_list))
f.close()

f = open(listfile_test, 'w')
f.write('\n'.join(test_list))
f.close()










