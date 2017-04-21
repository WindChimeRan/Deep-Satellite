import os
import numpy as np
from scipy import misc
from random import shuffle


def wash_data():

    dirx = "/raid/zhr/Deep-Satellite/data/crop_x"
    diry = "/raid/zhr/Deep-Satellite/data/crop_y"
    files = list(os.walk(diry))[0][2]

    for f in files:

        img = misc.imread(diry+'/'+f)
        if np.sum(img) < 1:
            os.remove(diry+'/'+f)
            os.remove(dirx+'/'+f)

    # print(len(files))


def split_data():

    # 18215 intotal
    # 1821 cross_validation 1821 test 14568
    dirx = "/raid/zhr/Deep-Satellite/data/crop_x"
    diry = "/raid/zhr/Deep-Satellite/data/crop_y"
    files = list(os.walk(diry))[0][2]
    shuffle(files)

    total = len(files)
    test_size = total//10
    test_files_name = files[:test_size]
    validation_files_name = files[test_size:2*test_size]
    train_files_name= files[2*test_size:]

    return test_files_name,validation_files_name,train_files_name

    # for f in files:
    #
    #     img = misc.imread(diry+'/'+f)


    # print(len(files))

if __name__ == '__main__':

    # wash_data()
    split_data()