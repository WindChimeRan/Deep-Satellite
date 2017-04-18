import os
import numpy as np
from scipy import misc


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

if __name__ == '__main__':

    wash_data()