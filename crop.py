import pandas as pd
import numpy as np
from scipy import misc
import tifffile as tiff
import pandas as pd
import numpy as np
import os
import cv2
from shapely.wkt import loads as wkt_loads
from shapely import affinity
import matplotlib.pyplot as plt
from os import listdir


class crop:

    # the cropped images are 100*100
    stride = 100

    wholeImage = {
        '6100_1_3',
        '6100_2_2',
        '6100_2_3',
        '6110_1_2',
        '6110_3_1',
        '6110_4_0',
        '6120_2_0',
        '6120_2_2',
        '6140_1_2',
        '6140_3_1'
    }

    # imagenames_3 = listdir('./three_band')
    # df = pd.read_csv('./train_wkt_v4.csv')
    # gs = pd.read_csv('./grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
    # df['ImageId'].unique()
    # trainImageIds = df.ImageId.unique()

    def set_stride(self,stride = 100):
        self.stride = stride

    def mkdir(self):

        if os.path.isdir('./crop_x/') == False:
            os.mkdir('./crop_x')

        if os.path.isdir('./crop_y/') == False:
            os.mkdir('./crop_y')

    def get_image(self,image_id,isY):

        if isY == True:
            filename = './train_y/' + image_id + '.tif'
        else:
            filename = './train_x/' + image_id + '.tif'

        img = tiff.imread(filename)

        if isY == False:
            img = np.array(img)
        else:
            img = np.array(img)
        return img

    def __init__(self):

        return

    def save(self,img,imageId,i,j,isY):

        if isY == False:
            misc.imsave('./crop_x/' + imageId + '_' + str(i) + '_' + str(j) + '.tif', img)
        else:
            misc.imsave('./crop_y/' + imageId + '_' + str(i) + '_' + str(j) + '.tif', img)


    def cropEach(self,img,imgId,isY):

        latch = np.min(img.shape[0:2])

        for i in range(0,latch-self.stride,self.stride):
            for j in range(0, latch-self.stride, self.stride):
                if isY == False:
                    cropped = img[i:i+self.stride,j:j+self.stride,:]
                else:
                    cropped = img[i:i+self.stride,j:j+self.stride]
                self.save(cropped,imgId,i,j,isY)

        for i in range(latch,self.stride,-self.stride):
            for j in range(0, latch-self.stride, self.stride):
                if isY == False:
                    cropped = img[i-self.stride:i,j:j+self.stride,:]
                else:
                    cropped = img[i-self.stride:i,j:j+self.stride]
                self.save(cropped,imgId,i,j,isY)

        for i in range(0,latch-self.stride,self.stride):
            for j in range(latch,self.stride, -self.stride):
                if isY == False:
                    cropped = img[i:i+self.stride,j-self.stride:j,:]
                else:
                    cropped = img[i:i+self.stride,j-self.stride:j]
                self.save(cropped,imgId,i,j,isY)

        for i in range(latch,self.stride, -self.stride):
            for j in range(latch,self.stride, -self.stride):
                if isY == False:
                    cropped = img[i-self.stride:i,j-self.stride:j,:]
                else:
                    cropped = img[i-self.stride:i,j-self.stride:j]
                self.save(cropped,imgId,i,j,isY)

    def cropAndSave(self):

        for imgId in self.wholeImage:
            img = self.get_image(imgId,isY = True)
            self.cropEach(img,imgId ,isY = True)
            img = self.get_image(imgId, isY = False)
            self.cropEach(img,imgId, isY = False)

def main():
    a = crop()
    a.mkdir()
    a.set_stride(100)
    a.cropAndSave()


if __name__=='__main__':
    main()
