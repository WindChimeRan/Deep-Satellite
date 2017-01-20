import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2


class img2tfrecords:

    crop_x = './crop_x/'
    crop_y = './crop_y/'
    filename = './TFRecord/tfDataSet.tfrecords'

    def __extract_x(self,filename):

        image = cv2.imread(filename)
        b, g, r = cv2.split(image)
        rgb_image = cv2.merge([r, g, b])
        return rgb_image

    def __extract_y(self,filename):

        image = Image.open(filename)
        return np.asarray(image,np.uint8)


    def __int64_feature(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def __bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def img2bytes(self):

        path = "./crop_x/"
        amount = len(sum([i[2] for i in os.walk(path)], []))
        i = 0

        writer = tf.python_io.TFRecordWriter(self.filename)
        for imgId in os.listdir(self.crop_x):

            xname = self.crop_x+imgId
            yname = self.crop_y+imgId

            imgx = self.__extract_x(xname)
            imgy = self.__extract_y(yname)

            irx = imgx.tostring()
            iry = imgy.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'x_raw': self.__bytes_feature(irx),
                'y_raw': self.__bytes_feature(iry)
            }))
            writer.write(example.SerializeToString())

            i = i+1
            if i%400 == 0:
                print(str(round(i/amount*100))+'%')


        writer.close()



    def mkdir(self):

        if os.path.isdir('./TFRecord/') == False:
            os.mkdir('./TFRecord/')

def main():

    a = img2tfrecords()
    a.mkdir()
    a.img2bytes()

if __name__ == '__main__':

    main()





