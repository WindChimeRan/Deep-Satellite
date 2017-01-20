import os
import tensorflow as tf
import numpy as np
from PIL import Image
from scipy import misc
import cv2


class img2tfrecords:

    crop_x = '../crop_x/'
    crop_y = '../crop_y/'
    filename = '../TFRecord/tfDataSet.tfrecords'

    def extract_x(self,filename):

        image = cv2.imread(filename)
        b, g, r = cv2.split(image)
        rgb_image = cv2.merge([r, g, b])
        return rgb_image

    def extract_y(self,filename):

        image = Image.open(filename)
        return np.asarray(image,np.uint8)


    def _int64_feature(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def img2bytes(self):

        # path = "../crop_x/"
        # amount = len(sum([i[2] for i in os.walk(path)], []))

        writer = tf.python_io.TFRecordWriter(self.filename)
        for imgId in os.listdir(self.crop_x):

            xname = self.crop_x+imgId
            yname = self.crop_y+imgId

            imgx = self.extract_x(xname)
            imgy = self.extract_y(yname)

            irx = imgx.tostring()
            iry = imgy.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'x_raw': self._bytes_feature(irx),
                'y_raw': self._bytes_feature(iry)
            }))
            writer.write(example.SerializeToString())



        writer.close()

    def _disp_tfrecord(self,tfrecord_file = '../TFRecord/tfDataSet.tfrecords'):

        filename_queue = tf.train.string_input_producer([tfrecord_file])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'x_raw': tf.FixedLenFeature([], tf.string),
                'y_raw': tf.FixedLenFeature([], tf.string)
            }
        )
        imgx = tf.decode_raw(features['x_raw'], tf.uint8)
        imgy = tf.decode_raw(features['y_raw'], tf.uint8)

        imgx = tf.reshape(imgx,[100,100,3])
        imgy = tf.reshape(imgy, [100, 100])
        imgx = tf.cast(imgx, tf.float32)
        imgy = tf.cast(imgy, tf.float32)


        return imgx, imgy

    def _read_tfrecord(self,tfrecord_file = '../TFRecord/tfDataSet.tfrecords'):

        imgx,imgy = self._disp_tfrecord(tfrecord_file)
        return  imgx * (1. / 255) - 0.5,imgy * (1. / 255)

    def disp_one(self):

        sess = tf.InteractiveSession()
        imgx, imgy = self._disp_tfrecord()
        x_batch, y_batch = tf.train.shuffle_batch([imgx, imgy],
                                                  num_threads=2,
                                                  batch_size=1, capacity=1000 + 3 * 32,
                                                  min_after_dequeue=1000)

        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(10):
            x, y = sess.run([x_batch, y_batch])

            x = np.asarray(x, np.uint8)
            y = np.asarray(y, np.uint8)
            y = np.reshape(y, [100, 100])
            x = np.reshape(x, [100, 100, 3])
            imx = Image.fromarray(x)
            imy = Image.fromarray(y)
            imx.show()
            imy.show()
            break

        coord.request_stop()
        coord.join(threads)
        sess.close()

    def read_batch(self,batch_size = 10):

        sess = tf.InteractiveSession()
        imgx, imgy = self._read_tfrecord()
        x_batch, y_batch = tf.train.shuffle_batch([imgx, imgy],
                                                  num_threads=2,
                                                  batch_size=batch_size, capacity=1000 + 3 * 32,
                                                  min_after_dequeue=1000)

        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        x, y = sess.run([x_batch, y_batch])

        coord.request_stop()
        coord.join(threads)
        sess.close()

        return x,y

if __name__ == '__main__':

    a = img2tfrecords()

    a.img2bytes()

    #a.disp_one()

    x,y = a.read_batch(1000)
    print(x.shape,y.shape)




