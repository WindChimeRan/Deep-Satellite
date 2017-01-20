import os
import tensorflow as tf
import numpy as np
from PIL import Image

def _disp_tfrecord(tfrecord_file = './TFRecord/tfDataSet.tfrecords'):

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

    imgx = tf.reshape(imgx ,[100 ,100 ,3])
    imgy = tf.reshape(imgy, [100, 100])
    imgx = tf.cast(imgx, tf.float32)
    imgy = tf.cast(imgy, tf.float32)


    return imgx, imgy

def read_tfrecord(tfrecord_file = './TFRecord/tfDataSet.tfrecords'):

    imgx ,imgy = _disp_tfrecord(tfrecord_file)
    return  imgx * (1. / 255) - 0.5 ,imgy * (1. / 255)

def disp_one():

    sess = tf.InteractiveSession()
    imgx, imgy = _disp_tfrecord()
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

def read_batch(batch_size = 100):

    sess = tf.InteractiveSession()
    imgx, imgy = read_tfrecord()
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

    return x ,y

if __name__ == '__main__':

    x,y = read_batch(100)
    print(x.shape,y.shape)
    disp_one()