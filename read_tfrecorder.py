import os
import tensorflow as tf
import numpy as np
from PIL import Image


#IMG_MEAN = np.array((42.9496727,  42.9496727,  42.9496727), dtype=np.float32)

IMG_MEAN = np.array((116.6875 ,115.8125 ,109.3125), dtype=np.float16)


def _disp_tfrecord(tfrecord_file = './data.tfrecords'):

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


def read_tfrecord(tfrecord_file = './data.tfrecords'):

    imgx ,imgy = _disp_tfrecord(tfrecord_file)
    return (imgx-IMG_MEAN), imgy * (1. / 255)


def disp_one():

    sess = tf.InteractiveSession()
    imgx, imgy = _disp_tfrecord()
    x_batch, y_batch = tf.train.shuffle_batch([imgx, imgy],
                                              num_threads=2,
                                              batch_size=10, capacity=1000 + 3 * 32,
                                              min_after_dequeue=1000)

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(10):
        x, y = sess.run([x_batch, y_batch])

        x = np.asarray(x, np.uint8)
        y = np.asarray(y, np.uint8)

        xx = [x[id,:,:,:] for id in range(10)]
        yy = [y[id, :, :] for id in range(10)]
        for i in range(10):

            imx = Image.fromarray(xx[i])
            imy = Image.fromarray(yy[i])
            imx.show()
            imy.show()

        break

    coord.request_stop()
    coord.join(threads)
    sess.close()


def read_batch_for_test(batch_size = 32):

    sess = tf.InteractiveSession()
    imgx, imgy = input_pipeline("./train.tfrecords",2)

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(10):
        x, y = sess.run([imgx, imgy])
        print y[0,0,1:10]

    coord.request_stop()
    coord.join(threads)
    sess.close()

    return x,y


def input_pipeline(data_path, batch_size):

    filename_queue = tf.train.string_input_producer([data_path])
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

    # imgx, imgy = imgx-IMG_MEAN, imgy * (1. / 255)
    imgx, imgy = imgx, imgy * (1. / 255)
    x_batch, y_batch = tf.train.shuffle_batch([imgx, imgy],
                                              num_threads=24,
                                              batch_size=batch_size, capacity=50000+6*32,
                                              min_after_dequeue=10000)
    return x_batch, y_batch


def img_mean():

    x, _ = read_batch_for_test(10000)
    return np.mean(x, axis=(0,1,2))

if __name__ == '__main__':
    #
    x,y = read_batch_for_test(100)
    print(y)
    # # print(np.max(x,axis=(0,1,2)))
    # # disp_one()
    # print x
    #print(img_mean())

    # print(trainable)
    # print(len(trainable))

