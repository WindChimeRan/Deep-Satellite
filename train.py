from scipy import misc
import sys
import time
import numpy as np
import tensorflow as tf
from network.model import deeplab
import read_tfrecorder



tf.app.flags.DEFINE_string("VGG_PATH", "imagenet-vgg-verydeep-19.mat",
                           "Path to vgg model weights")

tf.app.flags.DEFINE_string("TRAIN_IMAGES_PATH", "./data.tfrecords", "Path to training images")

tf.app.flags.DEFINE_integer("BATCH_SIZE", 64,
                            "Number of concurrent images to train on")

tf.app.flags.DEFINE_integer("TRAIN_NUM", 20000,
                            "Number of train epoch")

tf.app.flags.DEFINE_integer("NUM_GPUS", 4, "How many GPUs to use")

# tf.app.flags.DEFINE_string("SUMMARY_PATH", "tensorboard", "Path to store Tensorboard summaries")
# tf.app.flags.DEFINE_integer("IMAGE_SIZE", 100, "Size of output image")
# tf.app.flags.DEFINE_string("MODEL_PATH", "models","Path to read/write trained models")

FLAGS = tf.app.flags.FLAGS

def train():

    x_batch, y_batch = read_tfrecorder.input_pipeline(FLAGS.TRAIN_IMAGES_PATH, FLAGS.BATCH_SIZE)

    net = deeplab(FLAGS.VGG_PATH)

    loss = net.loss(x_batch, y_batch)

    optimiser = tf.train.AdamOptimizer(learning_rate=1e4)
    trainable = tf.trainable_variables()
    optim = optimiser.minimize(loss, var_list=trainable)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.InteractiveSession(config=config)

    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(len(trainable))


    for epoch in range(FLAGS.TRAIN_NUM):

        start_time = time.time()

        #loss_value = sess.run(loss)
        loss_value, _ = sess.run([loss,optim])
        duration = time.time() - start_time

        print('epoch {:d} \t loss = {:.8f}, ({:.3f} sec/step)'.format(epoch, loss_value, duration))

    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    train()