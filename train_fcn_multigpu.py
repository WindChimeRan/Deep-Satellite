from scipy import misc
import sys
import time
import numpy as np
import tensorflow as tf
from network.fcn import deeplab
import read_tfrecorder



tf.app.flags.DEFINE_string("VGG_PATH", "imagenet-vgg-verydeep-19.mat",
                           "Path to vgg model weights")

# tf.app.flags.DEFINE_string("TRAIN_IMAGES_PATH", "./data.tfrecords", "Path to training images")
tf.app.flags.DEFINE_string("TRAIN_IMAGES_PATH", "/home/vision_zhr/Desktop/data.tfrecords", "Path to training images")

tf.app.flags.DEFINE_integer("BATCH_SIZE", 32,
                            "Number of concurrent images to train on")

tf.app.flags.DEFINE_integer("FROZEN_LAYERS", 20,
                            "Number of concurrent images to train on")

tf.app.flags.DEFINE_integer("TRAIN_NUM", 200000,
                            "Number of train steps")

tf.app.flags.DEFINE_integer("EPOCH_MAX", 100,
                            "Max number of train epoch")

tf.app.flags.DEFINE_integer("LEARNING_RATE", 1e-4,
                            "learning rate")

tf.app.flags.DEFINE_integer("NUM_GPUS", 2, "How many GPUs to use")

# tf.app.flags.DEFINE_string("SUMMARY_PATH", "tensorboard", "Path to store Tensorboard summaries")
# tf.app.flags.DEFINE_integer("IMAGE_SIZE", 100, "Size of output image")
# tf.app.flags.DEFINE_string("MODEL_PATH", "models","Path to read/write trained models")

FLAGS = tf.app.flags.FLAGS


def tower_loss(scope):

    x_batch, y_batch = read_tfrecorder.input_pipeline(FLAGS.TRAIN_IMAGES_PATH, FLAGS.BATCH_SIZE)

    net = deeplab(FLAGS.VGG_PATH,FLAGS.FROZEN_LAYERS)

    _ = net.loss(x_batch, y_batch)

    losses = tf.get_collection('losses',scope)

    total_loss = tf.add_n(losses, name = 'total_losses')

    return total_loss

def average_gradients(tower_grads):

    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        grads = []
        for g, _ in grad_and_vars:

            expanded_g = tf.expand_dims(g,0)
            grads.append(expanded_g)

        grad = tf.concat(values=grads,axis=0)
        grad = tf.reduce_mean(grad,axis=0)

        v = grad_and_vars[0][1]
        grad_and_vars = (grad, v)
        average_grads.append(grad_and_vars)

    return average_grads


def train():

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        optimiser = tf.train.AdamOptimizer(learning_rate=FLAGS.LEARNING_RATE)
        trainable = tf.trainable_variables()

        tower_grads = []


        with tf.variable_scope(tf.get_variable_scope()):

            for i in range(2,4):
            #for i in range(FLAGS.NUM_GPUS):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('GPU_%d' % i) as scope:

                        loss = tower_loss(scope)
                        tf.get_variable_scope().reuse_variables()
                        grads = optimiser.compute_gradients(loss,tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope))
                        tower_grads.append(grads)


        grads = average_gradients(tower_grads)
        apply_gradient_op = optimiser.apply_gradients(grads)
        train_op = tf.group(apply_gradient_op)

        config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=False)
        config.gpu_options.allow_growth = True


        sess = tf.InteractiveSession(config=config)

        init = tf.global_variables_initializer()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # print(len(trainable))

        for step in range(FLAGS.TRAIN_NUM):

            start_time = time.time()

            _, loss_value = sess.run([train_op,loss])
            duration = time.time() - start_time

            if step%10 == 0:
                print('step {:d} \t loss = {:.8f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

        coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    train()