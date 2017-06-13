from __future__ import print_function
import tensorflow as tf
import numpy as np
from TensorflowUtils import get_model_data
from functools import partial

FLAGS = tf.flags.FLAGS
MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

# MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 1
IMAGE_SIZE = 100

dtype = tf.float32

def bottleneck_unit(x, out_chan1, out_chan2, down_stride=False, up_stride=False, name=None):
    """
    Modified implementation from github ry?!
    """

    def conv_transpose(tensor, out_channel, shape, strides, name=None):
        out_shape = tensor.get_shape().as_list()
        in_channel = out_shape[-1]
        kernel = weight_variable([shape, shape, out_channel, in_channel], name=name)
        shape[-1] = out_channel
        return tf.nn.conv2d_transpose(x, kernel, output_shape=out_shape, strides=[1, strides, strides, 1],
                                      padding='SAME', name='conv_transpose')

    def conv(tensor, out_chans, shape, strides, name=None):
        in_channel = tensor.get_shape().as_list()[-1]
        kernel = weight_variable([shape, shape, in_channel, out_chans], name=name)
        return tf.nn.conv2d(x, kernel, strides=[1, strides, strides, 1], padding='SAME', name='conv')

    def bn(tensor, name=None):

        return tf.contrib.layers.batch_norm(tensor,center = True,scale = True, is_training = True)
        # return tf.nn.lrn(tensor, depth_radius=5, bias=2, alpha=1e-4, beta=0.75, name=name)

    in_chans = x.get_shape().as_list()[3]

    if down_stride or up_stride:
        first_stride = 2
    else:
        first_stride = 1

    with tf.variable_scope('res%s' % name):
        if in_chans == out_chan2:
            b1 = x
        else:
            with tf.variable_scope('branch1'):
                if up_stride:
                    b1 = conv_transpose(x, out_chans=out_chan2, shape=1, strides=first_stride,
                                        name='res%s_branch1' % name)
                else:
                    b1 = conv(x, out_chans=out_chan2, shape=1, strides=first_stride, name='res%s_branch1' % name)
                b1 = bn(b1, 'bn%s_branch1' % name)

        with tf.variable_scope('branch2a'):
            if up_stride:
                b2 = conv_transpose(x, out_chans=out_chan1, shape=1, strides=first_stride, name='res%s_branch2a' % name)
            else:
                b2 = conv(x, out_chans=out_chan1, shape=1, strides=first_stride, name='res%s_branch2a' % name)
            b2 = bn(b2, 'bn%s_branch2a' % name)
            b2 = tf.nn.relu(b2, name='relu')

        with tf.variable_scope('branch2b'):
            b2 = conv(b2, out_chans=out_chan1, shape=3, strides=1, name='res%s_branch2b' % name)
            b2 = bn(b2, 'bn%s_branch2b' % name)
            b2 = tf.nn.relu(b2, name='relu')

        with tf.variable_scope('branch2c'):
            b2 = conv(b2, out_chans=out_chan2, shape=1, strides=1, name='res%s_branch2c' % name)
            b2 = bn(b2, 'bn%s_branch2c' % name)

        x = b1 + b2
        return tf.nn.relu(x, name='relu')
def iou(pred_y,y):

    pred_y = tf.squeeze(pred_y)
    inter = tf.reduce_sum(pred_y*y)
    union = tf.reduce_sum((pred_y+y)-pred_y*y)

    return inter/union

def class_balanced_sigmoid_cross_entropy(logits, label, name='cross_entropy_loss'):
    """
    The class-balanced cross entropy loss,
    as in `Holistically-Nested Edge Detection
    <http://arxiv.org/abs/1504.06375>`_.
    This is more numerically stable than class_balanced_cross_entropy

    :param logits: size: the logits.
    :param label: size: the ground truth in {0,1}, of the same shape as logits.
    :returns: a scalar. class-balanced cross entropy loss
    """
    y = tf.cast(label, tf.float32)

    count_neg = tf.reduce_sum(1. - y) # the number of 0 in y
    count_pos = tf.reduce_sum(y) # the number of 1 in y (less than count_neg)
    beta = count_neg / (count_neg + count_pos)
    pos_weight = beta / (1 - beta)
    cost = tf.nn.weighted_cross_entropy_with_logits(logits, y, pos_weight)
    cost = tf.reduce_mean(cost * (1 - beta), name=name)

    return cost

def loss(x,y,keep_probability):

    pred_annotation, logits = inference(x, keep_probability)

    # y = tf.cast(y, tf.float32)
    #
    # count_neg = tf.reduce_sum(1. - y) # the number of 0 in y
    # count_pos = tf.reduce_sum(y) # the number of 1 in y (less than count_neg)
    # beta = count_neg / (count_neg + count_pos)
    #
    # pos_weight = beta / (1 - beta)
    # cost = tf.nn.weighted_cross_entropy_with_logits(tf.squeeze(logits), y, pos_weight)
    # # cost = tf.reduce_mean(cost * (1 - beta), name='cross_entropy_loss')
    # loss = tf.reduce_mean(cost)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.squeeze(logits), labels=tf.cast(y,dtype)))



    return pred_annotation,loss

def vgg_net(weights, image):

    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )


    net = {}
    current = image
    for i, name in enumerate(layers):

        kind = name[:4]

        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]

            if i<FLAGS.frozen_rate:
                kernels = get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w",trainable=False)
                bias = get_variable(bias.reshape(-1), name=name + "_b",trainable=False)
            else:
                kernels = get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w",trainable=True)
                bias = get_variable(bias.reshape(-1), name=name + "_b",trainable=True)

            if name[:5] == 'conv5':
                current = atrous2d_basic(current,kernels,bias,rate=2)
            else:
                current = conv2d_basic(current, kernels, bias)

        elif kind == 'relu':
            # current = tf.contrib.layers.batch_norm(current)
            # current = tf.nn.elu(current, name=name)
            current = tf.nn.relu(current, name=name)

        elif kind == 'pool':
            current = avg_pool_2x2(current)
        net[name] = current


    # print("pool3 ", net['pool3'].get_shape())
    # print("pool4 ", net['pool4'].get_shape())
    # print("conv5_4 ", net['conv5_4'].get_shape())
    '''
    pool3  (?, 13, 13, 256)
    pool4  (?, 7, 7, 512)
    conv5_4  (?, ?, ?, 512)
    '''


    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return annotation_pred:
    :return conv_t3: logits
    """
    print("setting up vgg initialized conv layers ...")
    model_data = get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_4"]

        pool5 = max_pool_2x2(conv_final_layer)

        pool5 = bottleneck_unit(pool5, 512, 512)



        W6 = weight_variable([7, 7, 512, 4096], name="W6")
        b6 = bias_variable([4096], name="b6")
        conv6 = atrous2d_basic(pool5, W6, b6,rate=12)
        conv6 = tf.contrib.layers.batch_norm(conv6)
        relu6 = tf.nn.elu(conv6, name="relu6")

        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = bias_variable([4096], name="b7")
        conv7 = conv2d_basic(relu_dropout6, W7, b7)

        conv7 = tf.contrib.layers.batch_norm(conv7)
        relu7 = tf.nn.elu(conv7, name="relu7")

        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = conv2d_basic(relu_dropout7, W8, b8)

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.nn.sigmoid(conv_t3)
        annotation_pred = tf.round(annotation_pred)


    return annotation_pred, conv_t3


def get_variable(weights, name, dtype = tf.float32,trainable=True):

    init = tf.constant_initializer(weights, dtype=dtype)
    var = tf.get_variable(name=name, initializer=init,  shape=weights.shape,trainable=trainable)
    return var


def weight_variable(shape, name=None):

    initial = tf.contrib.layers.xavier_initializer_conv2d()
    if name is None:
        return tf.Variable(initial(shape=shape))
    else:
        return tf.get_variable(name, initializer=initial(shape = shape))

def bias_variable(shape, name=None):

    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def atrous2d_basic(x, W, bias,rate = 1):
    atrous = tf.nn.atrous_conv2d(x,W,rate=rate, padding="SAME")
    return tf.nn.bias_add(atrous, bias)


def conv2d_transpose_strided(x, W, b, output_shape=None, stride = 2):

    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]

    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)

def process_image(image, mean_pixel):
    return image - mean_pixel

max_pool_2x2 = partial(tf.nn.max_pool,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
avg_pool_2x2 = partial(tf.nn.avg_pool,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


