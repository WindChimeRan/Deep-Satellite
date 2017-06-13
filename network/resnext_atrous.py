from __future__ import print_function
import tensorflow as tf
import numpy as np
from TensorflowUtils import get_model_data
from functools import partial
from collections import OrderedDict
FLAGS = tf.flags.FLAGS
MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

# MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 1
IMAGE_SIZE = 100

dtype = tf.float32


BN_EPSILON = 0.001

def activation_summary(x):
    '''
    Add histogram and sparsity summaries of a tensor to tensorboard
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    Create a variable with tf.get_variable()
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''

    ## TODO: to allow different weight decay to fully connected layer and conv layer
    if is_fc_layer is True:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)

    new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables


def output_layer(input_layer, num_labels):
    '''
    Generate the output layer
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def batch_normalization_layer(input_layer, dimension):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride, relu=True):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :param relu: boolean. Relu after BN?
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    if relu is True:
        output = tf.nn.relu(bn_layer)
    else:
        output = bn_layer
    return output


def split(input_layer, stride):
    '''
    The split structure in Figure 3b of the paper. It takes an input tensor. Conv it by [1, 1,
    64] filter, and then conv the result by [3, 3, 64]. Return the
    final resulted tensor, which is in shape of [batch_size, input_height, input_width, 64]
    :param input_layer: 4D tensor in shape of [batch_size, input_height, input_width,
    input_channel]
    :param stride: int. 1 or 2. If want to shrink the image size, then stride = 2
    :return: 4D tensor in shape of [batch_size, input_height, input_width, input_channel/64]
    '''

    input_channel = input_layer.get_shape().as_list()[-1]
    num_filter = FLAGS.block_unit_depth
    # according to Figure 7, they used 64 as # filters for all cifar10 task

    with tf.variable_scope('bneck_reduce_size'):
        conv = conv_bn_relu_layer(input_layer, filter_shape=[1, 1, input_channel, num_filter],
                                  stride=stride)
    with tf.variable_scope('bneck_conv'):
        conv = conv_bn_relu_layer(conv, filter_shape=[3, 3, num_filter, num_filter], stride=1)

    return conv


def bottleneck_b(input_layer, stride):
    '''
    The bottleneck strucutre in Figure 3b. Concatenates all the splits
    :param input_layer: 4D tensor in shape of [batch_size, input_height, input_width,
    input_channel]
    :param stride: int. 1 or 2. If want to shrink the image size, then stride = 2
    :return: 4D tensor in shape of [batch_size, output_height, output_width, output_channel]
    '''
    split_list = []
    for i in range(FLAGS.cardinality):
        with tf.variable_scope('split_%i'%i):
            splits = split(input_layer=input_layer, stride=stride)
        split_list.append(splits)

    # Concatenate splits and check the dimension
    concat_bottleneck = tf.concat(values=split_list, axis=3, name='concat')

    return concat_bottleneck


def bottleneck_c1(input_layer, stride):
    '''
    The bottleneck strucutre in Figure 3c. Grouped convolutions
    :param input_layer: 4D tensor in shape of [batch_size, input_height, input_width,
    input_channel]
    :param stride: int. 1 or 2. If want to shrink the image size, then stride = 2
    :return: 4D tensor in shape of [batch_size, output_height, output_width, output_channel]
    '''
    input_channel = input_layer.get_shape().as_list()[-1]
    bottleneck_depth = FLAGS.block_unit_depth
    with tf.variable_scope('bottleneck_c_l1'):
        l1 = conv_bn_relu_layer(input_layer=input_layer,
                                filter_shape=[1, 1, input_channel, bottleneck_depth],
                                stride=stride)
    with tf.variable_scope('group_conv'):
        filter = create_variables(name='depthwise_filter', shape=[3, 3, bottleneck_depth, FLAGS.cardinality])
        l2 = tf.nn.depthwise_conv2d(input=l1,
                                    filter=filter,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME')
    return l2


def bottleneck_c(input_layer, stride):
    '''
    The bottleneck strucutre in Figure 3c. Grouped convolutions
    :param input_layer: 4D tensor in shape of [batch_size, input_height, input_width,
    input_channel]
    :param stride: int. 1 or 2. If want to shrink the image size, then stride = 2
    :return: 4D tensor in shape of [batch_size, output_height, output_width, output_channel]
    '''
    input_channel = input_layer.get_shape().as_list()[-1]
    bottleneck_depth = FLAGS.block_unit_depth * FLAGS.cardinality
    with tf.variable_scope('bottleneck_c_l1'):
        l1 = conv_bn_relu_layer(input_layer=input_layer,
                                filter_shape=[1, 1, input_channel, bottleneck_depth],
                                stride=stride)
    with tf.variable_scope('group_conv'):
        filter = create_variables(name='depthwise_filter', shape=[3, 3, bottleneck_depth, FLAGS.cardinality])
        l2 = conv_bn_relu_layer(input_layer=l1,
                                filter_shape=[3, 3, bottleneck_depth, bottleneck_depth],
                                stride=1)
    return l2


def resnext_block(input_layer, output_channel):
    '''
    The block structure in Figure 3b. Takes a 4D tensor as input layer and splits, concatenates
    the tensor and restores the depth. Finally adds the identity and ReLu.
    :param input_layer: 4D tensor in shape of [batch_size, input_height, input_width,
    input_channel]
    :param output_channel: int, the number of channels of the output
    :return: 4D tensor in shape of [batch_size, output_height, output_width, output_channel]
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    if FLAGS.bottleneck_implementation == 'b':
        concat_bottleneck = bottleneck_b(input_layer, stride)
    else:
        assert FLAGS.bottleneck_implementation == 'c'
        concat_bottleneck = bottleneck_c(input_layer, stride)

    bottleneck_depth = concat_bottleneck.get_shape().as_list()[-1]
    assert bottleneck_depth == FLAGS.block_unit_depth * FLAGS.cardinality

    # Restore the dimension. Without relu here
    restore = conv_bn_relu_layer(input_layer=concat_bottleneck,
                                 filter_shape=[1, 1, bottleneck_depth, output_channel],
                                 stride=1, relu=False)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                      input_channel // 2 ]])
    else:
        padded_input = input_layer

    # According to section 4 of the paper, relu is played after adding the identity.

    if restore.get_shape().as_list()[1:] != padded_input.get_shape().as_list()[1:]:
        padded_input = tf.pad(padded_input, [[0, 0], [1, 0], [1, 0], [0, 0]])

    output = tf.nn.relu(restore + padded_input)

    return output


def resnext(input_tensor_batch, n, reuse):
    '''
    The main function that defines the ResNeXt. total layers = 1 + 3n + 3n + 3n +1 = 9n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_resnext_blocks. The paper used n=3, 29 layers as demo
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''
    layers = OrderedDict()
    with tf.variable_scope('conv0', reuse=reuse):
        current = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 64], 1)
        activation_summary(current)
        layers['conv0']=current

    for i in range(n):
        with tf.variable_scope('conv1_%d' %i, reuse=reuse):
            current = resnext_block(current, 64)
            activation_summary(current)
            layers['conv1_%d' % i] = current
            # layers.append(conv1)

    for i in range(n):
        with tf.variable_scope('conv2_%d' %i, reuse=reuse):
            current = resnext_block(current, 128)
            activation_summary(current)
            layers['conv2_%d' % i] = current
            #layers.append(conv2)

    # shape = 25
    for i in range(1):
        with tf.variable_scope('conv3_%d' %i, reuse=reuse):
            current = resnext_block(current, 256)
            layers['conv3_%d' % i] = current
            # layers.append(conv3)resfcn
    # shape = 13, pool3

    for i in range(1):
        with tf.variable_scope('conv4_%d' % i, reuse=reuse):
            current = resnext_block(current, 512)
            layers['conv4_%d' % i] = current
            # layers.append(conv4)

    # shape = 7,pool4
    for i in range(1):
        with tf.variable_scope('conv5_%d' % i, reuse=reuse):
            current = resnext_block(current, 1024)
            layers['conv5_%d' % i] = current
            # layers.append(conv5)


    # shape = 7,logit
    for i in range(n):
        with tf.variable_scope('conv6_%d' % i, reuse=reuse):
            current = resnext_block(current, 1024)
            layers['conv6_%d' % i] = current


            # layers.append(conv6)
        # assert conv3.get_shape().as_list()[1:] == [8, 8, 256]

    # with tf.variable_scope('fc', reuse=reuse):
    #     global_pool = tf.reduce_mean(layers[-1], [1, 2])
    #
    #     # assert global_pool.get_shape().as_list()[-1:] == [256]
    #     output = output_layer(global_pool, 10)
    #     layers.append(output)


    return layers

def inference(image, keep_prob,reuse = None):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return annotation_pred:
    :return conv_t3: logits
    """
    # print("setting up vgg initialized conv layers ...")
    # model_data = get_model_data(FLAGS.model_dir, MODEL_URL)
    #
    # mean = model_data['normalization'][0][0][0]
    # mean_pixel = np.mean(mean, axis=(0, 1))
    #
    # weights = np.squeeze(model_data['layers'])
    #
    # processed_image = process_image(image, mean_pixel)

    with tf.variable_scope("inference"):

        n= 3
        image_net = resnext(image,n,reuse=None)
        conv_final_layer = image_net["conv6_2"]

        pool5 = max_pool_2x2(conv_final_layer)

        W6 = weight_variable([7, 7, 1024, 4096], name="W6")
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
        deconv_shape1 = image_net['conv5_0'].get_shape()
        W_t1 = weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net['conv5_0']))
        fuse_1 = tf.add(conv_t1, image_net['conv5_0'], name="fuse_1")

        deconv_shape2 = image_net['conv4_0'].get_shape()
        W_t2 = weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net['conv4_0']))
        fuse_2 = tf.add(conv_t2, image_net['conv4_0'], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.nn.sigmoid(conv_t3)
        annotation_pred = tf.round(annotation_pred)


    return annotation_pred, conv_t3


def iou(pred_y,y):

    pred_y = tf.squeeze(pred_y)
    inter = tf.reduce_sum(pred_y*y)
    union = tf.reduce_sum((pred_y+y)-pred_y*y)

    return inter/union


def loss(x,y,keep_probability):

    pred_annotation, logits = inference(x, keep_probability)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.squeeze(logits), labels=tf.cast(y,dtype)))

    # loss_area = tf.reduce_mean(tf.abs(tf.reduce_sum(tf.squeeze(logits))-tf.reduce_sum(tf.cast(y,dtype))))
    #
    # inter = tf.reduce_sum(tf.multiply(tf.squeeze(logits), tf.cast(y,dtype)))
    # union = tf.reduce_sum(tf.squeeze(logits)+ tf.cast(y,dtype)- tf.multiply(tf.squeeze(logits), tf.cast(y,dtype)))
    # iou_loss = 1.0 - inter/union

    # loss = loss + loss_area + iou_loss
    # loss = iou_loss

    return pred_annotation,loss

def test_graph(train_dir='logs'):
    '''
    Run this function to look at the graph structure on tensorboard. A fast way!
    :param train_dir:
    '''
    input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
    result = resnext(input_tensor, FLAGS.num_resnext_blocks, reuse=False)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

# test_graph()

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


