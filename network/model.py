import  tensorflow as tf
import numpy as np
import scipy.io
from scipy import misc


class deeplab(object):

    def __init__(self,data_path):

        data = scipy.io.loadmat(data_path)
        # mean = data['normalization'][0][0][0]
        # mean_pixel = np.mean(mean, axis=(0, 1))
        weights = data['layers'][0]

        self.layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'atrous5_1', 'relu5_1', 'atrous5_2', 'relu5_2', 'atrous5_3',
        'relu5_3', 'atrous5_4', 'relu5_4'

        # 'natrous_6_1','relu6_2''nconv6_3','relu6_4', 'nconv6_5'
    )

        self.const_parameter = {}


        for i, name in enumerate(self.layers):
            kind = name[:4]
            if kind == 'conv' or kind == 'atro':
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                bias = bias.reshape(-1)
                self.const_parameter[name] = [tf.constant(kernels), tf.constant(bias)]


        self.atrous6_1 = self.create_variable([3, 3, 512, 256])
        self.atrous6_2 = self.create_variable([3, 3, 256, 128])
        self.atrous6_3 = self.create_variable([3, 3, 128, 1])

        self.atrous6_1_b = self.create_bias_variable([256])
        self.atrous6_2_b = self.create_bias_variable([128])
        self.atrous6_3_b = self.create_bias_variable([1])

        del data
        del weights


    def net(self, input_image):

        net = {}
        current = input_image

        for i, name in enumerate(self.layers):
            kind = name[:4]
            if kind == 'conv':
                current = self._conv_layer(current, self.const_parameter[name][0], self.const_parameter[name][1], name=name)
            elif kind == 'atro':
                current = self._atrous_layer(current, self.const_parameter[name][0], self.const_parameter[name][1], 2, name=name)
            elif kind == 'relu':
                current = tf.nn.relu(current, name=name)
            elif kind == 'pool':
                current = self._pool_layer(current, name=name)
            net[name] = current


        assert len(net) == len(self.layers)


        current = tf.nn.avg_pool(current,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME')

        current = self._atrous_layer(current, self.atrous6_1, self.atrous6_1_b, 12)
        current = tf.nn.relu(current, name=None)
        net['atrous_6_1'] = current

        current = self._atrous_layer(current, self.atrous6_2, self.atrous6_2_b, 1)
        current = tf.nn.relu(current, name=None)
        net['atrous_6_2'] = current

        current = self._atrous_layer(current, self.atrous6_3,self.atrous6_3_b, 1)
        current = tf.nn.relu(current, name=None)
        net['atrous_6_3'] = current

        return current

    def create_bias_variable(self,shape):
        """Create a bias variable of the given name and shape,
           and initialise it to zero.
        """
        initialiser = tf.constant_initializer(value=0.0, dtype=tf.float32)
        variable = tf.Variable(initialiser(shape=shape), name=None)
        return variable

    def _conv_layer(self,input, weights, bias, name=None):

        conv = tf.nn.conv2d(input, weights, strides=(1, 1, 1, 1), padding='SAME', name=name)
        return tf.nn.bias_add(conv, bias)

    def _pool_layer(self,input, name=None):

        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME', name=name)

    def preprocess(self,image, mean_pixel):

        return image - mean_pixel

    def unprocess(self,image, mean_pixel):

        return image + mean_pixel


    def _atrous_layer(self,input, filter, bias, rate,padding='SAME',name=None):

        atrous = tf.nn.atrous_conv2d(input,filter,rate,padding)
        return  tf.nn.bias_add(atrous, bias)


    def create_variable(self, shape):

        """Create a convolution filter variable of the given name and shape,
           and initialise it using Xavier initialisation
           (http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf).
        """
        initialiser = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
        variable = tf.Variable(initialiser(shape=shape), name=None)
        return variable

    def prepare_label(self,y,new_size):

        y = tf.image.resize_nearest_neighbor(y,new_size)
        y = tf.squeeze(y,squeeze_dims=[3])
        return y
        # return tf.one_hot(y, depth=1)

    def loss(self,x, y_label):

        # print(x.shape)
        pre_y = self.net(tf.cast(x, tf.float32))
        y_label = tf.reshape(y_label, y_label.shape.as_list()+[1])
        newsize = tf.stack(pre_y.get_shape()[1:3])
        y_label = self.prepare_label(y_label,newsize)

        y_label = tf.reshape(y_label, [-1, 1])
        pre_y = tf.cast(tf.reshape(pre_y, [-1, 1]),tf.float32)

        # loss = tf.nn.softmax_cross_entropy_with_logits(logits=pre_y,labels=y_label)
        loss = tf.nn.l2_loss(pre_y-y_label)

        # return tf.reduce_sum(loss)
        return tf.reduce_mean(loss)