import tensorflow as tf
import numpy as np
import scipy.io


dtype = tf.float16

class deeplab(object):

    def __init__(self,data_path,frozen_layers):

        data = scipy.io.loadmat(data_path)
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

    )

        self.const_parameter = {}

        for i, name in enumerate(self.layers):
            kind = name[:4]

            if i<frozen_layers:
                with tf.variable_scope('frozen_layers') as scope:
                    if kind == 'conv' or kind == 'atro':
                        kernels, bias = weights[i][0][0][0][0]
                        kernels = np.transpose(kernels, (1, 0, 2, 3))
                        bias = bias.reshape(-1)
                        self.const_parameter[name] = [tf.constant(kernels,name=name,dtype=dtype), tf.constant(bias,name=name,dtype=dtype)]
            else:
                with tf.variable_scope('trainable_layers') as scope:
                    if kind == 'conv' or kind == 'atro':
                        kernels, bias = weights[i][0][0][0][0]
                        kernels = np.transpose(kernels, (1, 0, 2, 3))
                        bias = bias.reshape(-1)
                        self.const_parameter[name] = [tf.Variable(kernels,name=name,dtype=dtype), tf.Variable(bias,name=name,dtype=dtype)]


        self.atrous6_1 = self.create_variable([7, 7, 512, 4096],name = 'atrous6_1')
        self.atrous6_2 = self.create_variable([1, 1, 4096, 4096],name = 'atrous6_2')
        self.atrous6_3 = self.create_variable([1, 1, 4096, 1],name = 'atrous6_3')

        self.atrous6_1_b = self.create_bias_variable([4096],name='atrous6_1_b')
        self.atrous6_2_b = self.create_bias_variable([4096],name='atrous6_2_b')
        self.atrous6_3_b = self.create_bias_variable([1],name='atrous6_3_b')



    def inference(self, input_image):

        net = {}
        current = input_image

        for i, name in enumerate(self.layers):
            kind = name[:4]
            if kind == 'conv':
                current = self._conv_layer(current, self.const_parameter[name][0], self.const_parameter[name][1], name=name)
            elif kind == 'atro':
                current = self._atrous_layer(current, self.const_parameter[name][0], self.const_parameter[name][1], 1, name=name)
            elif kind == 'relu':
                current = tf.nn.relu(current, name=name)
            elif kind == 'pool':
                current = self._pool_layer(current, name=name)
            net[name] = current

        assert len(net) == len(self.layers)

        current = tf.nn.avg_pool(current,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME')

        # print(current.get_shape())

        # current = self._atrous_layer(current, self.atrous6_1, self.atrous6_1_b, 12)
        current = self._atrous_layer(current, self.atrous6_1, self.atrous6_1_b, 1)
        current = tf.nn.relu(current, name=None)
        net['atrous_6_1'] = current

        current = self._atrous_layer(current, self.atrous6_2, self.atrous6_2_b, 1)
        current = tf.nn.relu(current, name=None)
        net['atrous_6_2'] = current

        current = self._atrous_layer(current, self.atrous6_3,self.atrous6_3_b, 1)
        current = tf.nn.relu(current, name=None)
        net['atrous_6_3'] = current


        deconv_shape1 = net["pool4"].get_shape()

        W_t1 = self.create_variable([4, 4, deconv_shape1[3].value, 1], name="W_t1")
        b_t1 = self.create_bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = self.conv2d_transpose_strided(current, W_t1, b_t1, stride=1,output_shape=tf.shape(net["pool4"]))

        conv_t1 = tf.nn.relu(conv_t1, name=None)
        # print(net["pool4"].get_shape()) 7 7

        fuse_1 = tf.add(conv_t1, net["pool4"], name="fuse_1")
        deconv_shape2 = net["pool3"].get_shape()
        W_t2 = self.create_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")

        b_t2 = self.create_bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = self.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(net["pool3"]))

        conv_t2 = tf.nn.relu(conv_t2, name=None)
        #print(net["pool3"].get_shape())

        fuse_2 = tf.add(conv_t2, net["pool3"], name="fuse_2")

        shape = tf.shape(input_image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], 1])
        W_t3 = self.create_variable([16, 16, 1, deconv_shape2[3].value], name="W_t3")
        b_t3 = self.create_bias_variable([1], name="b_t3")
        conv_t3 = self.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        conv_t3 =  tf.nn.relu(conv_t3, name=None)

        #print(conv_t3.get_shape())


        return  conv_t3


    def create_bias_variable(self,shape, name = None):
        """Create a bias variable of the given name and shape,
           and initialise it to zero.
        """
        initialiser = tf.constant_initializer(value=0.0, dtype=dtype)
        variable = tf.Variable(initialiser(shape=shape), name=name)
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


    def create_variable(self, shape,name = None):

        """Create a convolution filter variable of the given name and shape,
           and initialise it using Xavier initialisation
           (http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf).
        """
        initialiser = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        variable = tf.Variable(initialiser(shape=shape), name=name)
        return variable

    def prepare_label(self,y,new_size):

        y = tf.image.resize_nearest_neighbor(y,new_size)
        y = tf.squeeze(y,squeeze_dims=[3])
        return y
        # return tf.one_hot(y, depth=1)

    def conv2d_transpose_strided(self,x, W, b, output_shape=None, stride=2):

        if output_shape is None:
            output_shape = x.get_shape().as_list()
            output_shape[1] *= 2
            output_shape[2] *= 2
            output_shape[3] = W.get_shape().as_list()[2]

        conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
        return tf.nn.bias_add(conv, b)

    def loss(self,x, y_label):

        y_label = tf.cast(y_label,tf.float16)

        pre_y = self.inference(tf.cast(x, dtype))

        y_label = tf.reshape(y_label, [-1, 1])
        pre_y = tf.reshape(pre_y, [-1, 1])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pre_y,labels=y_label))
        # loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=pre_y, targets=y_label,pos_weight=1))


        tf.add_to_collection('losses',loss)



        return tf.add_n(tf.get_collection('losses'), name = 'total_loss')

