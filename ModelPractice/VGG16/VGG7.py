import numpy as np
import tensorflow as tf
import time
from Config import FLAGS


"""
vgg 16的代码实现
设计目标是
(1). Vgg16 创建 vgg16的一个实例
(2). Vgg16.

"""

class Vgg16:
    """
    init function
    """
    def __init__(self, trainable = FLAGS.trainable, trainModel = tf.placeholder(tf.bool), inputData = tf.placeholder("float", [None, FLAGS.height, FLAGS.width, FLAGS.channel]),inputLabel=tf.placeholder("float", [None, 2]),dropOutRatio = 0.5, vgg16_npy_path=None):
        """
        :param trainable: True: 训练 False:测试 这是Python Bool类型 主要用于构建网络
        :param trainModel: True: 训练 False:测试 这是Tensor Bool类型 主要用于运行时使用
        :param inputData:  关联输入数据
        :param inputLabel: 关联输入标注
        :param dropOutRatio: dropout率
        :param vgg16_npy_path: 模型路径,训练的时候用于保险模型的生成路径.测试的时候用于加载模型路径

        trainable 要和 trainModel 一致,否则会出问题
        """

        if trainable == True: #如果当前是训练状态

            self.__dropOut = dropOutRatio #dropout率
            self.__check = True #检查信号
            self.__modelPath = vgg16_npy_path #保存路径
            self.__data_dict = None

        else: #处于测试阶段
            self.__dropOut = 1.0 #

            #加载权重
            if vgg16_npy_path is not None:
                self.__data_dict = np.load(vgg16_npy_path, encoding="bytes").item()
                self.__check = True
                self.__modelPath = vgg16_npy_path
            else:
                print("The %s is not invalid vgg16 npy path"%(vgg16_npy_path))
                self.__data_dict = None
                self.__check = False  #加载模型失败
                self.__modelPath = None

        #和外部数据进行关联
        self.__var_dict = {}
        self.__trainable = trainable    # Python bool类型
        self.__input = inputData        # Tensor类型
        self.__label = inputLabel       # Tensor类型
        self.__trainModel = trainModel  # Tensor类型
        self.__modelPath = vgg16_npy_path
        # self.__cost = 0

        #建立网络
        self.__build()


    def check(self):
        """
        :return: 如果初始化成功则返回True, 否则返回False
        """
        return self.__check

    def getInput(self):

        return self.__input

    def getTrainable(self):

        return self.__trainable

    def __build(self):

        """
        :param

        """
        #其实可以全部改成私有变量
        self.conv1_1 = self.conv_layer(self.__input, 3, 32, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 32, 32,"conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 32, 64, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 64, 64, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 64, 32, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 32, 32, "conv3_2")
        self.pool3 = self.max_pool(self.conv3_2, 'pool3')

        # self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        # self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        # self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512,"conv4_3")
        # self.pool4 = self.max_pool(self.conv4_3, 'pool4')
        #
        # self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        # self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512,"conv5_2")
        # self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512,"conv5_3")
        # self.pool5 = self.max_pool(self.conv5_3,'pool5')

        self.fc6 = self.fc_layer(self.pool3, 31*31*32, 64,"fc6")
        # self.fc6 = self.fc_layer(self.pool5, 32768, 512, "fc6")

        assert self.fc6.get_shape().as_list()[1:] == [64]

        self.relu6 = tf.nn.relu(self.fc6)

        self.relu6 = tf.cond(self.__trainModel, lambda: tf.nn.dropout(self.relu6, self.__dropOut), lambda: self.relu6)


        self.fc8 = self.fc_layer(self.relu6, 64, 2,"fc8")
        # self.fc8 = self.fc_layer(self.relu7, 64, 2, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")


    def cost_Layer(self):

        # crossLoss = tf.nn.softmax_cross_entropy_with_logits(logits=self.prob, labels=self.__label)
        #
        # self.__cost = tf.reduce_mean(crossLoss)

        return self.__cost

    def trainOptimizer(self):

        crossLoss = tf.nn.softmax_cross_entropy_with_logits(logits=self.prob, labels=self.__label)

        crossLoss = tf.reduce_mean(crossLoss)

        train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(crossLoss)
        # train_step = tf.train.AdamOptimizer(0.001).minimize(crossLoss)

        return train_step, crossLoss,self.__label

    #生成一个average pool层
    def avg_pool(self, bottom, name):

        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    # 生成一个max pool层
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    # 生成一个 卷积层
    def conv_layer(self, bottom, in_channels, out_channels, name):

        #卷积层包含2个参数 一个是卷积核参数 一个是 bias参数,name 表示具体那个层
        with tf.variable_scope(name):

            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):

        with tf.variable_scope(name):

            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):


        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.1)

        # aa = tf.variance_scaling_initializer(scale=1.0, mode="fan_in")
        #
        # initial_value = aa([filter_size, filter_size, in_channels, out_channels])

        filters = self.get_var(initial_value, name, 0, name + "_filters")   # 例如这样的形式 conv4_1_filters

        initial_value = tf.truncated_normal([out_channels], .0, .01)

        biases = self.get_var(initial_value, name, 1, name + "_biases")   # 例如这样的形式 conv4_1_biases

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):

        # aa = tf.variance_scaling_initializer(scale=1.0, mode="fan_in")
        #
        # initial_value = aa([in_size, out_size])

        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.1)

        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)

        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):

        if self.__data_dict is not None and name in self.__data_dict:
            value = self.__data_dict[name][idx]
        else:
            value = initial_value

        if self.__trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.__var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg16-save.npy"):

        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.__var_dict.items()):

            var_out = sess.run(var)  #得到参数数值

            if name not in data_dict:
                data_dict[name] = {}

            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):

        count = 0

        for v in list(self.var_dict.values()):

            pass

            # count += reduce(lambda x, y: x * y, v.get_shape().as_list())

        return count
