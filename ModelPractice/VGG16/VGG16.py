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
    def __init__(self, trainable = FLAGS.trainable, trainModel = tf.placeholder(tf.bool), dropOutRatio = 0.5, vgg16_npy_path=None):
        """
        :param trainable: True: 训练    False:测试 这是Python Bool类型 主要用于构建网络
        :param trainModel: True: 训练   False:验证 这是Tensor Bool类型 主要用于运行时使用,可以区分在训练阶段时,是否处于验证数据集阶段
        :param inputData:  关联输入数据
        :param inputLabel: 关联输入标注
        :param dropOutRatio: dropout率
        :param vgg16_npy_path: 模型路径,训练的时候用于保险模型的生成路径.测试的时候用于加载模型路径
        :param validData:   验证数据集
        :param validLabel:  验证标注

        trainable 要和 trainModel 一致,否则会出问题
        """

        if trainable == True: #如果当前是训练状态

            self.__dropOut = dropOutRatio #dropout率
            self.__check = True #检查信号
            self.__modelPath = vgg16_npy_path #保存路径
            self.__data_dict = None

        else: #处于测试阶段,如果提供了权重路径 vgg16_npy_path 则加载这个路径,否则出错
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

        self.__trainModel = trainModel  # Tensor类型

        self.__modelPath = vgg16_npy_path

        #用于定义动态学习率
        self.global_step = tf.Variable(0, trainable=False)
        self.__num_epochs = 3
        self.learnRatio = 0

        # self.__cost = 0

        #建立网络
        # self.__build()


    def check(self):
        """
        :return: 如果初始化成功则返回True, 否则返回False
        """
        return self.__check

    def getInput(self):

        return self.__input

    def getTrainable(self):

        return self.__trainable

    def build(self,inputData):

        """
        :param

        """
        #其实可以全部改成私有变量
        self.conv1_1 = self.conv_layer(inputData, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64,"conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256,"conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512,"conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512,"conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512,"conv5_3")
        self.pool5 = self.max_pool(self.conv5_3,'pool5')

        #这地方有时候不对.有的版本是2X开头数字
        self.fc6 = self.fc_layer(self.pool5, 32768, 4096, "fc6")

        assert self.fc6.get_shape().as_list()[1:] == [4096]

        self.relu6 = tf.nn.relu(self.fc6)


        #如果是正在训练阶段,则在构建网络的时候,需要考虑验证阶段dropout的问题
        if self.__trainable == True:
            self.relu6 = tf.cond(self.__trainModel, lambda: tf.nn.dropout(self.relu6, self.__dropOut), lambda: self.relu6)



        self.fc7 = self.fc_layer(self.relu6, 4096, 2048,"fc7")

        self.relu7 = tf.nn.relu(self.fc7)

        if self.__trainable == True:
            self.relu7 = tf.cond(self.__trainModel, lambda: tf.nn.dropout(self.relu7, self.__dropOut), lambda: self.relu7)

        self.fc8 = self.fc_layer(self.relu7, 2048, 2,"fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")


    #计算准确率
    def calcAccuracy(self,inputLabel):

        correctPrediction = tf.equal(tf.argmax(self.prob, 1), tf.argmax(inputLabel, 1))

        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        return accuracy

    def dynamicLearning(self):

        #动态学习率调整应该以epoch为单位,即 每个epoch(所有样本为循环遍历一次)进行调整.当然这个是不是觉得,但是如果每次训练(batch为单位),就会发现lr变化
        #lr = initlearnRatio * decay_rate ** (global_step/__num_epochs) global_step/__num_epochs 是个小数 但是,
        # staircase=True , 则 global_step/__num_epochs 转换为整数 这样可以做到精确控制没隔多少个batch学习率 则会进行一定指数级的衰减

        self.learnRatio = tf.train.exponential_decay(1.0, self.global_step, decay_steps = self.__num_epochs, decay_rate=0.1, staircase=True)

        return self.learnRatio

    def trainOptimizer(self, inputLabel):

        """
        :param inputLabel:  训练时候用到的标注数据
        :return:
        """
        crossLoss = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc8, labels=inputLabel)

        crossLoss = tf.reduce_mean(crossLoss)

        #根据需要选择自己的学习率
        lr = self.dynamicLearning()

        train_step = tf.train.GradientDescentOptimizer(lr).minimize(crossLoss, global_step=self.global_step)

        return train_step, crossLoss,inputLabel

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


        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)

        filters = self.get_var(initial_value, name, 0, name + "_filters")   # 例如这样的形式 conv4_1_filters

        initial_value = tf.truncated_normal([out_channels], .0, .001)

        biases = self.get_var(initial_value, name, 1, name + "_biases")   # 例如这样的形式 conv4_1_biases

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):

        # aa = tf.variance_scaling_initializer(scale=1.0, mode="fan_in")
        #
        # initial_value = aa([in_size, out_size])

        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)

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

            # var = tf.constant(value, dtype=tf.float32, name=var_name)

            #如果要进行fine tuning 则不能使用tf.constant,否则在构建优化器的时候,会出现错误,因为所有变量军事const 没有可以训练的various.
            #所以可以控制具体某个变量是否需要重新训练
            var = tf.Variable(value, name=var_name)

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
