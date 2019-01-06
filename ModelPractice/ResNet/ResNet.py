import tensorflow as tf
from Config import FLAGS


UPDATE_OPS_COLLECTION = "BN_OPS"



class ResNet():

    BN_EPSILON = 0.001

    def __init__(self):

        self.__data_dict = None
        self.__var_dict = {}
        self.__trainable = True


        self.globalStep = tf.placeholder(tf.int32)
        self.isTrain = tf.placeholder(tf.bool)

        self.data  = tf.placeholder(tf.float32, shape=[None, 244, 244, 3], name='data')
        self.label = tf.placeholder(tf.float32, shape=[None, 2], name="label")

        self.prob = None
        self.extraOp = None
        self.lossUsed = None


        self.build()



    def build(self):
        """
        :param n: resnet的层数 total layers = 1(第一层卷基层) + 2 n (每个残差block 由 2个卷积 组成) +1 = 2n + 2
        :return:
        """

        with tf.variable_scope('conv0') as scope:
            name = tf.get_variable_scope().name

            conv0 = self.__conv_bn_relu_layer(self.data, 3, 32, 3, True, self.isTrain, self.globalStep, name)

        with tf.variable_scope('residual_1') as scope:
            name = tf.get_variable_scope().name
            residual_1 = self.residual_block(conv0, 64, name, first_block=False)

        with tf.variable_scope('residual_2') as scope:
            name = tf.get_variable_scope().name
            residual_2 = self.residual_block(residual_1, 64, name, first_block=False)

        with tf.variable_scope('residual_3') as scope:
            name = tf.get_variable_scope().name
            residual_3 = self.residual_block(residual_2, 64, name, first_block=False)

        with tf.variable_scope('residual_4') as scope:
            name = tf.get_variable_scope().name
            residual_4 = self.residual_block(residual_3, 64, name, first_block=False)

        with tf.variable_scope('fc') as scope:

            name = tf.get_variable_scope().name

            inSizeList  = residual_4.get_shape().as_list()

            inSize = inSizeList[1]*inSizeList[2]*inSizeList[3]

            self.lossUsed = self.__full_connection_layer(residual_4, inSize, 2, True , name)


        with tf.variable_scope("softmax") as scope:

            self.prob = tf.nn.softmax(self.lossUsed)


        self.extraOp = tf.get_collection("bn")


    def calcAccuracy(self):

        correctPrediction = tf.equal(tf.argmax(self.prob, 1), tf.argmax(self.label, 1))

        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        return accuracy


    def calcLoss(self):

        crossLoss = tf.nn.softmax_cross_entropy_with_logits(logits=self.lossUsed, labels=self.label)
        crossLoss = tf.reduce_mean(crossLoss)

        return crossLoss


    def residual_block(self, bottom, output_channel, name , first_block=False):

        #输入特征 chanel 大小
        input_channel = bottom.get_shape().as_list()[-1]

        #输出是输入的2 channel大小2倍, 则我们需要进行特征图的resize
        if input_channel * 2 == output_channel:

            increase_dim = True
            stride = 2

        elif input_channel == output_channel:

            increase_dim = False
            stride = 1

        else:

            raise ValueError('输入和输出通道不符合残差网络规格')

        #建立每个残差块单元
        with tf.variable_scope('conv1_in_block', reuse=tf.AUTO_REUSE):

            now_name = name + "_" + 'conv1_in_block'

            if first_block == True:

                #网络的整体输入的第一层不需要残差 就是个普通卷积网络就可以了.
                net = self.__convolution_layer(bottom, input_channel, output_channel, 3, 1, True, now_name + "_" + "first_conv")

            else:

                net = self.__bn_relu_conv_layer(bottom, input_channel, output_channel, 3, stride, True, self.isTrain, self.globalStep, now_name + "_" + "bn_relu_conv_layer")


        with tf.variable_scope('conv2_in_block', reuse=tf.AUTO_REUSE):

            now_name = name + "_" + 'conv2_in_block'

            #在某个残差单元内部 第二个 bn_relu_conv_layer 中的卷积 stride = 1
            net = self.__bn_relu_conv_layer(net, output_channel, output_channel, 3, 1, True, self.isTrain, self.globalStep, now_name + "_" + "bn_relu_conv_layer")

        if increase_dim is True:

            #将输入特征进行池化平均计算,以便和该层的残差进行运算
            pooled_input = tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')

            #需要理解下tf.pad的含义
            padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2, input_channel // 2]])

        else:

            padded_input = bottom

        output = net + padded_input

        return output


    #制作一个卷积->bn->relu的层次
    def __conv_bn_relu_layer(self, input_layer, in_channels, out_channels, kernel_size, regularFlag, isTrain, iter, name):

        #进行卷积操作
        net = self.__convolution_layer(input_layer, in_channels, out_channels, kernel_size, 2, regularFlag, name)

        #batch_normalization 操作
        net = self.__batch_normalization_layer(net, isTrain, True, iter, name)

        #非线性函数操作
        net = tf.nn.relu(net)

        return net

    def __bn_relu_conv_layer(self, input_layer, in_channels, out_channels, kernel_size, stride , regularFlag, isTrain, iter, name):


        #batch_normalization 操作
        now_name = name + "_" + "bn"
        net = self.__batch_normalization_layer(input_layer, isTrain, True, iter, now_name)

        #非线性函数操作
        net = tf.nn.relu(net)

        #进行卷积操作
        now_name = name + "_" + "cov"
        net = self.__convolution_layer(net, in_channels, out_channels, kernel_size, stride, regularFlag, now_name)

        return net

    #一个batch normalization
    def __batch_normalization_layer(self, bottom, isTrain, convolutional, iteration , name):
        """
        :param bottom:  来底层的Tensor
        :param isTrain: 是否为训练状态Tensor
        :param convolutional: 是否是卷积层
        :param iteration: 迭代次数 place_holder
        :param name: 名称
        :return:
        """

        #使用滑动平均方式更新 mean 和 variance decay = 0.998需要再考虑下
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.5)

        #针对不同的类型,得到的 mean 和 variance 是不同的
        if convolutional:  # 针对卷积的BN 简单理解 有多少个卷积核 就有多少个mean 和 variance
            mean, variance = tf.nn.moments(bottom, axes=[0, 1, 2])
        else:
            mean, variance = tf.nn.moments(bottom, axes=[0])

        #是进行滑动平均操作
        def mean_var_with_update():
            update_moving_averages = exp_moving_avg.apply([mean, variance])
            with tf.control_dependencies([update_moving_averages]):
                return tf.identity(mean), tf.identity(variance)

        #得到mean 和 variance
        mean ,variance    = tf.cond(isTrain, mean_var_with_update , lambda: (exp_moving_avg.average(mean), exp_moving_avg.average(variance)))

        beta, gamma = self.__get_batch_normalization_var(bottom, name)

        bn_layer = tf.nn.batch_normalization(bottom, mean, variance, beta, gamma, self.BN_EPSILON)

        # exp_moving_avg.variables_to_restore()

        # tf.add_to_collection("bn", update_moving_averages)

        # 注意要对 update_moving_averages 进行run 操作
        return bn_layer

    #生成一个残差块
    def __residual_block(self, input_layer, output_channel, name, first_block=False):
        pass

    def __convolution_layer(self, bottom, in_channels, out_channels, kernel_size, stride, regularizer, name):

        #卷基核大小层默认参数为3
        filt, conv_biases = self.__get_convolution_var(kernel_size, in_channels, out_channels, regularizer, name)

        conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding='SAME')

        net = tf.nn.bias_add(conv, conv_biases)


        return net


    def __get_batch_normalization_var(self, input_layer , name):

        dimension = input_layer.get_shape().as_list()[-1]  #根据channel得到

        # initial_value = tf.zeros_initializer(dimension)
        initial_value = tf.zeros([dimension])

        #这地方记录并获得变量,bn层不需要 正则化
        beta = self.__get_var(initial_value, False , name, 0, name + "_beta")

        initial_value = tf.ones([dimension])

        # 这地方记录并获得变量
        gamma = self.__get_var(initial_value, False, name, 1, name + "_gamma")

        #返回获得的beta 和 gamma
        return beta, gamma



    def __full_connection_layer(self, bottom, in_size, out_size, regularFlag, name):

        with tf.variable_scope(name):

            weights, biases = self.__get_full_connection_layer_var(in_size, out_size, regularFlag, name)

            x = tf.reshape(bottom, [-1, in_size])

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc


    def __get_convolution_var(self, filter_size, in_channels, out_channels , regularizer, name):


        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)

        filters = self.__get_var(initial_value, regularizer, name, 0, name + "_filters")   # 例如这样的形式 conv4_1_filters

        initial_value = tf.truncated_normal([out_channels], .0, .001)

        biases = self.__get_var(initial_value, regularizer, name, 1, name + "_biases")   # 例如这样的形式 conv4_1_biases

        return filters, biases


    def __get_full_connection_layer_var(self, in_size, out_size, regularFlag, name):

        # aa = tf.variance_scaling_initializer(scale=1.0, mode="fan_in")
        #
        # initial_value = aa([in_size, out_size])

        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)

        weights = self.__get_var(initial_value, regularFlag, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)

        biases = self.__get_var(initial_value, regularFlag, name, 1, name + "_biases")

        return weights, biases


    def __get_var(self, initial_value, regularFlag, name, idx, var_name):

        if self.__data_dict is not None and name in self.__data_dict:

            value = self.__data_dict[name][idx]

        else:
            value = initial_value

        if self.__trainable:

            #加入L2正则
            if regularFlag == True:

                regularizer = tf.contrib.layers.l2_regularizer(scale=0.005)

            else:

                regularizer = None

            var = tf.get_variable(var_name, initializer=value, regularizer=regularizer)

            # var = tf.Variable(value, name=var_name)
        else:

            # var = tf.constant(value, dtype=tf.float32, name=var_name)

            #如果要进行fine tuning 则不能使用tf.constant,否则在构建优化器的时候,会出现错误,因为所有变量军事const 没有可以训练的various.
            #所以可以控制具体某个变量是否需要重新训练
            var = tf.Variable(value, name=var_name)

        self.__var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def getCurentVar(self):

        return self.__var_dict


#测试res net block
if __name__ == '__main__':

    data = tf.placeholder(dtype=tf.float32,shape=[16,224,224,32])

    model = ResNet()

    # with tf.variable_scope('residual_1') as scope:
    #     name = tf.get_variable_scope().name
    #     net = model.residual_block(data, 32, name, first_block=True)
    #
    # with tf.variable_scope('residual_2') as scope:
    #     name = tf.get_variable_scope().name
    #     net = model.residual_block(net, 64, name, first_block=False)
    #
    # print(model.getCurentVar())

