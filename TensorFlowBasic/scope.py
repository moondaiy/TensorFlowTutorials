import tensorflow as tf
import numpy as np


# demo1
# def my_image_file(input):
#     conv1_weights = tf.Variable(tf.random_normal([3,4]),name="conv1_weights")
#     return conv1_weights
#
# input1=tf.get_variable(name="var1", initializer=np.ones (shape=[2,3],dtype=np.float32))
# input2=tf.get_variable(name="var2", initializer=np.zeros(shape=[2,3],dtype=np.float32))
#
# #对于这个地方,如果我们想  input1 和 input2都进入到 my_image_file函数, 并使用同一组参数进行处理,那么当我们这样
# #调用函数的时候 是做不到这样的目的 因为调用了2次会产生2组conv1_weights 而不是一个.,如果我们想实现共享变量问题,则看Demo2
# ret1=my_image_file(input1)
# ret2=my_image_file(input2)
#
# init =tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print (sess.run(ret1))
#     print (sess.run(ret2))



# demo2
#利用字典(或者全局变量方式) 先创建一个全局的变量, 这样是可以实现权重共享的.
# variables_dict = {
#     "conv1_weights":tf.Variable(tf.random_normal([2,3]),name="conv1_weights"),
#     "conv1_biases":tf.Variable(tf.zeros([5]), name="conv1_biases")
# }
#
# def my_image_file(input):
#     conv1_weights = variables_dict['conv1_weights']
#     return conv1_weights
#
# input1=tf.get_variable(name="var1", initializer=np.ones (shape=[2,3],dtype=np.float32))
# input2=tf.get_variable(name="var2", initializer=np.zeros(shape=[2,3],dtype=np.float32))
#
# ret1=my_image_file(input1)
# ret2=my_image_file(input2)
#
# init =tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print (sess.run(ret1))
#     print (sess.run(ret2))


# demo3
# 利用scope方式进行
# def my_image_file(input_images):
#     conv1_weights = tf.get_variable("weights", [3,4],initializer=tf.random_normal_initializer())
#     return conv1_weights
#
#
# input1=tf.get_variable(name="var1", initializer=np.ones (shape=[2,3],dtype=np.float32))
# input2=tf.get_variable(name="var2", initializer=np.zeros(shape=[2,3],dtype=np.float32))
#
# #variable scope = image_filters
# with tf.variable_scope("image_filters") as scope:
#
#     ret1 = my_image_file(input1)
#
#     #这是关键代码
#     scope.reuse_variables()
#
#     ret2 = my_image_file(input2)
#
#
#
# init =tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print (ret1.name,sess.run(ret1))
#     print (ret2.name,sess.run(ret2))




#demo4
#这个个错误的例子
def my_image_file(input_images):
    with tf.variable_scope("my_image_file") as scope:
        conv1_weights = tf.get_variable("weights2", [3,4],initializer=tf.random_normal_initializer())
        return conv1_weights


input1=tf.get_variable(name="var1", initializer=np.ones (shape=[2,3],dtype=np.float32))
input2=tf.get_variable(name="var2", initializer=np.zeros(shape=[2,3],dtype=np.float32))


with tf.variable_scope("image_filters",reuse=tf.AUTO_REUSE) as scope:
    # 创建在这里面创建一个
    # 如果是tf.Variable 创建的变量 如果在使用 tf.get_variable 想得到同一名字的变量时,会产生错误.因此需要用 tf.get_variable来创建变量,设置reuse标志
    # ret0 = tf.Variable(initial_value=tf.random_normal([3, 4]), name="weights")

    # scope.reuse_variables()
    #一旦调用这个上面这个函数后,会发现就不能用tf.get_variable 创建变量了,因此我觉得 在这个函数之后,在使用
    #tf.get_variable只会查看已经有的又tf.get_variable生成的变量,
    #但是如果with tf.variable_scope("image_filters",reuse=tf.AUTO_REUSE) as scope: 而不调用scope.reuse_variables()
    #则还是可以进行正常的变量提取,有则提取有的,无则创建新的
    #如果二者同时都有,则还是不能用scope.reuse_variables()创建新的变量只能使用以前用的.

    #如果不是绝对强调一定要复用某些变量的话 则最好不要使用scope.reuse_variables()的方式 而是
    # 采用 with tf.variable_scope("image_filters",reuse=tf.AUTO_REUSE) as scope:的方式

    #这个地方其实有三种状态 reuse=tf.AUTO_REUSE(如果没有变量则会创建) True(完全的复用模式) None(继承上层的关系)

    ret0 = tf.get_variable("weights", [3,4],initializer=tf.random_normal_initializer())

    ret3 = tf.get_variable("weights1", [3, 4], initializer=tf.random_normal_initializer())
    #
    ret1 = my_image_file(input1)
    ret2 = my_image_file(input2)



init =tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print (ret0.name,sess.run(ret0))
    print (ret1.name,sess.run(ret1))
    print (ret2.name,sess.run(ret2))