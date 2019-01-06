import tensorflow as tf
import numpy as np

#dtype：数据类型。常用的是tf.float32, tf.float64等数值类型
#shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2, 3], [None, 3] 表示列是3，行不定
inputPlaceHolder1 = tf.placeholder(tf.float32, shape=[2,3])
inputPlaceHolder2 = tf.placeholder(tf.float32, shape=[3,2])

output=tf.matmul(inputPlaceHolder1,inputPlaceHolder2)

#这里没有变量，就不需要 init =tf.global_variables_initializer() 这一步了

with tf.Session() as sess:
    output_result, input1, input2 = sess.run([output,inputPlaceHolder1, inputPlaceHolder2],feed_dict={inputPlaceHolder1:np.random.rand(2,3), inputPlaceHolder2:np.random.rand(3,2)})

    print(output_result)
    print(input1)
    print(input2)
