import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow.python.ops import tensor_array_ops
import seaborn as sns


power = 4
t = 0.4
sum_ = (math.pow(1-t, power + 1) - math.pow(-t, power + 1)) / (power + 1)  #求积分
x = np.linspace(0, 1, 100)
#常数值c
c = 0.6**4/sum_
cc = [c for xi in x]
plt.plot(x, cc, '--',label='c*f(x)')
#目标概率密度函数的值f(x)
y = [math.pow(xi - t, power)/sum_ for xi in x]
plt.plot(x, y,label='f(x)')



def body(x, y):

    x = tf.random_uniform([])
    y = tf.random_uniform([])

    return x, y


def condition(x, y):

    c = tf.convert_to_tensor(7.3636)
    split_val = tf.convert_to_tensor(0.4)
    power = tf.convert_to_tensor(4.0)

    x = x - split_val

    a = tf.pow(x, power)

    b = tf.multiply(y, c)

    d = tf.less(b, a)

    result = tf.cond(d,lambda : tf.constant(False),lambda : tf.constant(True))

    return result


samples = []

def getOneSample():

    currentValueX = tf.random_uniform([])
    currentValueY = tf.random_uniform([])

    currentValueX, currentValueY = tf.while_loop(condition, body, [currentValueX, currentValueY])

    return currentValueX

currentValueX = tf.random_uniform([])
currentValueY = tf.random_uniform([])
debug = condition(currentValueX, currentValueY)

init = tf.global_variables_initializer()

oneSample = getOneSample()

with tf.Session() as sess:

    sess.run(init)
    # sess.run(init1)

    for i in range(1000):

        result = sess.run([oneSample])

        samples.append(result[0])

        print("Now Iter is %d"%(i))


plt.hist(samples, bins=50, normed=True,label='sampling')
plt.legend()
plt.show()

