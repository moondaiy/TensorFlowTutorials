import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


def fun(x,case):

    if case == 0:
        x = x + 0
    elif case == 1:
        x = x + 1
    elif case == 2:
        x = x + 2
    elif case == 3:
        x = x + 3

    return x

def lamba_1(x):

    return  x - 1

def lamba_2(x):

    return x*100

def func1(x):

    y = control_flow_ops.cond(tf.less(x , 2), true_fn = lambda : lamba_1(x), false_fn = lambda : lamba_2(x))

    return y

num_cases = 4

if __name__ == "__main__":

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)

        # result = control_flow_ops.merge([fun(control_flow_ops.switch([1,2,3,4],tf.equal(case, sel))[1], case) for case in range(4)])

        result = control_flow_ops.merge([func1(sel) for case in range(4)])

        result1 = sess.run(result)

        print(result1)

        # print(result[0].eval())

        # output_false,output_true = control_flow_ops.switch([5,6,7,8],True)
        #
        # print(output_true.eval())
