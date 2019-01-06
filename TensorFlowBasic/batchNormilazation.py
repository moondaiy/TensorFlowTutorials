
import tensorflow as tf


tf.set_random_seed(1)
out_size=3
img = tf.Variable(tf.random_normal([2, out_size],mean=5))   #2行3列的数据 每一列是一个单独的数据 这2数据组成了1个batch
# img = tf.Variable(tf.random_normal([16, 5, 5, 3],mean=5))

fc_mean, fc_var = tf.nn.moments(img,axes=[0])
# fc_mean, fc_var = tf.nn.moments(img,axes=[0,1,2])

scale = tf.Variable(tf.ones([out_size]))
shift = tf.Variable(tf.zeros([out_size]))
epsilon = 0.001

#batch_normalization 这个就是直接他套用公式进行了.
#这个其实没涉及到反向传播更新r 和 b 直接相当与直接的前向传播.
Wx_plus_b = tf.nn.batch_normalization(img, fc_mean, fc_var, shift, scale, epsilon)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


print (sess.run([img, shift, scale]))
print("-----------------------")
print (sess.run([fc_mean, fc_var]))
print("-----------------------")
print (sess.run(Wx_plus_b))
print("-----------------------")
print (sess.run([fc_mean, shift]))