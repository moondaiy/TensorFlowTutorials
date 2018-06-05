import tensorflow as tf

w = tf.Variable(1.0)

ema = tf.train.ExponentialMovingAverage(0.9)

update = tf.assign_add(w, 1.0)

with tf.control_dependencies([update]):
    #返回一个op,这个op用来更新moving_average,i.e. shadow value
    # w = tf.Print(w,[w],message="The value of w : ")
    # print(w)
    ema_op = ema.apply([w])#这句和下面那句不能调换顺序


# 以 w 当作 key， 获取 shadow value 的值,在实际用处就是 取出过滤后的数值
ema_val = ema.average(w)#参数不能是list，有点蛋疼

with tf.Session() as sess:

    tf.global_variables_initializer().run()

    for i in range(3):

        sess.run(ema_op)
        print(sess.run(ema_val))
        # print(sess.run(update))