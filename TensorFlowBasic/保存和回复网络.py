import tensorflow as tf

# saveTensor1 = tf.Variable(initial_value=[[1,2],[2,3]],name="tensor1")   #是保存模型的时候的数值
saveTensor1 = tf.Variable(initial_value=[[2,2],[2,3]],name="tensor1")   #是回复模型前 初始化的数值
saveTensor2 = tf.Variable(initial_value=[[4,5],[6,7]],name="tensor2")
saveTensor3 = tf.Variable(initial_value=[[1,1],[1,1]],name="tensor3")

init = tf.global_variables_initializer()


#max_to_keep 表示最多可以保存的模型数量,后续的max_to_keep个
#[saveTensor1, saveTensor2] 表示需要保存的tensor的个数,没有对saveTensor3 进行保存
saver = tf.train.Saver([saveTensor1, saveTensor2], max_to_keep=4)

isTrain = False

with tf.Session() as sess:

    if isTrain == True:

        sess.run(init)

        for i in range(3):
            #global_step  对保存的模型后缀加个标示,如果没有这个,则只会保存一个checkpoint
            save_path = saver.save(sess, "my_net/save_net.ckpt",global_step=i)

        print(save_path)

    else:

        #首先初始化变量,这时候所有tensor会进行初始化操作
        sess.run(init)

        # 打印操作
        print(sess.run(saveTensor1))
        print(sess.run(saveTensor2))

        # restore操作
        saver.restore(sess, "my_net/save_net.ckpt-2")

        # restore操作 之后的数值 可以发现是被更改了
        print(sess.run(saveTensor1))
        print(sess.run(saveTensor2))
        print(sess.run(saveTensor3))



