import tensorflow as tf
import time
import sys

class Timer(object):
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


class SolverWrapper(object):

    LEARNING_RATE = 0.001
    SOLVER = "MOMENT"
    MOMENTUM = 0.9
    STEPSIZE = 1000
    GAMMA = 0.1
    DISPLAY = 10

    def __init__(self, sess, network, getBatchData, pretrained_model = None):

        self.net = network
        self.dataPrepare = getBatchData

        self.saver = tf.train.Saver(max_to_keep=5, write_version=tf.train.SaverDef.V2)


    def trainModel(self,sess, max_iters, restore=False):

        total_loss = self.net.calcLoss()
        otherOp = self.net.extraOp

        #建立优化器
        lr = tf.Variable(self.LEARNING_RATE, trainable=False)

        if self.SOLVER == 'Adam':
            opt = tf.train.AdamOptimizer(self.LEARNING_RATE)
        elif self.SOLVER == 'RMS':
            opt = tf.train.RMSPropOptimizer(self.LEARNING_RATE)
        else:
            # lr = tf.Variable(0.0, trainable=False)
            momentum = self.MOMENTUM
            opt = tf.train.MomentumOptimizer(lr, momentum)

        global_step_train = tf.Variable(0, trainable=False)

        with_clip = True

        if with_clip:

            tvars = tf.trainable_variables()

            grads, norm = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), 10.0)

            train_op = opt.apply_gradients(list(zip(grads, tvars)), global_step=global_step_train)

        else:
            train_op = opt.minimize(total_loss, global_step=global_step_train)


        # trainStep = [train_op, otherOp]

        image_batch, label_batch = self.dataPrepare("/home/tcl/Project/GitPrj/TensorFlowTutorials/ModelPractice/Utils/cat_dog.tfrecords", 32)

        sess.run(tf.global_variables_initializer())

        timer = Timer()
        restore_iter = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        global_step = 0

        # g_list = tf.global_variables()
        # bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        # bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]

        for iter in range(restore_iter, max_iters):

            timer.tic()

            # global_step = global_step + 1
            # 更新学习率
            if iter != 0 and iter % self.STEPSIZE == 0:
                sess.run(tf.assign(lr, lr.eval() * self.GAMMA))
                print(lr)

            #得到一个batch数据

            global_step = global_step + 1

            batchImage, batchLabel = sess.run([image_batch, label_batch])

            feed_dict={
                self.net.data: batchImage,
                self.net.label: batchLabel,
                self.net.isTrain : True,
                self.net.globalStep:global_step
            }

            res_fetches=[]

            fetch_list = [train_op] + otherOp
            fetch_list.append(total_loss)

            total_loss_val = sess.run(fetches=fetch_list, feed_dict=feed_dict)

            _diff_time = timer.toc(average=False)


            if (iter) % (self.DISPLAY) == 0:
                print('iter: %d / %d, total loss: %.4f, lr: %f'%\
                        (iter, max_iters, total_loss_val[-1], lr.eval()))
                print('speed: {:.3f}s / iter'.format(_diff_time))


        coord.request_stop()
        coord.join(threads)