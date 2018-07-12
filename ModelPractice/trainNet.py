import ResNet.ResNet as ResNet
import SolverWrapper.SolverWarper as Wrapper
import tensorflow as tf
import Utils.create_records as createData





def train():

    net = ResNet.ResNet()

    with tf.Session() as sess:


        sw = Wrapper.SolverWrapper(sess, net, createData.read_and_decode)

        print('Solving...')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sw.trainModel(sess, 1000, restore=False)


        coord.request_stop()
        coord.join(threads)

        print('done solving')


if __name__ == '__main__':

    train()
