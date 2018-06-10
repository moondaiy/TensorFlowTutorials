import tensorflow as tf
import sys
sys.path.append("../")
from Utils.create_records import read_and_decode
import VGG7 as VGG16
from Config import FLAGS



if __name__ == '__main__':

    trainModel = tf.placeholder(bool)

    image_batch, label_batch = read_and_decode("../Utils/cat_dog.tfrecords", 32)

    mode = VGG16.Vgg16(trainable = FLAGS.trainable,trainModel =  trainModel, inputData= image_batch, inputLabel= label_batch,dropOutRatio = 0.5, vgg16_npy_path=None)

    trainStep, loss ,label = mode.trainOptimizer()
    # loss = mode.cost_Layer()

    init = tf.initialize_all_variables()

    with tf.Session()as sess:

        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(1000):

            _, lossDisplay = sess.run([trainStep, loss],feed_dict={trainModel:True})

            print(lossDisplay)

        coord.request_stop()
        coord.join(threads)

    print("OK")