import tensorflow as tf
import sys
sys.path.append("../")
from Utils.create_records import read_and_decode
import VGG16 as VGG16
from Config import FLAGS



if __name__ == '__main__':

    trainModel = tf.placeholder(bool)

    image_batch, label_batch = read_and_decode("../Utils/cat_dog.tfrecords", 32)
    #需要验证集 则需要另外调用read_and_decode 生成验证集的数据集合

    mode = VGG16.Vgg16(trainable = True,trainModel =  trainModel,dropOutRatio = 0.5, vgg16_npy_path="./vgg16-save.npy")

    inputData  = tf.placeholder(dtype=tf.float32, shape= [None,244,244,3])
    inputLabel = tf.placeholder(dtype=tf.float32, shape=[None,2])

    #建立网络
    mode.build(inputData)

    #得到训练步骤
    trainStep, loss ,label = mode.trainOptimizer(inputLabel)

    #得到准确度
    accuracy = mode.calcAccuracy(inputLabel)

    #初始化所有操作
    init = tf.initialize_all_variables()

    with tf.Session()as sess:

        sess.run(init)

        # trainVarious = sess.run(tf.trainable_variables())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(20):

            #得带训练图像
            batchImage,batchLabel = sess.run([image_batch, label_batch])

            print("Image Sum is %f"%(batchImage.sum()))

            _, lossDisplay, labelDis = sess.run([trainStep, loss, label],feed_dict={ inputData:batchImage, inputLabel:batchLabel, trainModel:True})

            print(lossDisplay)


            #从这个操作来看global_step 这个参数是在反向传播完成后,进行更新的
            print("global step is %d"%(sess.run(mode.global_step)))

            print("lr is %f "%(sess.run(mode.learnRatio)))

            #进入验证阶段
            if i % 10 == 0:

                batchImage, batchLabel = sess.run([image_batch, label_batch])

                accuracyResult = sess.run([accuracy],feed_dict={inputData:batchImage, inputLabel:batchLabel, trainModel:False})

                print("Current Iter is %d and Accuracy is %f"%(i, accuracyResult[0]))


        coord.request_stop()
        coord.join(threads)

        #保存网络成npy形式
        mode.save_npy(sess)


    print("OK")