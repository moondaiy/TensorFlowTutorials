import cv2
import tensorflow as tf
import numpy as np
import os

"""
创建这个文件的初期是打算针对猫狗识别进行的
数据集是Kaggle的一个比赛狗vs猫
"""

RESIZE_WIDTH  = 244
RESIZE_HEIGHT = 244

def get_files(file_dir):
    """
    :param 输入文件夹,里面包含了Train或Test(这2部分是需要分别进行传入的,分别生成tfrecoard文件):
    :return list of images and labels:
    """
    image_cats = []
    label_cats = []
    image_dogs = []
    label_dogs = []

    for file in os.listdir(file_dir):

        name = file.split(".") #根据

        #这地方是根据猫狗数据集的特点来进行的
        if name[0] == "cat":
            image_cats.append(os.path.join(file_dir,file))
            label_cats.append(0)
        elif name[0] == "dog":
            image_dogs.append(os.path.join(file_dir,file))
            label_dogs.append(1)
        else:
            print("error %s"%(file_dir + file))

    print("The dogs number is %d and the cat number is %d"%(len(image_dogs), len(image_cats)))

    image_list = np.hstack((image_cats, image_dogs))
    label_list = np.hstack((label_cats, label_dogs))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])

    label_list = [int(i) for i in label_list]

    #返回了被打乱以后的数据和标注数据
    return image_list, label_list

def int64_feature(value):

    if not isinstance(value,list):
        value = [value]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecoard(images, labels, save_dir, name):

    filename = (save_dir + name + ".tfrecords")

    if os.path.exists(filename):
        os.remove(filename)

    n_sample = len(labels)

    if np.shape(images)[0] != n_sample:
        raise ValueError("Image size %d does not match label size %d ."%(images.shape[0], n_sample))

    writer = tf.python_io.TFRecordWriter(filename)

    print("Transoform Start ...")

    for i in range(n_sample):
        try:
            image = cv2.imread(images[i])
            image = cv2.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT))

            # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #颜色空间转换,其实这地方如果训练的时候就是BGR 那么在预测的时候也是BGR就好

            image_raw = image.tostring()
            label = int(labels[i])

            example = tf.train.Example(features=tf.train.Features(feature={
                "label":int64_feature(label),
                "image_raw":bytes_feature(image_raw)
            }))

            writer.write(example.SerializeToString())

        except IOError as e:
            print("Could not read : %s"%(images[i]))
            print("error : %s "%(e))
            print("Skip it")

    writer.close()
    print("Transform Done")

def read_and_decode(tfrecord_file, bath_size):

    maxCapacity = 1000 + 3*bath_size

    filename_queue = tf.train.string_input_producer([tfrecord_file])

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    img_features = tf.parse_single_example(serialized_example,
                                    features={"label":tf.FixedLenFeature([], tf.int64),
                                                                  "image_raw":tf.FixedLenFeature([],tf.string)})

    image = tf.decode_raw(img_features["image_raw"], tf.uint8)

    image = tf.reshape(image, [RESIZE_WIDTH, RESIZE_HEIGHT,3])

    label = tf.cast(img_features["label"],tf.float32)

    image = tf.image.per_image_standardization(image) #对图像进行归一化操作,但是 这地方可以是自己减去一个均值

    image_batch, label_batch = tf.train.batch([image,label],batch_size=bath_size,num_threads=64,capacity=maxCapacity)

    label_batch = tf.cast(label_batch,dtype=tf.int32)

    label_batch = tf.one_hot(label_batch, 2, axis=1, dtype=tf.float32)

    return image_batch, tf.reshape(label_batch,[bath_size,-1])

def test_read_and_decode(tfrecord_file, bath_size):

    #测试 read_and_decode 函数
    image_batch, label_batch = read_and_decode(tfrecord_file, bath_size)

    with tf.Session() as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(10):

            images , labels = sess.run([image_batch, label_batch])

            # tf.one_hot()

            print(labels)

            shapeList = list(images.shape)

            shapeList = shapeList[1:]

            image = images.reshape(shapeList)

            # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

            cv2.imshow("current", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':

    # images , labels = get_files("/home/tcl/数据备份/dataset_kaggledogvscat/train")
    # convert_to_tfrecoard(images , labels, "./", "cat_dog")

    test_read_and_decode("./cat_dog.tfrecords", 1)






