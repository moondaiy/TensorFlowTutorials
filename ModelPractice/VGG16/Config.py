import tensorflow  as tf

"""
定义VGG16的配置文件
"""

FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer(
    'height', 224, 'the height of image')

tf.app.flags.DEFINE_boolean(
    'width', 224, 'the width of image')

tf.app.flags.DEFINE_boolean(
    'channel', 3, 'the channel of image')

tf.app.flags.DEFINE_boolean(
    'trainable', True, 'vgg 16 trainable')