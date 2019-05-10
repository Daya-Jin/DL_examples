import numpy as np
import os
import tensorflow as tf


class VGG16:
    def __init__(self):
        self.VGG_MEAN = [103.939, 116.779, 123.68]  # 图片需要减去的mean

        load_path = os.path.join(os.path.dirname(__file__), 'vgg16.npy')
        self.param_dict = np.load(load_path, allow_pickle=True,
                                  encoding='latin1').item()

    def get_conv_filter(self, name):
        '''
        获得卷积w参数
        '''
        return tf.constant(self.param_dict[name][0], name='conv')

    def get_fc_weight(self, name):
        '''
        获得FC的w参数
        '''
        return tf.constant(self.param_dict[name][0], name='fc')

    def get_bias(self, name):
        '''
        获得b参数
        '''
        return tf.constant(self.param_dict[name][1], name='bias')

    def conv_layer(self, x, name):
        '''
        构建卷积层
        '''
        with tf.name_scope(name):
            conv_w = self.get_conv_filter(name)
            conv_b = self.get_bias(name)
            conv_layer = tf.nn.conv2d(x, conv_w, [1, 1, 1, 1], padding='SAME')
            conv_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, conv_b))
            return conv_layer

    def pooling_layer(self, x, name):
        '''
        构建池化层
        '''
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def fc_layer(self, x, name, activation=tf.nn.relu):
        '''
        构建全连接层
        '''
        with tf.name_scope(name):
            fc_w = self.get_fc_weight(name)
            fc_b = self.get_bias(name)
            fc_layer = tf.nn.bias_add(tf.matmul(x, fc_w), fc_b)

            if not activation:
                return fc_layer
            else:
                return activation(fc_layer)

    def flatten(self, x):
        '''
        实现tf.flatten的效果，用于连接pooling与FC
        '''
        x_shape = x.get_shape().as_list()
        # 除第一维外的所有维度相乘
        dim = 1
        for d in x_shape[1:]:
            dim *= d
        x = tf.reshape(x, [-1, dim])  # flatten
        return x

    def structure(self, inputs):
        self.conv1_1 = self.conv_layer(inputs, 'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, 'conv1_2')
        self.pool1 = self.pooling_layer(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, 'conv2_2')
        self.pool2 = self.pooling_layer(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 'conv3_3')
        self.pool3 = self.pooling_layer(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 'conv4_3')
        self.pool4 = self.pooling_layer(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, 'conv5_3')
        self.pool5 = self.pooling_layer(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.flatten(self.pool5), 'fc6')
        self.fc7 = self.fc_layer(self.fc6, 'fc7')
        self.fc8 = self.fc_layer(self.fc7, 'fc8', activation=None)
        self.logits = tf.nn.softmax(self.fc8, name='logits')

        del self.param_dict

    def build(self, img):
        '''
        构建VGG16
        '''
        # 注意VGG网络使用的图像不是RGB通道的，而是BGR通道的，需要转换
        r, g, b = tf.split(img, num_or_size_splits=3, axis=3)
        img_trans = tf.concat([b - self.VGG_MEAN[0],
                               g - self.VGG_MEAN[1],
                               r - self.VGG_MEAN[2]], axis=3)

        assert img_trans.get_shape().as_list()[1:] == [224, 224, 3]

        self.structure(img_trans)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 数据准备
    raw_img = tf.gfile.FastGFile('../dataset/img/cock.jpg', 'rb').read()
    img = tf.image.convert_image_dtype(tf.image.decode_jpeg(raw_img),
                                       dtype=tf.float32)
    img = tf.reshape(tf.image.resize_images(img, (224, 224), method=0), [-1, 224, 224, 3])
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8)

    # 搭建网络
    X = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg16 = VGG16()
    vgg16.build(X)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 按需使用显存

    # 测试网络
    with tf.Session(config=config) as sess:
        img_input = img.eval()

        plt.clf()
        plt.imshow(img_input[0])
        plt.show()

        pred = sess.run(vgg16.logits, feed_dict={X: img_input})

    print(pred.argmax())  # 这里应该输出7
