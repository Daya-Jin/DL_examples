import tensorflow as tf


def mini_CNN(X_img, activation=tf.nn.relu, initializer=None,
             filters=32, conv_size=(3, 3), pool_size=(2, 2), strides=(2, 2), unit_O=10):
    conv1 = tf.layers.conv2d(X_img, filters=filters,
                             kernel_size=conv_size, padding='same',
                             activation=tf.nn.relu, kernel_initializer=initializer, name='conv1')
    pooling1 = tf.layers.max_pooling2d(conv1, pool_size=pool_size,
                                       strides=strides, name='pooling1')
    conv2 = tf.layers.conv2d(pooling1, filters=filters,
                             kernel_size=conv_size, padding='same',
                             activation=tf.nn.relu, kernel_initializer=initializer, name='conv2')
    pooling2 = tf.layers.max_pooling2d(conv2, pool_size=pool_size,
                                       strides=strides, name='pooling2')
    conv3 = tf.layers.conv2d(pooling2, filters=filters,
                             kernel_size=conv_size, padding='same',
                             activation=tf.nn.relu, kernel_initializer=initializer, name='conv3')
    pooling3 = tf.layers.max_pooling2d(conv3, pool_size=pool_size,
                                       strides=strides, name='pooling3')
    logits = tf.layers.dense(tf.layers.flatten(pooling3), unit_O,
                             activation=None, kernel_initializer=initializer)

    return logits
