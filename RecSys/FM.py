import tensorflow as tf


class FMLayer(tf.keras.layers.Layer):
    def __init__(self, k):
        '''
        自定义FM层

        :param k: 隐向量的维度
        '''
        super(FMLayer, self).__init__()

        self.linear_part = tf.keras.layers.Dense(1, activation=None)

        self.k = k

    def build(self, input_shape):
        '''

        :param input_shape: tf会自动根据输入来获取input_shape
        :return:
        '''
        # 矩阵V的形状为(n_features, k)
        self.v = self.add_weight(shape=(input_shape[-1], self.k),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True)

    def call(self, inputs):
        linear = self.linear_part(inputs)

        wide = 0.5 * tf.reduce_sum(tf.pow(tf.matmul(inputs, self.v), 2) -
                                   tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2)),
                                   axis=1, keepdims=True)
        return linear + wide


class FMClassifier(tf.keras.Model):
    def __init__(self, k):
        '''
        因子分解机，二分类模式

        :param k: 隐向量的维度
        '''
        super(FMClassifier, self).__init__()

        self.fm_layer = FMLayer(k)

    def call(self, inputs):
        output = tf.nn.sigmoid(self.fm_layer(inputs))
        return output


class FMRegressor(tf.keras.Model):
    def __init__(self, k):
        '''
        因子分解机，回归模式

        :param k: 隐向量的维度
        '''
        super(FMRegressor, self).__init__()

        self.fm_layer = FMLayer(k)

    def call(self, inputs):
        return self.fm_layer(inputs)


if __name__ == "__main__":
    import numpy as np

    fm = FMRegressor(5)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    fm.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())

    x_data = np.array([
        #    Users  |     Movies     |    Movie Ratings   | Time | Last Movies Rated
        #   A  B  C | TI  NH  SW  ST | TI   NH   SW   ST  |      | TI  NH  SW  ST
        [1, 0, 0, 1, 0, 0, 0, 0.3, 0.3, 0.3, 0, 13, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0.3, 0.3, 0.3, 0, 14, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0.3, 0.3, 0.3, 0, 16, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0.5, 0.5, 5, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0.5, 0.5, 8, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 0, 0.5, 0, 0.5, 0, 9, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0.5, 0, 0.5, 0, 12, 1, 0, 0, 0]
    ])
    # ratings
    y_data = np.array([5, 3, 1, 4, 5, 1, 5])

    # Let's add an axis to make tensoflow happy.
    y_data.shape += (1,)

    hist = fm.fit(x_data, y_data, epochs=10)
