import tensorflow as tf


class LeNet:
    def __init__(self, n_features):
        self._get_params(n_features)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_input_part()
            self._build_LeNet()
            self._build_other_part()
            self._get_conf()

    def _get_params(self, n_features):
        '''
        模型参数包含在这里
        :param n_features: 输入数据的特征数
        :return:
        '''
        self.unit_I = n_features  # 输入单元数，等于特征数

        self.n_filters = [5, 16, 120]  # 卷积核数量
        self.conv_sizes = [(5, 5), (5, 5), (1, 1)]  # 卷积核尺寸

        self.pool_size = (2, 2)  # 池化核尺寸
        self.strides = (2, 2)  # 核移动的步长

        self.FC_size = 84  # 全连接层单元数

        self.unit_O = 10  # 输出单元数，类别数

        self.lr = 1e-3

    def _build_input_part(self):
        # 输入必须是可由用户指定的，所以设为placeholder
        self.X = tf.placeholder(tf.float32,
                                [None, self.unit_I])  # 数据的样本数不指定，只指定特征数
        self.Y = tf.placeholder(tf.int64, [None])  # 目标值为列向量，int64为了兼容
        # 转为图片格式送入模型，(n_samples,width,height,depth)
        self.X_img = tf.reshape(self.X, [-1, 28, 28, 1])
        self.training = tf.placeholder_with_default(False, shape=[],
                                                    name='training')

    def _build_LeNet(self):
        # 网络结构图
        with tf.name_scope('LeNet-5'):
            C1 = tf.layers.conv2d(self.X_img, filters=self.n_filters[0],
                                  kernel_size=self.conv_sizes[0], padding='same',
                                  activation=tf.nn.tanh, name='C1')
            S2 = tf.layers.max_pooling2d(C1, pool_size=self.pool_size,
                                         strides=self.strides, name='S2')
            C3 = tf.layers.conv2d(S2, filters=self.n_filters[1],
                                  kernel_size=self.conv_sizes[1],
                                  activation=tf.nn.tanh, name='C3')
            S4 = tf.layers.max_pooling2d(C3, pool_size=self.pool_size,
                                         strides=self.strides, name='S4')
            C5 = tf.layers.conv2d(S4, filters=self.n_filters[2],
                                  kernel_size=self.conv_sizes[2],
                                  activation=tf.nn.tanh, name='C5')
            FC6 = tf.layers.dense(tf.layers.flatten(C5), self.FC_size,
                                  activation=tf.nn.tanh)
            # 最后一层直接输出logits，无激活函数
            self.logits = tf.layers.dense(FC6, self.unit_O)

    def _build_other_part(self):
        # 评估图
        with tf.name_scope('Eval'):
            # 计算一维向量与onehot向量之间的损失
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.Y,
                                                               logits=self.logits)
            self.predict = tf.argmax(self.logits, 1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predict,
                                                            self.Y), tf.float32))

        # 优化图
        with tf.name_scope('train_op'):
            self.train_op = tf.train.AdamOptimizer(self.lr) \
                .minimize(self.loss)

    def _get_conf(self):
        self.init = tf.global_variables_initializer()
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True  # 按需使用显存
