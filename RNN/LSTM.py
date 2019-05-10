import tensorflow as tf


def gen_lstm_layer(inputs, unit_I, unit_O, t_size=5, batch_size=32, init=None):
    '''
    生成一层LSTM
    inputs: 序列数据，维度为(n_samples,t_size,n_features)
    '''
    if not init:
        init = tf.random_uniform_initializer(-1, 1)

    def gen_params(unit_I, unit_O):
        '''
        生成权重与偏置参数
        '''
        w_x = tf.get_variable('w_x', shape=[unit_I, unit_O])
        w_h = tf.get_variable('w_h', shape=[unit_O, unit_O])
        b = tf.get_variable('bias', shape=[1, unit_O],
                            initializer=tf.constant_initializer(0.0))
        return w_x, w_h, b

    with tf.variable_scope('LSTM_layer', initializer=init):
        with tf.variable_scope('i'):
            w_ix, w_ih, b_i = gen_params(unit_I, unit_O)
        with tf.variable_scope('f'):
            w_fx, w_fh, b_f = gen_params(unit_I, unit_O)
        with tf.variable_scope('g'):
            w_gx, w_gh, b_g = gen_params(unit_I, unit_O)
        with tf.variable_scope('o'):
            w_ox, w_oh, b_o = gen_params(unit_I, unit_O)

        # 初始的c与h，零初始化
        c = tf.Variable(tf.zeros([batch_size, unit_O]), trainable=False)
        h = tf.Variable(tf.zeros([batch_size, unit_O]), trainable=False)

        for t in range(t_size):
            input_t = inputs[:, t, :]  # 提取时间维度
            input_t = tf.reshape(input_t, [batch_size, unit_I])

            f = tf.sigmoid(tf.matmul(input_t, w_fx) + tf.matmul(h, w_fh) + b_f)
            i = tf.sigmoid(tf.matmul(input_t, w_ix) + tf.matmul(h, w_ih) + b_i)
            g = tf.tanh(tf.matmul(input_t, w_gx) + tf.matmul(h, w_gh) + b_g)
            o = tf.sigmoid(tf.matmul(input_t, w_ox) + tf.matmul(h, w_oh) + b_o)

            c = c * f + g * i
            h = o * tf.tanh(c)

        return h


if __name__ == '__main__':
    from dataset.dataset import load_cifar10
