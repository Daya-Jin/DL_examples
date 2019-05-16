from sklearn.preprocessing import MinMaxScaler
import numpy as np
import struct
import os


##################### MNIST START ######################

def load_image(path):
    with open(path, 'rb') as fd:
        _, _, _, _ = struct.unpack('>IIII', fd.read(16))
        res = np.fromfile(fd, dtype=np.uint8).reshape(-1, 784)
    return res


def load_label(path):
    with open(path, 'rb') as fd:
        _, _ = struct.unpack('>II', fd.read(8))
        res = np.fromfile(fd, dtype=np.uint8)
    return res


class MnistData:
    def __init__(self, data_path, label_path, batch_size=32, normalize=False, shuffle=False):
        '''
        paths: 文件路径
        '''
        self.data = list()
        self.target = list()
        self._n_samples = 0
        self.n_features = 0

        self._idx = 0  # mini-batch的游标
        self._batch_size = batch_size

        self._load(data_path, label_path)

        if shuffle:
            self._shuffle_data()
        if normalize:
            self._normalize_data()

        print(self.data.shape, self.target.shape)

    def _load(self, data_path, label_path):
        '''
        载入数据
        '''
        self.data = load_image(data_path)
        self.target = load_label(label_path)

        self._n_samples, self.n_features = self.data.shape[0], self.data.shape[1]

    def _shuffle_data(self):
        '''
        打乱数据
        '''
        idxs = np.random.permutation(self._n_samples)
        self.data = self.data[idxs]
        self.target = self.target[idxs]

    def _normalize_data(self):
        scaler = MinMaxScaler()
        self.data = scaler.fit_transform(self.data)

    def next_batch(self):
        '''
        生成mini-batch
        '''
        while self._idx + self._batch_size < self._n_samples:
            yield self.data[self._idx: (self._idx + self._batch_size)], self.target[
                                                                        self._idx: (self._idx + self._batch_size)]
            self._idx += self._batch_size

        self._idx = 0
        self._shuffle_data()


def load_mnist(batch_size=64):
    MNIST_DIR = os.path.join(os.path.dirname(__file__), 'MNIST')
    train_data_path = os.path.join(MNIST_DIR, 'train-images.idx3-ubyte')
    train_label_path = os.path.join(MNIST_DIR, 'train-labels.idx1-ubyte')
    test_data_path = os.path.join(MNIST_DIR, 't10k-images.idx3-ubyte')
    test_label_path = os.path.join(MNIST_DIR, 't10k-labels.idx1-ubyte')

    train_data = MnistData(train_data_path, train_label_path, batch_size=batch_size,
                           normalize=True, shuffle=True)
    test_data = MnistData(test_data_path, test_label_path, batch_size=batch_size,
                          normalize=True, shuffle=False)

    return train_data, test_data


##################### CIFAR-10 START ######################

def unpickle(file):
    '''
    CIFAR-10数据读取函数
    '''
    import pickle
    with open(file, 'rb') as fd:
        data = pickle.load(fd, encoding='bytes')
    return data[b'data'], np.array(data[b'labels'])


class CifarData:
    def __init__(self, paths, batch_size=32, normalize=False, shuffle=False):
        '''
        paths: 文件路径
        '''
        self.data = list()
        self.target = list()
        self._n_samples = 0
        self.n_features = 0

        self._idx = 0  # mini-batch的游标
        self._batch_size = batch_size

        self._load(paths)

        if shuffle:
            self._shuffle_data()
        if normalize:
            self._normalize_data()

        print(self.data.shape, self.target.shape)

    def _load(self, paths):
        '''
        载入数据
        '''
        for path in paths:
            data, labels = unpickle(path)
            self.data.append(data)
            self.target.append(labels)

        # 将所有批次的数据拼接起来
        self.data, self.target = np.vstack(
            self.data), np.hstack(self.target)

        self._n_samples, self.n_features = self.data.shape[0], self.data.shape[1]

    def _shuffle_data(self):
        '''
        打乱数据
        '''
        idxs = np.random.permutation(self._n_samples)
        self.data = self.data[idxs]
        self.target = self.target[idxs]

    def _normalize_data(self):
        scaler = MinMaxScaler()
        self.data = scaler.fit_transform(self.data)

    def next_batch(self):
        '''
        生成mini-batch
        '''
        while self._idx + self._batch_size < self._n_samples:
            yield self.data[self._idx: (self._idx + self._batch_size)], self.target[
                                                                        self._idx: (self._idx + self._batch_size)]
            self._idx += self._batch_size

        self._idx = 0
        self._shuffle_data()


def load_cifar10(batch_size=64):
    CIFAR_DIR = os.path.join(os.path.dirname(__file__), "cifar-10-batches-py")
    train_filenames = [os.path.join(CIFAR_DIR, 'data_batch_{}'.format(i))
                       for i in range(1, 6)]
    test_filenames = [os.path.join(CIFAR_DIR, 'test_batch')]

    train_data = CifarData(train_filenames, batch_size=batch_size,
                           normalize=True, shuffle=True)
    test_data = CifarData(test_filenames, batch_size=batch_size,
                          normalize=True, shuffle=False)
    return train_data, test_data


##################### MovieLens START ######################

class MLData:
    def __init__(self, path, batch_size=32, shuffle=True):
        self._data = list()
        self._target = list()
        self._n_samples = 0
        self._n_features = 0

        self._idx = 0  # mini-batch的游标
        self._batch_size = batch_size

        self._load(path)

        if shuffle:
            self._shuffle_data()

        print(self._data.shape, self._target.shape)

    def _load(self, path):
        tmp = np.load(path, allow_pickle=True)
        self._data = tmp[:, :-1]
        self._target = tmp[:, -1]

        self._n_samples, self.n_features = self._data.shape[0], self._data.shape[1]

    def _shuffle_data(self):
        '''
        打乱数据
        '''
        idxs = np.random.permutation(self._n_samples)
        self._data = self._data[idxs]
        self._target = self._target[idxs]

    def next_batch(self):
        '''
        生成mini-batch
        '''
        while self._idx < self._n_samples:
            yield self._data[self._idx: (self._idx + self._batch_size)], self._target[
                                                                         self._idx: (self._idx + self._batch_size)]
            self._idx += self._batch_size

        self._idx = 0
        self._shuffle_data()

    @property
    def u_id(self):
        return np.array(self._data[:, 0], dtype=np.int32)

    @property
    def u_occu(self):
        return np.array(self._data[:, 2], dtype=np.int32)

    @property
    def u_age_gender(self):
        return np.array(self._data[:, 1], dtype=np.int32)

    @property
    def m_id(self):
        return np.array(self._data[:, 3], dtype=np.int32)

    @property
    def m_title(self):
        return self._data[:, 4]

    @property
    def m_genres(self):
        return self._data[:, 5]

    @property
    def m_year(self):
        return np.array(self._data[:, 6], dtype=np.int32)


def load_ml(batch_size=64):
    ML_DIR = os.path.join(os.path.dirname(__file__), "movielens")
    train_filename = os.path.join(ML_DIR, 'train.npy')
    test_filename = os.path.join(ML_DIR, 'test.npy')

    train_data = MLData(train_filename, batch_size=batch_size)
    test_data = MLData(test_filename, batch_size=batch_size)
    return train_data, test_data


##################### news_CN START ######################

class TextData:
    def __init__(self, filename, vocal, cat_dict, t_size=5, batch_size=32, shuffle=True):
        '''
        :param filename: 经过分词的文本文件，每行的格式为'label    text'
        :param vocal: 文本编码器
        :param cat_dict: 类别编码器
        :param t_size: 时间尺寸
        :param batch_size:
        :param shuffle:
        '''
        self._data = list()
        self._target = list()
        self._n_samples = 0

        self._idx = 0  # mini-batch的游标
        self._batch_size = batch_size

        self._vocal = vocal
        self._cat_dict = cat_dict
        self._t_size = t_size

        self._load_data(filename)

        if shuffle:
            self._shuffle_data()

        print(self._data.shape, self._target.shape)

    def _load_data(self, filename):
        with open(filename, 'r', encoding='utf-8') as fd:
            text = fd.readlines()

        for line in text:
            label, content = line.strip().split('\t')
            x = self._vocal.s2id(content)
            y = self._cat_dict.cat2id(label)

            x = x[:self._t_size]
            n_pad = self._t_size - len(x)  # 需要填充的位数
            x = x + [self._vocal.unk for _ in range(n_pad)]

            self._data.append(x)
            self._target.append(y)

        self._data = np.array(self._data)
        self._target = np.array(self._target)
        self._n_samples = self._data.shape[0]

    def _shuffle_data(self):
        '''
        打乱数据
        '''
        idxs = np.random.permutation(self._n_samples)
        self._data = self._data[idxs]
        self._target = self._target[idxs]

    def next_batch(self):
        '''
        生成mini-batch
        '''
        while self._idx + self._batch_size < self._n_samples:
            yield self._data[self._idx: (self._idx + self._batch_size)], self._target[
                                                                         self._idx: (self._idx + self._batch_size)]
            self._idx += self._batch_size

        self._idx = 0
        self._shuffle_data()

    @property
    def voc_size(self):
        return self._vocal.size


def load_news(batch_size=32, cnt_thresh=10, t_size=5):
    '''
    载入news数据
    :param batch_size:
    :param cnt_thresh:
    :param t_size:
    :return:
    '''
    from dataset.news_CN.utils import CatDict
    from NLP.vocab import Vocab

    NEWS_DIR = os.path.join(os.path.dirname(__file__), "news_CN")
    # 分词后的文件
    seg_train_file = os.path.join(NEWS_DIR, 'cnews.seg_train.txt')
    seg_val_file = os.path.join(NEWS_DIR, 'cnews.seg_val.txt')
    seg_test_file = os.path.join(NEWS_DIR, 'cnews.seg_test.txt')
    # 词表
    vocal_table = os.path.join(NEWS_DIR, 'cnews.vocal.txt')
    # 类别表
    cat_file = os.path.join(NEWS_DIR, 'cnews.cat.txt')

    voc_cls = Vocab(vocal_table, cnt_thresh)  # 文本编码器
    cat_dict = CatDict(cat_file)  # 类别编码器

    train_data = TextData(seg_train_file, voc_cls, cat_dict,
                          t_size, batch_size=batch_size)
    test_data = TextData(seg_test_file, voc_cls, cat_dict,
                         t_size, batch_size=batch_size)

    return train_data, test_data


if __name__ == '__main__':
    # _, _ = load_mnist(32)
    # _, _ = load_cifar10()
    # _, _ = load_ml()
    _, _ = load_news()
