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


##################### MNIST END ######################

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


##################### CIFAR-10 END ######################

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


##################### MovieLens END ######################


if __name__ == '__main__':
    # _, _ = load_mnist(32)
    # _, _ = load_cifar10()
    _,_=load_ml()