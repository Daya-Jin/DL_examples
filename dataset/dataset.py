from sklearn.preprocessing import MinMaxScaler
import numpy as np
import struct
import os


##################### MNIST ######################

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
        while self._idx+self._batch_size < self._n_samples:
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


##################### MNIST ######################

##################### CIFAR-10 ######################

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

##################### CIFAR-10 ######################


if __name__ == '__main__':
    # _, _ = load_mnist(32)
    _, _ = load_cifar10()
