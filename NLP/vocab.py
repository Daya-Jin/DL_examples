# 使用TensorFlow文件接口以支持HDFS
from tensorflow import gfile


class Vocab:
    '''
    词典类，接受一个词典映射文件作为输入，生成一个文本编码解码器
    '''

    def __init__(self, voc_file: str, cnt_thresh: int):
        '''
        :param voc_file: 词典文件路径
        :param cnt_thresh: 单词出现的阈值，低于阈值的单词将不被编码
        '''
        self._word2id = dict()
        self._id2word = dict()
        self._unk = 0
        self._cnt_thresh = cnt_thresh
        self._load_voc(voc_file)

    def _load_voc(self, voc_file: str):
        with gfile.GFile(voc_file, 'r') as fd:
            data = fd.readlines()

        for line in data:
            idx, word, cnt = line.strip().split('\t')
            idx = int(idx)
            cnt = int(cnt)

            if cnt < self._cnt_thresh:
                continue

            self._word2id[word] = idx
            self._id2word[idx] = word

    def word2id(self, word: str):
        '''
        单次级别的编码
        :param word:
        :return:
        '''
        return self._word2id.get(word, self._unk)

    def id2word(self, idx: int):
        '''
        单次级别的解码
        :param idx:
        :return:
        '''
        return self._id2word.get(idx, '<UNK>')

    def s2id(self, s: str):
        '''
        句子级别编码
        :param s:
        :return:
        '''
        return [self.word2id(word) for word in s.split(' ')]

    def id2s(self, idxs) -> str:
        '''
        句子级别解码
        :param idxs:
        :return:
        '''
        return ' '.join([self.id2word(idx) for idx in idxs])

    @property
    def unk(self) -> int:
        return self._unk

    @property
    def size(self) -> int:
        return len(self._word2id)
