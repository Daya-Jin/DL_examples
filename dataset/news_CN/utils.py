import os


class CatDict:
    '''
    对类别做编码的数据类
    '''

    def __init__(self, cat_file):
        '''
        cat_file: 类别文件
        '''
        self._cat2id = dict()
        self._load_table(cat_file)

    def _load_table(self, filename):
        with open(filename, 'r', encoding='utf-8') as fd:
            data = fd.readlines()

        for line in data:
            idx, cat, _ = line.split('\t')
            self._cat2id[cat] = int(idx)

    def cat2id(self, cat):
        if cat not in self._cat2id:
            raise Exception('{} is not in cat'.format(cat))
        else:
            return self._cat2id[cat]
