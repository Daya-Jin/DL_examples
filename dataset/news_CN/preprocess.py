import jieba

# raw file
train_file = 'cnews.train.txt'
val_file = 'cnews.val.txt'
test_file = 'cnews.test.txt'

# 分词后的文件
seg_train_file = 'cnews.seg_train.txt'
seg_val_file = 'cnews.seg_val.txt'
seg_test_file = 'cnews.seg_test.txt'

# 词表
vocal_table = 'cnews.vocal.txt'

# 类别表
cat_file = 'cnews.cat.txt'


def gen_seg_file(file_in, file_out):
    '''
    生成分词后的文件
    :param file_in: 原始未分词的文件
    :param file_out: 输出文件，词语使用' '分隔
    :return:
    '''
    with open(file_in, 'r', encoding='utf-8') as fd:
        text = fd.readlines()
    with open(file_out, 'w', encoding='utf-8') as fd:
        for line in text:
            label, data = line.strip().split('\t')
            words = jieba.cut(data)
            words_trans = ''

            # 去除切分出来的空白词
            for word in words:
                word = word.strip()
                if word != '':
                    words_trans += word + ' '

            out_line = '{}\t{}\n'.format(label, words_trans.strip())
            fd.write(out_line)


def gen_vocab(file_in, file_out):
    '''
    生成词典文件，每行格式为'idx word word_cnt'
    :param file_in:
    :param file_out:
    :return:
    '''
    with open(file_in, 'r', encoding='utf-8') as fd:
        text = fd.readlines()

    word_dict = dict()
    for line in text:
        _, data = line.strip().split('\t')
        for word in data.split():
            word_dict.setdefault(word, 0)
            word_dict[word] += 1
    word_dict = sorted(word_dict.items(), key=lambda x: x[1],  # 以频数排序
                       reverse=True)

    with open(file_out, 'w', encoding='utf-8') as fd:
        fd.write('0\t<UNK>\t99999\n')
        for idx, item in enumerate(word_dict):
            fd.write('{}\t{}\t{}\n'.format(idx + 1, item[0], item[1]))


def gen_cat(file_in, file_out):
    '''
    生成类别编码文件
    :param file_in:
    :param file_out:
    :return:
    '''
    with open(file_in, 'r', encoding='utf-8') as fd:
        text = fd.readlines()

    label_dict = dict()
    for line in text:
        label, _ = line.strip().split('\t')
        label_dict.setdefault(label, 0)
        label_dict[label] += 1
    label_dict = sorted(label_dict.items(), key=lambda x: x[1],
                        reverse=True)

    with open(file_out, 'w', encoding='utf-8') as fd:
        for idx, item in enumerate(label_dict):
            fd.write('{}\t{}\t{}\n'.format(idx, item[0], item[1]))


if __name__ == '__main__':
    # 首先对原始文件进行分词
    gen_seg_file(train_file, seg_train_file)
    gen_seg_file(val_file, seg_val_file)
    gen_seg_file(test_file, seg_test_file)

    # 使用分词后的训练文件来构建词典
    gen_vocab(seg_train_file, vocal_table)

    # 使用训练文件来生成类别编码
    gen_cat(train_file, cat_file)
