import os
import tensorflow as tf
from tensorflow import gfile, logging
import numpy as np


def gen_voc(cap_path, output_path):
    '''
    以caption文件构造词典
    :param cap_path: caption文件路径
    '''
    with open(cap_path, 'r', encoding='utf-8') as fd:
        text = fd.readlines()

    len_dict = dict()
    voc_dict = dict()

    for line in text:
        img_id, cap = line.strip().split('\t')
        words = cap.strip().split()

        len_dict.setdefault(len(words), 0)
        len_dict[len(words)] += 1

        for word in words:
            voc_dict.setdefault(word, 0)
            voc_dict[word] += 1

    voc_list = sorted(voc_dict.items(), key=lambda x: x[1], reverse=True)

    with open(output_path, 'w', encoding='utf-8') as fd:
        fd.write('<unk>\t99999\n')  # 未知单词
        for item in voc_list:
            fd.write('{}\t{}\n'.format(item[0], item[1]))  # 写入(word,word_cnt)

    return voc_dict  # 返回词典字典备用


# def gen_imgName(cap_path, output_path):
#     img2cap=dict()
#     with gfile.GFile(cap_path,'r') as fd:
#         text=fd.readlines()
#
#     for line in text:
#         img_name=line.strip().split('\t')[0].split('#')[0]
#         img2cap.setdefault(img_name,'')
#         img2cap[]


if __name__ == '__main__':
    caption_file = os.path.join(os.path.dirname(__file__), 'results_20130124.token')
    output_file = os.path.join(os.path.dirname(__file__), 'vocab.txt')
    _ = gen_voc(caption_file, output_file)
