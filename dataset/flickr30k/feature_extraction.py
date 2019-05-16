import os
import tensorflow as tf
from tensorflow import gfile, logging
import numpy as np
import pickle


def parse_cap(cap_path):
    '''
    解析cap文件，返回img与cap的映射表
    对于flickr30k，每张图片对应5个caption，所以映射关系为{img:[cap0,...,cap4]}
    :param cap_path: caption文件路径
    :return:
    '''
    with gfile.GFile(cap_path, 'r') as fd:
        text = fd.readlines()

    img2cap = dict()

    for line in text:
        img_name, cap = line.strip().split('\t')
        img_name = img_name.split('#')[0]
        cap = cap.strip()

        img2cap.setdefault(img_name, list())
        img2cap[img_name].append(cap)

    return img2cap


def load_graph(pb_path):
    '''
    载入用于提取特征的模型图
    :param pb_path: pb文件路径
    :return: None
    '''
    with gfile.FastGFile(pb_path, 'rb') as fd:
        graph = tf.GraphDef()
        graph.ParseFromString(fd.read())
        _ = tf.import_graph_def(graph, name='')


if __name__ == '__main__':
    caption_file = os.path.join(os.path.dirname(__file__), 'results_20130124.token')
    model_path = os.path.join(os.path.dirname(__file__), '..', '..',
                              'CNN', 'Models', 'inception-v3', 'classify_image_graph_def.pb')
    img_dir = os.path.join(os.path.dirname(__file__), 'flickr30k-images')
    feature_dir = os.path.join(os.path.dirname(__file__), 'features')

    logging.info('Create Dir')
    if not gfile.Exists(feature_dir):
        gfile.MakeDirs(feature_dir)

    img2cap = parse_cap(caption_file)
    img_names = list(img2cap.keys())

    batch_size = 1000
    n_batches = len(img_names) // batch_size
    n_batches += 1 if len(img_names) // batch_size else 0

    logging.info('Loading Model...')
    load_graph(model_path)

    with tf.Session() as sess:
        # 取pooling层的数据流作为特征
        feature_layer = sess.graph.get_tensor_by_name('pool_3:0')

        for i in range(n_batches):
            logging.info('Handling batch {}'.format(i))
            name_batch = img_names[i * batch_size:(i + 1) * batch_size]
            feature_batch = list()

            for img_name in name_batch:
                img_path = os.path.join(img_dir, img_name)
                img = gfile.FastGFile(img_path, 'rb').read()
                img_feature = sess.run(feature_layer,
                                       feed_dict={"DecodeJpeg/contents:0": img})
                feature_batch.append(img_feature)

            feature_batch = np.vstack(feature_batch)
            feature_path = os.path.join(feature_dir, 'img_feature-{}.pickle'.format(i))

            logging.info('Writing file {}'.format(i))
            with gfile.GFile(feature_path, 'w') as fd:
                pickle.dump((name_batch, feature_batch), fd)
