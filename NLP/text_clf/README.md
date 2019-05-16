# Text Classification

所用数据位于```./dataset/news_CN/```下，三个文件：
- ```cnews.train.txt```
- ```cnews.val.txt```
- ```cnews.test.txt```

所用到的```py```文件和模块有：
- ```./dataset/news_CN/preprocess.py```，对原始文件做预处理，包括分词与构建词典
- ```./dataset/news_CN/utils.py```，提供类别编码器
- ```./NLP/vocab.py```，提供文本编码器
- ```./dataset/dataset.py```，提供数据封装与数据载入