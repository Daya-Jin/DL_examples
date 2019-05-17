# Embedding Based Movie RecSys

[博客地址](http://localhost:4000/2019/04/10/EmbBasedMovieRec/)

实现一个使用embedding方法并结合TextCNN的电影推荐算法框架，依赖环境为：

- ```Python 3.5.2```
- ```tensorflow-gpu 1.13.1```
- ```numpy 1.16.3```

所用数据位于```DL_for_learner/dataset/movielens/ml-1m/```下，三个文件：

- ```users.dat```
- ```movies.dat```
- ```ratings.dat```

所用到的```py```文件和模块有：

- ```DL_for_learner/dataset/movielens/preprocess.py```，对原始文件做预处理，并将处理好的数据保存成```*.npy```
- ```DL_for_learner/dataset/dataset.py```，提供数据封装与数据载入
