# -*- coding: UTF-8 -*-

from config import Config

import pandas as pd
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector, Dense, TimeDistributed




def import_word_vector():
    '''
    读取词向量并返回字典形式
    '''
    dic = {}
    with open(Config.path_word_vector, 'r', encoding='utf-8') as f:
        for line in f:
            tmp_line_split_list = line.split()
            dic[tmp_line_split_list[0]] = np.array([tmp_line_split_list[i] for i in range(1, len(tmp_line_split_list))], dtype=np.float32)
    return dic

# %%
def line2wordvec(list_line, word_dim, sentend, word_vector_dict, len_limit):
    """
    处理传入的切割好的一行的list
    长度超过14则删除后边的内容，不足则使用sentend补全
    最后处理成为词向量
    """

    len_list_line = len(list_line)
    if len_list_line > len_limit:
        list_line[len_limit:] = []
        list_line.append(sentend)
    else:
        for i in range(len_limit + 1 - len_list_line):
            list_line.append(sentend)
    for i in range(len_list_line if len_list_line <= len_limit else len_limit):
        list_line[i] = word_vector_dict[list_line[i]] if list_line[i] in word_vector_dict else word_vector_dict['UNK']  # 将前14个word其转换为vec
    return list_line
# %%
def qaword2vec(source_name, line_limit, len_limit):
    """
    读取问题和答案的文件
    转换为向量形式进行保存
    返回转换后的list
    """

    word_vector_dict = import_word_vector()
    print("构建字典完毕")

    word_dim = len(word_vector_dict['说']) # 计算词向量维度
    print("词向量维度为：", word_dim)
    sentend = np.ones((word_dim,), dtype = np.float32) # 设置结束词

    # python保存和读取list： https://blog.csdn.net/HHTNAN/article/details/93488969
    qlist = []
    index = 0
    with open(Config.get_path(source_name, Config.File_Kind.Question), 'r', encoding='utf-8') as f:
        for line in f:
            index += 1
            if index >= line_limit: break # 限制长度为100w
            tmp_line_split_list = line.split()
            processeed_line_wordvec_list = line2wordvec(tmp_line_split_list, word_dim, sentend, word_vector_dict, len_limit) # 按照14个限定处理好之后的wordvec的list
            qlist.append(processeed_line_wordvec_list)
    np.save(Config.get_path(source_name, Config.File_Kind.Que_Vec), qlist)

    alist = []
    index = 0
    with open(Config.get_path(source_name, Config.File_Kind.Answer), 'r', encoding='utf-8') as f:
        for line in f:
            index += 1
            if index >= line_limit: break # 限制长度为100w
            tmp_line_split_list = line.split()
            processeed_line_wordvec_list = line2wordvec(tmp_line_split_list, word_dim, sentend, word_vector_dict, len_limit) # 按照14个限定处理好之后的wordvec的list
            alist.append(processeed_line_wordvec_list)
    np.save(Config.get_path(source_name, Config.File_Kind.Ans_Vec), alist)

    return qlist, alist



# %%
def seq2seq(source_name, X_vector, Y_vector, epoch_time, embedding_size):
    # 将 X_vector、Y_vector 转化为数组形式
    X_vector = np.array(X_vector, dtype=np.float32)
    Y_vector = np.array(Y_vector, dtype=np.float32)

    

    # 手动切分数据为：训练集、测试集
    pos = int(0.8 * X_vector.shape[0])
    # pos = 800000
    X_train, X_test = X_vector[:pos], X_vector[pos:]
    Y_train, Y_test = Y_vector[:pos], Y_vector[pos:]

    timesteps = X_train.shape[1]
    word_dim = X_train.shape[2]
    print(X_train.shape)
    print(timesteps)
    print(word_dim)
    print(X_train.shape[1:])
    print(type(X_train))
    print(X_train.shape)

    # 构建一个空容器
    model = Sequential()

    # TF2.x使用LSTM：https://blog.csdn.net/qq_41094332/article/details/105670613

    # 编码
    # 报错：__init__() missing 1 required positional argument: 'units'
    # 解决：https://stackoverflow.com/questions/56106546/typeerror-init-missing-1-required-positional-argument-units
    # model.add(LSTM(units = 200, output_dim=word_dim, input_shape=X_train.shape[1:], return_sequences=False))
    # 报错：after upgrade to alpha 2 version i get wrong results from LSTM module.
    # 解决：https://github.com/apple/tensorflow_macos/issues/157#issuecomment-792416813
    model.add(LSTM(word_dim, input_shape=X_train.shape[1:], return_sequences=False))

    # 将问句含义进行复制
    model.add(RepeatVector(timesteps))

    # 解码
    model.add(LSTM(word_dim, return_sequences=True))

    # 添加全连接层
    model.add(TimeDistributed(Dense(word_dim, activation="linear")))

    # 编译模型
    model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

    

    checkpoint_path = Config.path_model_tmp_finder + Config.get_path_with_kind(source_name) + "/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # 创建一个保存模型权重的回调
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1, period=5)

    if os.path.exists(Config.path_model_tmp_finder + Config.get_path_with_kind(source_name) + '/checkpoint'):
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest)
        tmp_len = len(Config.path_model_tmp_finder + Config.get_path_with_kind(source_name)) + 4 # 路径长度
        start_epoch_value = int(latest[tmp_len:tmp_len + 4])
        model.fit(X_train, Y_train, epochs=epoch_time, validation_data=(X_test, Y_test), callbacks=[cp_callback], initial_epoch=start_epoch_value)
    else:

        # 使用 `checkpoint_path` 格式保存权重
        model.save_weights(checkpoint_path.format(epoch=0))
        
        # 训练、保存模型
        # 报错：TypeError: fit() got an unexpected keyword argument 'nb_epoch'
        # 参考内容：https://github.com/keras-team/keras/issues/14135#issuecomment-649105439
        # model.fit(X_train, Y_train, epochs=5000, validation_data=(X_test, Y_test), callbacks=[cp_callback])
        model.fit(X_train, Y_train, epochs=epoch_time, callbacks=[cp_callback])
    model.save(Config.get_path(source_name, Config.File_Kind.Model, embedding_size, 'tf', epoch_time))

    return model


# %%
def train_model_tf(source_name, line_limit, len_limit, epoch_time, embedding_size):
    qlist, alist = qaword2vec(source_name, line_limit, len_limit)
    print("问答list成功向量化")
    model = seq2seq(source_name, qlist, alist, epoch_time, embedding_size)
# %%
