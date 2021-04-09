# -*- coding: UTF-8 -*-

from config import Config

from process_pipelines.PreProcess import process_source_file
from process_pipelines.Word2Vec import train_word2vec_tf
from process_pipelines.Word2Vec_gensim import train_word2vec_gensim
from process_pipelines.Train_tf import train_model_tf
from process_pipelines.Train_gensim import train_model_gensim
from process_pipelines.Chat_tf import chat_tf
from process_pipelines.Chat_gensim import chat_gensim
from process_pipelines.Delete import remove_file

import os

use_gensim = False # 是否使用gensim

if __name__ == '__main__':

    source_name = Config.Source_Name.Mem
    embedding_size = 50 # 每个词的词向量(嵌入)维度
    skip_window = 2 # 左右两边各取两个词。
    min_count = 5 # 设置最低频率，默认是5，如果一个词语在文档中出现的次数小于5，那么就会丢弃； 

    len_limit = 5 # 每句话取得多少词
    line_limit = 10000 # 限制最多学习问答条数
    epoch_time = 20 # 学习次数

    if source_name == Config.Source_Name.Mem: remove_file(use_gensim=use_gensim, source_name=source_name, qasplit=False, word2vec=True, train_tmp=True, train_model=True, embedding_size=embedding_size)
    if source_name == Config.Source_Name.Douban_multi: remove_file(use_gensim=use_gensim, source_name=source_name, qasplit=False, word2vec=False, train_tmp=True, train_model=True, embedding_size=embedding_size)
    if source_name == Config.Source_Name.Xiaohuangji: remove_file(use_gensim=use_gensim, source_name=source_name, qasplit=False, word2vec=True, train_tmp=True, train_model=True, embedding_size=embedding_size)
    # if source_name == Config.Source_Name.Douban_multi: remove_file(use_gensim=use_gensim, source_name=source_name, qasplit=True, word2vec=True, train_tmp=True, train_model=True)

    if not os.path.exists(Config.get_path(source_name, Config.File_Kind.Answer)):
        print("不存在"+Config.get_path(source_name, Config.File_Kind.Answer))
        process_source_file(source_name) # 文本预处理（切分问答对，切词）
    else: print("存在"+Config.get_path(source_name, Config.File_Kind.Answer))

    if use_gensim:
        if not os.path.exists(Config.get_path(source_name, Config.File_Kind.Gen_VecB, embedding_size)):
            train_word2vec_gensim(source_name, embedding_size, skip_window, min_count) # 训练词向量(gensim)
        if not os.path.exists(Config.get_path(source_name, Config.File_Kind.Model)):
            train_model_gensim(source_name, line_limit, len_limit, epoch_time, embedding_size)# 使用tf训练的词向量训练模型(gensim)
        chat_gensim(source_name, len_limit, embedding_size, epoch_time) # 使用gensim词向量的模型 回答

    else:
        if not os.path.exists(Config.path_word_vector):
            train_word2vec_tf(source_name, embedding_size, skip_window) # 训练词典和文本字符串(tf)
        train_model_tf(source_name, line_limit, len_limit, epoch_time, embedding_size) # 使用tf训练的词向量训练模型(tf)
        chat_tf(source_name, embedding_size, epoch_time) # 使用tf训练的词向量的模型 回答
        # pass