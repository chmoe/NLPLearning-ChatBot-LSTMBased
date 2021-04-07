# -*- coding: UTF-8 -*-

from config import Config

from process_pipelines.PreProcess import process_source_file
from process_pipelines.Word2Vec import train_word2vec_tf
from process_pipelines.Word2Vec_gensim import train_word2vec_gensim
from process_pipelines.Train_tf import train_model_tf
from process_pipelines.Train_gensim import train_model_gensim
from process_pipelines.Chat_tf import chat_tf
from process_pipelines.Chat_gensim import chat_gensim

import os

use_gensim = True

if __name__ == '__main__':

    source_name = Config.Source_Name.Douban_multi
    embedding_size = 100 # 每个词的词向量(嵌入)维度
    skip_window = 2 # 左右两边各取两个词。
    min_count = 5 # 设置最低频率，默认是5，如果一个词语在文档中出现的次数小于5，那么就会丢弃； 

    len_limit = 8 # 每句话取得多少词
    line_limit = 1000000 # 限制最多学习问答条数
    epoch_time = 100 # 学习次数

    if not os.path.exists(Config.get_path(source_name, Config.File_Kind.Answer)):
        process_source_file(source_name) # 文本预处理（切分问答对，切词）

    if use_gensim:
        if not os.path.exists(Config.get_path(source_name, Config.File_Kind.Gen_VecB)):
            train_word2vec_gensim(source_name, embedding_size, skip_window, min_count) # 训练词向量(gensim)
        train_model_gensim(source_name, line_limit, len_limit, epoch_time)# 使用tf训练的词向量训练模型(gensim)
        chat_gensim(source_name) # 使用gensim词向量的模型 回答

    else:
        if not os.path.exists(Config.path_word_vector):
            train_word2vec_tf(source_name, embedding_size, skip_window) # 训练词典和文本字符串(tf)
        train_model_tf(source_name) # 使用tf训练的词向量训练模型(tf)
        chat_tf(source_name) # 使用tf训练的词向量的模型 回答
        # pass