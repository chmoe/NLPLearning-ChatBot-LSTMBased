# -*- coding: UTF-8 -*-

from config import Config

import numpy as np
from tensorflow.keras.models import load_model # 加载训练模型
import jieba
import gensim.models


def load_word_dic(source_name, embedding_size):
    model = gensim.models.KeyedVectors.load_word2vec_format(Config.get_path(source_name, Config.File_Kind.Gen_VecB, embedding_size), binary=True)
    return model

def line2wordvec(list_line, model, sentend, len_limit):
    """
    处理传入的切割好的一行的list
    长度超过8则删除后边的内容，不足则使用sentend补全
    最后处理成为词向量
    """
    # len_limit = 8 # 每句话取得多少个词语

    len_list_line = len(list_line)
    if len_list_line > len_limit:
        list_line[len_limit:] = []
        list_line.append(sentend)
    else:
        for i in range(len_limit + 1 - len_list_line):
            list_line.append(sentend)
    for i in range(len_list_line if len_list_line <= len_limit else len_limit):
        list_line[i] = model[list_line[i]] if list_line[i] in model else sentend # 将前14个word其转换为vec
    return list_line
def input2list(input_str):
    """
    针对输入的内容进行文本预处理
    """
    question = input_str.strip()
    print(question)
    question_list = list(jieba.cut(question))
    print(question_list)
    # que_list_split_with_space = " ".join(question_list)
    return question_list
def get_answer(input_str, model, sentend, chat_model, len_limit):
    qusetion_list = input2list(input_str)
    question_vec = np.array([line2wordvec(qusetion_list, model, sentend, len_limit)], dtype=np.float32)
    predictions = chat_model.predict(question_vec)
    return predictions
    # answer_list = 


def chat_gensim(source_name, len_limit, embedding_size, epoch_time):
    chat_model = load_model(Config.get_path(source_name, Config.File_Kind.Model, embedding_size, 'gensim', epoch_time)) # 加载训练好的模型
    model = load_word_dic(source_name, embedding_size)
    word_dim = len(model['你']) # 计算词向量维度
    print("词向量维度为：", word_dim)
    sentend = np.ones((word_dim,), dtype = np.float32) # 设置结束词

    while(1):
        input_str = input("请输入：")
        answer = get_answer(input_str, model, sentend, chat_model, len_limit)
        print(answer.shape)
        result = [model.most_similar([answer[0][i]])[0][0] for i in range(answer.shape[1])]
        print(result)

#%%

# %%
# import gensim.models
# model = gensim.models.KeyedVectors.load_word2vec_format('../tmp/mem_word2vec_gensim.bin', binary=True)
# model['你']
# # %%
# model.similar_by_word('你',topn=10) # 查看相似度前10的词语
# # %%

# # %%
# '猹' in model
# %%
