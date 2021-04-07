# -*- coding: UTF-8 -*-

from config import Config
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
import jieba
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import multiprocessing


def import_word_vector():
    '''
    读取词向量并返回字典形式
    '''
    dic = {}
    with open(Config.path_word_vector, 'r') as f:
        for line in f:
            tmp_line_split_list = line.split()
            dic[tmp_line_split_list[0]] = np.array([tmp_line_split_list[i] for i in range(1, len(tmp_line_split_list))], dtype=np.float32)
    return dic

# %%
def line2wordvec(list_line, word_vector_dict, sentend):
    """
    处理传入的切割好的一行的list
    长度超过8则删除后边的内容，不足则使用sentend补全
    最后处理成为词向量
    """
    len_limit = 8 # 每句话取得多少个词语

    len_list_line = len(list_line)
    if len_list_line > len_limit:
        list_line[len_limit:] = []
        list_line.append(sentend)
    else:
        for i in range(len_limit + 1 - len_list_line):
            list_line.append(sentend)
    for i in range(len_list_line if len_list_line <= len_limit else len_limit):
        list_line[i] = word_vector_dict[list_line[i]] if list_line[i] in word_vector_dict else word_vector_dict["UNK"] # 将前14个word其转换为vec
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
def get_answer(input_str, word_vector_dict, chat_model, sentend):
    qusetion_list = input2list(input_str)
    question_vec = np.array([line2wordvec(qusetion_list, word_vector_dict, sentend)], dtype=np.float32)
    predictions = chat_model.predict(question_vec)
    return predictions
    # answer_list = 

# %%

def compute_similarity(answer_array_list):
    '''
    循环计算每个向量和词典中向量的相似度，找出相似度最接近的项目
    然后返回拼接好的答案
    '''
    index = 1
    result = [0] * 9
    # print(len(result))
    def compute_every_word(i, index):
        print("Word no.", str(index), "starting computing.")
        max_similarity = 0
        tmp_word = '<!Individual> '
        for key, value in word_vector_dict.items():
            # similarity = cosine_similarity([i], [value]) # 余弦相似度  速度非常慢
            # similarity = np.abs(cosine_similarity([i], [value])) # 余弦相似度绝对值  速度非常慢
            similarity = np.abs(np.sqrt(np.sum(np.square(i, value)))) # 欧氏距离 速度快 结果奇怪UNK
            # similarity = cosine(i, value) # 余弦距离 # 速度较慢
            # similarity = pearsonr(i, value)[0] # 皮尔森相关系数 # 速度较慢
            if similarity >= max_similarity:
                max_similarity = similarity
                tmp_word = key
        print("similarity is: ", max_similarity)
        # print(str(index - 1), len(result))
        result[index - 1] = tmp_word
        print("Word no.", str(index), "finished computing. Is: ", tmp_word, "\n")

    for i in answer_array_list:
        compute_every_word(i, index)
        index += 1
        

    
    return result


# %%
def chat_tf():
    word_vector_dict = import_word_vector() # 词向量字典
    chat_model = load_model(Config.path_train_model) # 加载训练好的模型
    word_dim = len(word_vector_dict['说']) # 计算词向量维度
    print("词向量维度为（chat_tf）：", word_dim)
    sentend = np.ones((word_dim,), dtype = np.float32) # 设置结束词

    while(1):
        question = input("请输入：")
        answer = get_answer(question, word_vector_dict, chat_model, sentend)
        print(answer.shape)
        result = compute_similarity(answer[0])
        print(result)