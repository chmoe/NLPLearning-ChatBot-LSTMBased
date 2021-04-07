# -*- coding: UTF-8 -*-
#%%
import os
from enum import Enum
#%%

#%%
class Config(object):
    Source_Name = Enum('Name', ('Douban_single', 'Douban_multi', 'Qingyun', 'Tieba', 'Weibo', 'Xiaohuangji'))
    File_Kind = Enum('Name', ('Question', 'Answer', 'Que_Ans', "Que_Vec", "Ans_Vec", "Gen_Mod", "Gen_Vec", "Gen_VecB"))
    encoding = 'utf-8'

    # 各种路径
    path_data_finder = "./data/" # 数据用路径
    path_source_finder = './data/clean_chat_corpus/'
    path_data_others_finder = "./data/others/"
    path_tmp_finder = "./tmp/" # 临时文件用路径
    path_model_tmp_finder = "./model/tmp/" # 训练模型的中间变量路径
    path_model_finder = "./model/result/" # 训练好的模型的最终路径

    # 各种文件名
    name_train_file_question = "_question.txt" # 用于训练的问题文件（切割好
    name_train_file_answer = "_answer.txt" # 用于训练的答案文件（切割好
    name_train_file_que_ans = "_que_ans.txt" # 合并问题和答案的文件（切割好(gensim)
    name_word2vec_model = "_word2vec_gensim.model" # 训练好的词向量模型(gensim)
    name_word2vec_vector = "_word2vec_gensim.vector" # 得到的词向量(gensim)
    name_word2vec_vector_bin = "_word2vec_gensim.bin" # 得到的词向量（二进制(gensim)
    name_file_word_vector = "word_vector.txt" # 训练出的词向量(tf)
    name_train_file_question_vec = "_question_vec.npy" # 训练好的问题集词向量(tf)
    name_train_file_answer_vec = "_answer_vec.npy" # 训练好的答案集词向量(tf)
    name_word_list_file = "word_dict.npy" # 保存有词语和每个词语出现的次数(tf)
    name_train_file_train_model = "model.h5" # 训练好的模型
    name_stopword_file = "cn_stopwords.txt" # 停用词表

    # 组合好的路径
    path_word_vector = path_data_finder + name_file_word_vector # 词向量(txt)
    path_train_model = path_model_finder + name_train_file_train_model # 训练好的模型(h5)
    path_stopword = path_data_others_finder + name_stopword_file
    path_word_list_file = path_tmp_finder + name_word_list_file

    
    @staticmethod
    def get_path_with_kind(kind_name):
        if kind_name == Config.Source_Name.Douban_single:
            return "douban_single"
        elif kind_name == Config.Source_Name.Douban_multi:
            return "douban_multi"
        elif kind_name == Config.Source_Name.Qingyun:
            return "qingyun"
        elif kind_name == Config.Source_Name.Tieba:
            return "tieba"
        elif kind_name == Config.Source_Name.Weibo:
            return "weibo"
        elif kind_name == Config.Source_Name.Xiaohuangji:
            return "xiaohuangji"
        else: return "NULL"
    @staticmethod
    def get_path(kind_name, file_kind):
        '''
        获取不同路径
        '''
        name = Config.get_path_with_kind(kind_name)
        if file_kind == Config.File_Kind.Question:
            return os.path.join(Config.path_tmp_finder, name + Config.name_train_file_question)
        elif file_kind == Config.File_Kind.Answer:
            return os.path.join(Config.path_tmp_finder, name + Config.name_train_file_answer)
        elif file_kind == Config.File_Kind.Que_Vec:
            return os.path.join(Config.path_data_finder, name + Config.name_train_file_question_vec)
        elif file_kind == Config.File_Kind.Ans_Vec:
            return os.path.join(Config.path_data_finder, name + Config.name_train_file_answer_vec)
        elif file_kind == Config.File_Kind.Que_Ans:
            return os.path.join(Config.path_tmp_finder, name + Config.name_train_file_que_ans)
        elif file_kind == Config.File_Kind.Gen_Mod:
            return os.path.join(Config.path_tmp_finder, name + Config.name_word2vec_model)
        elif file_kind == Config.File_Kind.Gen_Vec:
            return os.path.join(Config.path_tmp_finder, name + Config.name_word2vec_vector)
        elif file_kind == Config.File_Kind.Gen_VecB:
            return os.path.join(Config.path_tmp_finder, name + Config.name_word2vec_vector_bin)
    @staticmethod
    def get_sourse_path(kind_name):
        '''
        根据名称获得训练内容的路径
        '''
        if kind_name == Config.Source_Name.Douban_single:
            return os.path.join(Config.path_source_finder, "douban_single_turn.tsv")
        elif kind_name == Config.Source_Name.Douban_multi:
            return os.path.join(Config.path_source_finder, "douban.tsv")
        elif kind_name == Config.Source_Name.Qingyun:
            return os.path.join(Config.path_source_finder, "qingyun.tsv")
        elif kind_name == Config.Source_Name.Tieba:
            return os.path.join(Config.path_source_finder, "tieba.tsv")
        elif kind_name == Config.Source_Name.Weibo:
            return os.path.join(Config.path_source_finder, "weibo.tsv")
        elif kind_name == Config.Source_Name.Xiaohuangji:
            return os.path.join(Config.path_source_finder, "xiaohuangji.tsv")
        else: return "NULL"
#%%
# Config.get_sourse_path(Config.Source_Name.Tieba)
# # %%
# Config.get_path(Config.Source_Name.Douban_single, Config.File_Kind.Question)
# %%

# %%
