# -*- coding: UTF-8 -*-

# 删除指定内容的文件，重新进行训练等

from config import Config
import os

def del_path_list(path_list):
    for item in path_list:
        print("正在查找：" ,item)
        if os.path.isdir(item):
            print("发现文件目录", item)
            del_list = os.listdir(item)
            for f in del_list:
                file_path = os.path.join(item, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print("delete: ", file_path)
        elif os.path.exists(item):
            print("删除：",item)
            os.remove(item)
        else:
            print("未找到", item, "即将跳过...")

def remove_file(use_gensim, source_name, qasplit, word2vec, train_tmp, train_model):
    delete_file_list = []
    if qasplit:
        delete_file_list.append(Config.path_tmp_finder + Config.name_question_list)
        delete_file_list.append(Config.path_tmp_finder + Config.name_answer_list)
        delete_file_list.append(Config.get_path(source_name, Config.File_Kind.Question))
        delete_file_list.append(Config.get_path(source_name, Config.File_Kind.Answer))
        delete_file_list.append(Config.get_path(source_name, Config.File_Kind.Que_Ans))
    if use_gensim:
        if word2vec:
            delete_file_list.append(Config.get_path(source_name, Config.File_Kind.Gen_Mod))
            delete_file_list.append(Config.get_path(source_name, Config.File_Kind.Gen_Vec))
            delete_file_list.append(Config.get_path(source_name, Config.File_Kind.Gen_VecB))
    else:
        if word2vec:
            delete_file_list.append(Config.path_word_list_file)
            delete_file_list.append(Config.path_word_vector)
        if train_tmp:
            delete_file_list.append(Config.get_path(source_name, Config.File_Kind.Que_Vec))
            delete_file_list.append(Config.get_path(source_name, Config.File_Kind.Ans_Vec))
    if train_tmp:
        delete_file_list.append(Config.path_model_tmp_finder + Config.get_path_with_kind(source_name)+"/")
    if train_model:
        delete_file_list.append(Config.get_path(source_name, Config.File_Kind.Model))
    del_path_list(delete_file_list)