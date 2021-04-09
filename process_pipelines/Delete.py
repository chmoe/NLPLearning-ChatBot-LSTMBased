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

def remove_file(use_gensim=True, source_name=False, qasplit=False, word2vec=False, train_tmp=False, train_model=False, embedding_size=50, epoch_time = 50, all_ = False):
    delete_file_list = []
    if all_:
        delete_file_list = [
            Config.path_tmp_finder + Config.name_question_list,
            Config.path_tmp_finder + Config.name_answer_list,
            Config.get_path(source_name, Config.File_Kind.Question),
            Config.get_path(source_name, Config.File_Kind.Answer),
            Config.get_path(source_name, Config.File_Kind.Que_Ans),
            Config.get_path(source_name, Config.File_Kind.Gen_Mod, embedding_size),
            Config.get_path(source_name, Config.File_Kind.Gen_Vec, embedding_size),
            Config.get_path(source_name, Config.File_Kind.Gen_VecB, embedding_size),
            Config.get_path(source_name, Config.File_Kind.Model, embedding_size, 'gensim'),
            Config.path_word_list_file,
            Config.path_word_vector,
            Config.get_path(source_name, Config.File_Kind.Que_Vec),
            Config.get_path(source_name, Config.File_Kind.Ans_Vec),
            Config.get_path(source_name, Config.File_Kind.Model, embedding_size, 'tf'),
            Config.path_model_tmp_finder + Config.get_path_with_kind(source_name)+"/",
        ]
    else:
        if qasplit:
            delete_file_list.append(Config.path_tmp_finder + Config.name_question_list)
            delete_file_list.append(Config.path_tmp_finder + Config.name_answer_list)
            delete_file_list.append(Config.get_path(source_name, Config.File_Kind.Question))
            delete_file_list.append(Config.get_path(source_name, Config.File_Kind.Answer))
            delete_file_list.append(Config.get_path(source_name, Config.File_Kind.Que_Ans))
        if use_gensim:
            if word2vec:
                delete_file_list.append(Config.get_path(source_name, Config.File_Kind.Gen_Mod, embedding_size))
                delete_file_list.append(Config.get_path(source_name, Config.File_Kind.Gen_Vec, embedding_size))
                delete_file_list.append(Config.get_path(source_name, Config.File_Kind.Gen_VecB, embedding_size))

            if train_model:
                delete_file_list.append(Config.get_path(source_name, Config.File_Kind.Model, embedding_size, 'gensim', epoch_time))
        else:
            if word2vec:
                delete_file_list.append(Config.path_word_list_file)
                delete_file_list.append(Config.path_word_vector)
            if train_tmp:
                delete_file_list.append(Config.get_path(source_name, Config.File_Kind.Que_Vec))
                delete_file_list.append(Config.get_path(source_name, Config.File_Kind.Ans_Vec))
            if train_model:
                delete_file_list.append(Config.get_path(source_name, Config.File_Kind.Model, embedding_size, 'tf', epoch_time))
        if train_tmp:
            delete_file_list.append(Config.path_model_tmp_finder + Config.get_path_with_kind(source_name)+"/")
    del_path_list(delete_file_list)