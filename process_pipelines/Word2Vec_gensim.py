# -*- coding: UTF-8 -*-

# 中文参考资料：https://zhuanlan.zhihu.com/p/40016964

from config import Config

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence # 将文本文件转换为可迭代的对象
import multiprocessing
import os



def combine_question_answer(source_name):
    combine_path = Config.get_path(source_name, Config.File_Kind.Que_Ans)
    if not os.path.exists(combine_path):
        print("正在合并问题和答案集合...")
        print("创建合并后文件：" + combine_path)
        f = open(combine_path, 'w')
        print("正在写入问题集合...")
        for line in open(Config.get_path(source_name, Config.File_Kind.Question)):
            f.writelines(line)
        print('问题集合写入完成')
        f.write('\n')
        print("正在写入答案集合...")
        for line in open(Config.get_path(source_name, Config.File_Kind.Answer)):
            f.writelines(line)
        print('答案集合写入完成')
        f.write('\n')
        f.close()
        print("合并文件 " + combine_path + "写入完成")
    else:
        print("合并文件已经存在，即将推出...")
    return combine_path

def train_word2vec_gensim(source_name, embedding_size, skip_window, min_count):
    combine_path = combine_question_answer(source_name)
    if not os.path.exists(Config.get_path(source_name, Config.File_Kind.Gen_Mod, embedding_size)):
        print("即将开始训练(gensim)")
        model = Word2Vec(LineSentence(combine_path), vector_size=embedding_size, window=skip_window, min_count=min_count, workers=multiprocessing.cpu_count())
        print("词向量训练完成(gensim)")
        print("正在保存模型 ")
        model.save(Config.get_path(source_name, Config.File_Kind.Gen_Mod, embedding_size))
        print("模型保存完成" + Config.get_path(source_name, Config.File_Kind.Gen_Mod, embedding_size))
    else:
        print("模型已经存在，将直接加载模型")
        model = Word2Vec.load(Config.get_path(source_name, Config.File_Kind.Gen_Mod, embedding_size))
    print("正在保存词向量")
    model.wv.save_word2vec_format(Config.get_path(source_name, Config.File_Kind.Gen_Vec, embedding_size), binary=False) # 非二进制
    print("词向量保存完成：", (Config.get_path(source_name, Config.File_Kind.Gen_Vec, embedding_size)))
    print("正在保存词向量(bin)")
    model.wv.save_word2vec_format(Config.get_path(source_name, Config.File_Kind.Gen_VecB, embedding_size), binary=True)
    print("词向量保存完成：", (Config.get_path(source_name, Config.File_Kind.Gen_VecB, embedding_size)))

'''
报错解决：AttributeError: ‘Word2VecKeyedVectors‘ object has no attribute ‘save_Word2Vec_format‘
链接：https://www.secn.net/article/1633666.html
'''