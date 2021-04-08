# -*- coding: UTF-8 -*-
# 将给定的输入文件转换成分别的输入和输出
# 处理输入和输出，分别进行分词
# %%
from config import Config
import pandas as pd
import jieba
# %%
print("加载了preprocess")
def split_word_from_file(file_path, file_path1):
    '''
    使用jieba分词对问题和答案集合进行分词
    '''
    print("正在对" + file_path + "进行分词")
    inputFile_NoSegment = open(file_path, 'r')
    outputFile_Segment = open(file_path1, 'w', encoding='utf-8')

    # 读取语料文本中的每一行文字
    lines = inputFile_NoSegment.readlines()

    # 为每一行文字分词
    for i in range(len(lines)):
        line = lines[i]
        if line:
            line = line.strip()
            seg_list = jieba.cut(line)

            segments = ''
            for word in seg_list:
                segments = segments + ' ' + word
            segments += '\n'
            segments = segments.lstrip()

            # 将分词后的语句，写进文件中
            outputFile_Segment.write(segments)

    inputFile_NoSegment.close()
    outputFile_Segment.close()
    print("已经成功写入文件：", file_path1)
# %%
def process_source_file(source_name):
    '''将指定的源文件切分成为问题和答案的两个文件'''
    # print(Config.get_sourse_path(source_name))
    # 遇到错误的行跳过：https://www.huaweicloud.com/articles/76bbd9a2d0bfa97d499fe8290816936a.html
    train_df = pd.read_csv(Config.get_sourse_path(source_name), sep = '\t', header = None, error_bad_lines=False)
    train_df = train_df.astype(str)
    # %% 将答案和问题分别保存到两个文件中
    # series转txt参考：https://zhuanlan.zhihu.com/p/32672042
    train_df[0].to_csv(Config.path_tmp_finder + Config.name_question_list, header = None, index=False)
    train_df[1].to_csv(Config.path_tmp_finder + Config.name_answer_list, header = None, index=False)
    split_word_from_file(Config.path_tmp_finder + Config.name_question_list, Config.get_path(source_name, Config.File_Kind.Question))
    split_word_from_file(Config.path_tmp_finder + Config.name_answer_list, Config.get_path(source_name, Config.File_Kind.Answer))