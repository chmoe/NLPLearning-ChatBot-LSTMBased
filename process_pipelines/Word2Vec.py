# -*- coding: UTF-8 -*-
# 训练词向量
# 停用词库来源：https://github.com/goto456/stopwords

# tensorflow官方样例：https://github.com/tensorflow/tensorflow/blob/r2.3/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
# 相关中文解释：https://www.cnblogs.com/Luv-GEM/p/10727432.html

from config import Config
import tensorflow as tf
import numpy as np
import collections
import random
import math
from six.moves import xrange

# 字典存取参考：https://blog.csdn.net/yangtf07/article/details/81571371
def dict_create_or_add(dic_name, key):
    '''计算词语在词典中出现的次数'''
    if key in dic_name:
        dic_name[key] += 1
    else: dic_name[key] = 1

def create_word_dict(source_name):
    '''处理并生成一个保存有单词出现次数的dict，同时保存成文件'''
    word_dict_dict = {"UNK":0} # UNK代表未知词语

    stop_list = [i.strip() for i in open(Config.path_stopword, 'r', encoding='utf-8')] # 导入停用词表

    print("处理问题list...")
    with open(Config.get_path(source_name, Config.File_Kind.Question), 'r') as f:
        for line in f:
            for word in line.split():
                if word not in stop_list:
                    dict_create_or_add(word_dict_dict, word)
    print("问题list...处理完成")

    print("处理答案list...")
    with open(Config.get_path(source_name, Config.File_Kind.Answer), 'r') as f:
        for line in f:
            for word in line.split():
                if word not in stop_list:
                    dict_create_or_add(word_dict_dict, word)
    print("答案list...处理完成")
        
    print("正在将词语表写入文件...")
    np.save(Config.path_word_list_file, word_dict_dict)
    print("文件" + Config.path_word_list_file + "写入完成")
    return word_dict_dict

def read_word_list_dict(word_dict_dict):
    '''
    读取包含次数的字典，并排序生成新的字典
    '''
    # 读取word_list并构建词典， value对应出现次数
    word_dict_dict = np.load(Config.path_word_list_file, allow_pickle=True).item() # 读取文件做为字典
    # 字典排序参考：https://blog.csdn.net/tangtanghao511/article/details/47810729
    word_dict_dict_sorted = sorted(word_dict_dict.items(), key = lambda x: x[1],reverse=True) # 为字典排序
    # 按照索引构建词典
    word_index_dict = {"UNK":0} # 词典排序后, 对应每个词的索引，如
    # {'UNK': 0, '中': 1, '月': 2, '年': 3, '说': 4, '中国': 5,...}
    index = 1
    for word in word_dict_dict_sorted:
        word_index_dict[word[0]] = index
        index += 1
    # 字典翻转，用于根据索引取词
    word_index_dict["UNK"] = 0
    word_index_dict_reverse = dict(zip(word_index_dict.values(), word_index_dict.keys()))
    data = [] # 最初的词典所对应的按照词频排序后的词典中所对应的索引序号
    unk_count = 0
    for word in word_dict_dict.keys():
        if word in word_index_dict:
            index = word_index_dict[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    return data, word_index_dict_reverse


""" 第三步：为skip-gram模型生成训练的batch """
def generate_batch(batch_size, num_skips, skip_window, data_index, data):
    '''
    各种参数：
    - batch_size:
    - num_skips: 重复使用输入以生成标签的次数
    - skip_window: 左右个考虑多少个词语
    - data_index: 词典中第n个词，n为索引
    '''
    # 断言，如果不满足条件则报错
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)          # 存储开头词（中心词）的编号
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)      # 存储目标词（非中心词）的编号
    span = 2 * skip_window + 1  # [ 左边skip_window target 右边skip_window ]（一共取多少个词）
    buffer = collections.deque(maxlen=span)      # deque 是一个双向列表,限制最大长度为span， 可以从两端append和pop数据。
    
    for _ in range(span):  # 每次从原始词典中依次取出5个单词
        buffer.append(data[data_index]) # 取出词语列表（单词在修改后词典中的序号）加入到buffer中
        data_index = (data_index + 1) % len(data)      
        # 循环结束后得到buffer为 deque([26927, 936, 110, 449, 102], maxlen=5)，也就是取到了data的前五个值, 对应词语列表的前5个词。
        
    for i in range(batch_size // num_skips):      # 整除，重复使用输入2次以生成标签
        # i取值0,1，是表示一个batch能取两个中心词
        target = skip_window                # target值为2，意思是中心词在buffer这个列表中的位置是2。
        targets_to_avoid = [skip_window]    # 列表是用来存已经取过的词的索引，下次就不能再取了，从而把buffer中5个元素不重复的取完。
         
        for j in range(num_skips): # j取0，1，2，3，意思是在中心词周围取4个词。
            while target in targets_to_avoid: # 避免已经取过的数字（如果不是则退出循环
                target = random.randint(0, span - 1) # 2是中心词的位置，所以j的第一次循环要取到不是2的数字，也就是取到0，1，3，4其中的一个，才能跳出循环。
            batch[i * num_skips + j] = buffer[skip_window] # 取到中心词的索引。前四个元素都是同一个中心词的索引。
            labels[i * num_skips + j, 0] = buffer[target] # 取到中心词指向的词的索引。一共会取到上下各两个。
            targets_to_avoid.append(target) # 增加当前这一条到已经去过的地方
        buffer.append(data[data_index]) # 第一次循环结果为buffer：deque([512, 1023, 3977, 1710, 1413], maxlen=5)，
        # 所以明白了为什么限制为5，因为可以把第一个元素去掉。这也是为什么不用list。

        # 回溯以防止超过最大索引
        data_index = (data_index + 1) % len(data)
        
    return batch, labels, data_index




""" 第四步：定义和训练skip-gram模型"""
def train_word2vec_tf(source_name, embedding_size, skip_window):

    num_steps = 10

    data_index = 0

    batch_size = 128 # 上面那个数量为8的batch只是为了展示以下取样的结果，实际上是batch-size 是128。
    num_skips = 4 # 要取4个上下文词，同一个中心词也要重复取4次。
    num_sampled = 64 # 负采样的负样本数量为64

    data, word_index_dict_reverse = read_word_list_dict(create_word_dict(source_name))

    vocabulary_size = len(set(data))
    print("vocabulary_size: ", vocabulary_size)

    graph = tf.Graph()         

    with graph.as_default():                   
        #  把新生成的图作为整个 tensorflow 运行环境的默认图，详见第二部分的知识点。
        
        # Input data.
        with tf.name_scope('inputs'):
            train_inputs = tf.compat.v1.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, 1])
            # valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        print("train_labels",train_labels)
        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(tf.compat.v1.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)) #产生-1到1之间的均匀分布, 看作是初始化隐含层和输出层之间的词向量矩阵。
            embed = tf.nn.embedding_lookup(embeddings, train_inputs) # 用词的索引在词向量矩阵中得到对应的词向量。shape=(128, 300)

        print("embed",embed)
        # Construct the variables for the NCE loss
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(tf.compat.v1.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
            # 初始化损失（loss）函数的权重矩阵和偏置矩阵
            # 生成的值服从具有指定平均值和合理标准偏差的正态分布，如果生成的值大于平均值2个标准偏差则丢弃重新生成。这里是初始化权重矩阵。
            # 对标准方差进行了限制的原因是为了防止神经网络的参数过大。
        print("nce_weights", nce_weights)
        with tf.name_scope('biases'):
            # 初始化偏置矩阵，生成了一个vocabulary_size * 1大小的零矩阵。
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        print("nce_biases", nce_biases)
        with tf.name_scope('loss'):
            # 这个tf.nn.nce_loss函数把多分类问题变成了正样本和负样本的二分类问题。用的是逻辑回归的交叉熵损失函数来求，而不是softmax  。
            tf_nn_nce_los = tf.nn.nce_loss(
                    weights=nce_weights, 
                    biases=nce_biases,
                    labels=train_labels, 
                    inputs=embed, 
                    num_sampled=num_sampled, 
                    num_classes=vocabulary_size
            )
            print("tf_nn_nce_los", tf_nn_nce_los)
            loss = tf.reduce_mean(tf_nn_nce_los,0)
            print("loss",loss)
        # 将损失值作为标量添加到汇总中
        tf.summary.scalar('loss', loss)

        # 使用1.0的学习率构造SGD优化器
        with tf.name_scope('optimizer'):
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(1.0).minimize(loss) # Construct the SGD optimizer using a learning rate of 1.0.

        # 计算小批量样本和所有词嵌入的余弦相似度
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))       
        # shape=(196871, 1), 对词向量矩阵进行归一化
        normalized_embeddings = embeddings / norm

        # 增加初始化变量
        init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session(graph=graph) as session:
        
        init.run()
        print('初始化了.')
        
        average_loss = 0
    
        for step in xrange(num_steps):
            print(step)
            batch_inputs, batch_labels, data_index = generate_batch(batch_size, num_skips, skip_window, data_index, data)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            
            run_metadata = tf.compat.v1.RunMetadata()

            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict, run_metadata=run_metadata)

            average_loss += loss_val
            
            final_embeddings = normalized_embeddings.eval()
            print(final_embeddings)        
            print("*"*20)
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0
                
        final_embeddings = normalized_embeddings.eval()      
        # 训练得到最后的词向量矩阵。
        print(final_embeddings)
        fp=open(Config.path_word_vector,'w',encoding='utf8')
        for k,v in word_index_dict_reverse.items():
            t=tuple(final_embeddings[k])         
            s=''
            for i in t:
                i=str(i)
                s+=i+" "               
            fp.write(v+" "+s+"\n")  
                
            # s为'0.031514477 0.059997283 ...'  , 对于每一个词的词向量中的300个数字，用空格把他们连接成字符串。
            #把词向量写入文本文档中。不过这样就成了字符串，我之前试过用np保存为ndarray格式，这里是按源码的保存方式。

        fp.close()


