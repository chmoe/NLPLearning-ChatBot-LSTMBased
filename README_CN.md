# NLPLearning-ChatBot-LSTMBased
 这个仓库存放着基于LSTM的聊天机器人代码。

i18n: 简体中文 | [English](/README.md)

# 文件目录树
下面表示了仓库中的目录树
```
% tree -N
.
├── LICENSE
├── README.md
├── README_CN.md
├── config.py
├── data
│   ├── clean_chat_corpus
│   │   └── 在这里存放语料库.txt
│   └── others
│       └── cn_stopwords.txt
├── main.py
├── model
│   ├── result
│   │   └── 在这里会生成训练好的模型.txt
│   └── tmp
│       └── 这里会存放训练过程中的中间文件.txt
├── process_pipelines
│   ├── Chat_gensim.py
│   ├── Chat_tf.py
│   ├── PreProcess.py
│   ├── PreProcess.pyc
│   ├── Train_gensim.py
│   ├── Train_tf.py
│   ├── Word2Vec.py
│   ├── Word2Vec_gensim.py
│   └── __init__.py
└── tmp
    └── 这里会存放生成的部分文件.txt
```

# 介绍

本仓库使用的数据集来源于这个仓库: [codemayq/chinese_chatbot_corpus](https://github.com/codemayq/chinese_chatbot_corpus).

为使用本仓库的代码，你需要先克隆上方的仓库，按照链接中的指示去执行相应的文件，最后可以得到一个叫做`clean_chat_corpus`的文件夹。

然后克隆本仓库，将上述文件夹拷贝或移动到目录 `/data/clean_chat_corpus`下。然后可以在cmd或terminal中执行命令 `python main.py`来执行本仓库的代码。

如果你需要使用不同的数据及或者训练方法以及其他参数，可以在文件`main.py`中更改。

由于github的限制，在这里使用了数个txt文档作为必要文件夹的占位，你可以删除或忽视他们。

# 关于

这个仓库仅用于我的毕业设计和毕业论文。因此无法保证全部代码和数据的准确性。

此仓库所有者保留一切随时变更内容的权利，恕不另行通知。

