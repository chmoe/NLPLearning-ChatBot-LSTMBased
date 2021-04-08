# NLPLearning-ChatBot-LSTMBased
 This is a repository for LSTM based chatbot.

i18n: English | [简体中文](/README_CN.md)

# File Tree
The file tree of this repo as follow.
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

# Intro

This project was built with data from this repo: [codemayq/chinese_chatbot_corpus](https://github.com/codemayq/chinese_chatbot_corpus).

You should clone that repo first, and then, execute the python file. Then you will get a folder called `clean_chat_corpus`.

Clone this repo, move or copy that folder to the path `/data/clean_chat_corpus`. Then you can excute with `python main.py` in the terminal open in this root.

Change the variables on the file `main.py` if you need to use different datasets or train methods.

Cause of the rule on github, I use some txt files to take space for some empty folder, ignore those txt files is required.

# About

I create it just for my Graduation Project and thesis. I can not promise all the code is right.

I reserve the right to make adjustments without prior notification.
