#!/usr/bin/env python
# encoding: utf-8

"""
@description: 预测模型

@author: baoqiang
@time: 2019/3/6 下午2:26
"""

import gensim
from collections import OrderedDict

model = None


def init():
    global model
    if not model:
        model = gensim.models.Word2Vec.load("data/wiki_chs.model")


def simi_words(word):
    init()

    simis = model.most_similar(word)

    return OrderedDict(simis)


def distance(word1, word2):
    init()

    return model.similarity(word1, word2)


def run():
    word = '绿萝'
    simis = simi_words(word)

    print("{} most simi words: \n")
    for k, v in simis.items():
        print('{} -> {}\n'.format(k, v))

    print('{}\n\n'.format('-' * 30))

    word1 = '吊兰'
    word2 = '吊篮'

    print('{}&{} distance is: {:.6f}\n'.format(word, word1, distance(word, word1)))
    print('{}&{} distance is: {:.6f}\n'.format(word, word2, distance(word, word2)))


if __name__ == '__main__':
    run()
