#!/usr/bin/env python
# encoding: utf-8

"""
@description: 预测模型

@author: baoqiang
@time: 2019/3/6 下午2:26
"""

import gensim
from collections import OrderedDict
import datetime

model = None


def init():
    global model
    if not model:
        start = datetime.datetime.now()

        model_bin_file = "data/wiki_chs.bin"
        model = gensim.models.KeyedVectors.load_word2vec_format(model_bin_file, binary=True)

        end = datetime.datetime.now()
        print('load model complete, used {} seconds'.format(end - start))


def simi_words(word, n=10):
    init()

    try:
        simis = model.most_similar(word, topn=n)
    except KeyError as e:
        print('key err: {}'.format(e))
        return {}

    return OrderedDict({k[0]: float('{:.6f}'.format(k[1])) for k in simis})


def distance(word1, word2):
    init()

    try:
        dis = model.similarity(word1, word2)
    except KeyError as e:
        print('key err: {}'.format(e))
        return 0

    return float('{:.6f}'.format(dis))


def run():
    word = '篮球'
    simis = simi_words(word)

    print("[{}] most simi words is: ".format(word))
    for k, v in simis.items():
        print('{} -> {}'.format(k, v))

    print('{}'.format('-' * 30))

    word1 = '足球'
    word2 = '饥饿'

    print('{}&{} distance is: {}'.format(word, word1, distance(word, word1)))
    print('{}&{} distance is: {}'.format(word, word2, distance(word, word2)))


if __name__ == '__main__':
    run()

'''
[篮球] most simi words is: 
美式足球 -> 0.691394
橄榄球 -> 0.672521
棒球 -> 0.660013
男子篮球 -> 0.656124
排球 -> 0.655132
橄榄球队 -> 0.637557
冰球 -> 0.613005
篮球运动 -> 0.607658
足球 -> 0.602666
网球 -> 0.602374
------------------------------
篮球&足球 distance is: 0.602666
篮球&饥饿 distance is: 0.006415
'''
