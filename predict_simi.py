#!/usr/bin/env python
# encoding: utf-8

"""
@description: 预测模型

@author: baoqiang
@time: 2019/3/6 下午2:26
"""

import gensim
from collections import OrderedDict, defaultdict
import time

# test = True
test = False


def run():
    word2vec = Word2Vec()

    word2vec.batch_simi_file()


class Word2Vec:
    """
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
    """

    model = None

    def __init__(self):
        start = time.time()

        if test:
            model_bin_file = "data/wiki_chs.bin.sample"
        else:
            model_bin_file = "data/wiki_chs.bin"

        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_bin_file, binary=True)

        end = time.time()
        print('load model complete, used {:.6f} seconds'.format(end - start))

    def batch_simi_file(self):
        """
        从文件中读取文件，然后累加分值，simi取50个
        :return:
        """
        results = defaultdict(float)

        filename = '/Users/baoqiang/Downloads/words.txt'
        with open(filename, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]

        for word in words:
            simis = self.get_simis(word, n=50)

            results[word] += 1

            for simi, score in simis.items():
                results[simi] += score

        sort_words = sorted(results.items(), key=lambda x: x[1], reverse=True)

        for word, score in sort_words:
            print('{} -> {:.6f}'.format(word, score))

    def batch_simi(self):
        """
        打印多个单词的词向量
        :return:
        """
        words = ["一嗨租车", "神州租车"]
        for word in words:
            print("[{}] most simi words is: ".format(word))
            simis = self.get_simis(word)
            for k, v in simis.items():
                print('{} -> {}'.format(k, v))

    def sample(self):
        """
        如何使用的一些简单入门例子
        :return:
        """
        word = '篮球'
        simis = self.get_simis(word)

        print("[{}] most simi words is: ".format(word))
        for k, v in simis.items():
            print('{} -> {}'.format(k, v))

        print('{}'.format('-' * 30))

        word1 = '足球'
        word2 = '饥饿'

        print('{}&{} distance is: {}'.format(word, word1, self.distance(word, word1)))
        print('{}&{} distance is: {}'.format(word, word2, self.distance(word, word2)))

    def get_simis(self, word, n=10):
        """
        基础的获取单个单词词向量的方法
        :param word:
        :param n:
        :return:
        """
        try:
            simis = self.model.most_similar(word, topn=n)
        except KeyError as e:
            print('key err: {}'.format(e))
            return {}

        return OrderedDict({k[0]: float('{:.6f}'.format(k[1])) for k in simis})

    def distance(self, word1, word2):
        """
        基础的获取两个单词的词向量的距离
        :param word1:
        :param word2:
        :return:
        """
        try:
            dis = self.model.similarity(word1, word2)
        except KeyError as e:
            print('key err: {}'.format(e))
            return 0

        return float('{:.6f}'.format(dis))


if __name__ == '__main__':
    run()
