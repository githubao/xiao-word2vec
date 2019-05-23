#!/usr/bin/env python
# encoding: utf-8

"""
@description: 热门名词top5000

@author: baoqiang
@time: 2019-05-23 17:34
"""

from pyspark import SparkContext
import os
import shutil
import jieba.posseg as pseg
from pathlib import Path

# input_file = 'words.txt'
input_file = '{}/Downloads/words.txt'.format(Path.home())
out_path = 'word_cnt'

os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3"


def count_words():
    sc = SparkContext('local', 'wordcount')

    # read file
    file = sc.textFile(input_file)
    words = file.flatMap(lambda line: line.split(" "))

    # count
    counts = words.filter(lambda w: isnoun(w)) \
        .map(lambda w: (w, 1)) \
        .reduceByKey(lambda x, y: x + y) \
        .sortBy(lambda x: x[1], False) \
        .filter(lambda x: x[1] >= 10) \
        .map(lambda x: '{}\t{}'.format(x[0], x[1]))

    # print
    # for word, cnt in counts.collect():
    #     print(word, cnt)

    # for res in counts.collect():
    #     print(res)

    print("process cnt: {}".format(counts.count()))

    # make out dir
    if os.path.exists(out_path):
        shutil.rmtree(out_path, True)

    # save to file
    counts.saveAsTextFile(out_path)


def isnoun(word):
    words = list(pseg.cut(word))
    # print(word, words)

    if len(words) != 1:
        return False

    for w, pos in words:
        return pos.startswith('n')


def hello():
    a = '篮球'
    print(isnoun(a))


if __name__ == '__main__':
    count_words()
    # hello()
