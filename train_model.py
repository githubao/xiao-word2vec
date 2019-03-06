#!/usr/bin/env python
# encoding: utf-8

"""
@description: 

@author: baoqiang
@time: 2019/3/6 上午11:08
"""

from __future__ import print_function

import logging
import os
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def train():
    """
    使用gensim的Word2vec类进行训练，大小为200维
    保存模型为model1，保存word2vec格式的模型为model2
    :return:
    """

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 3:
        print("Useing: python train_word2vec_model.py input_text "
              "output_text")
        sys.exit(1)
    inp, outp = sys.argv[1:3]

    model = Word2Vec(LineSentence(inp), size=200, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())

    model.save('{}.model'.format(outp))
    model.wv.save_word2vec_format('{}.vec'.format(outp), binary=False)
    model.wv.save_word2vec_format('{}.bin'.format(outp), binary=True)


if __name__ == '__main__':
    train()
