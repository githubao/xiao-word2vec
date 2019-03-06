#!/usr/bin/env python
# encoding: utf-8

"""
@description: 预处理中文语料

@author: baoqiang
@time: 2019/3/6 上午11:02
"""

from __future__ import print_function

import logging
import os.path
import six
import sys

from gensim.corpora import WikiCorpus


# finished iterating over Wikipedia corpus of 334014 documents with 76295909 positions
# (total 3257644 articles, 90346224 positions before pruning articles shorter than 50 words)
def process():
    """
    一共334014篇文章
    使用gensim的wikiCorpus类处理语料，把每篇文章写成一行text的文本，去掉标点符号
    设置lemmatize为false禁用英文pattern类的normalize工作
    :return:
    """
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) != 3:
        print("Using: python process_wiki.py enwiki.xxx.xml.bz2 wiki.en.text")
        sys.exit(1)
    inp, outp = sys.argv[1:3]
    space = " "
    i = 0

    output = open(outp, 'w')
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})

    for text in wiki.get_texts():
        if six.PY3:
            output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
        #   ###another method###
        #    output.write(
        #            space.join(map(lambda x:x.decode("utf-8"), text)) + '\n')
        else:
            output.write(space.join(text) + "\n")
        i = i + 1

        if i % 10000 == 0:
            logger.info("Saved " + str(i) + " articles")

    output.close()
    logger.info("Finished Saved " + str(i) + " articles")


if __name__ == '__main__':
    process()
