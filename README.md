# xiao-word2vec
training word2vec

following

http://www.52nlp.cn/%E4%B8%AD%E8%8B%B1%E6%96%87%E7%BB%B4%E5%9F%BA%E7%99%BE%E7%A7%91%E8%AF%AD%E6%96%99%E4%B8%8A%E7%9A%84word2vec%E5%AE%9E%E9%AA%8C

0. 安装python3, gensim, opencc, jieba
1. 下载语料：https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2
2. 预处理去标点：python3 process_wiki.py ~/Downloads/zhwiki-latest-pages-articles.xml.bz2 wiki_zh.txt
3. 繁简转化：opencc -i wiki_zh.txt -o wiki_chs.txt -c t2s.json
4. 分词处理：python3 -m jieba -d " " wiki_chs.txt > wiki_chs.seg
5. 训练模型：python3 train_model.py wiki_chs.seg wiki_chs
6. 使用模型：python3 predict_simi.py