#!/usr/bin/env python2
#-*- coding: UTF-8 -*-
#File: train.py
#Date: 20140810
#Author: Yang Liu <largelymfs@gmail.com>
#Description Train the topic representation using the topic model and the word2vec's skip gram

import gensim #modified gensim version
from gensim.models.word2vec import CombinedSentence
import pre_process # read the wordmap and the tassgin file and create the sentence
import sys

def train_twe1(sentences, topics, topic_number, tmp="tmp", output="output"):
    #id2word = pre_process.load_id2word(wordmapfile)
    #pre_process.load_sentences(tassignfile, id2word, tmp)
    #sentence_word = gensim.models.word2vec.LineSentence("%s/word.file" % tmp)
    sentence_word = [words.split() for words in sentences]
    print "Training the word vector..."
    w = gensim.models.Word2Vec(sentence_word, size=400, workers=20)
    sentence = [zip(sentence, topic) for sentence, topic in zip(sentences, topics)]
    print "Training the topic vector..."
    w.train_topic(topic_number, sentence)
    print "Saving the topic vectors..."
    w.save_topic("%s/topic_vector.txt" % output)
    print "Saving the word vectors..."
    w.save_wordvector("%s/word_vector.txt" % output)


if __name__=="__main__":
    if len(sys.argv)!=4:
        print "Usage : python train.py wordmap tassign topic_number"
        sys.exit(1)	
    reload(sys)
    sys.setdefaultencoding('utf-8')
    wordmapfile = sys.argv[1]
    tassignfile = sys.argv[2]
    topic_number = int(sys.argv[3])
    train_twe1(wordmapfile, tassignfile, topic_number)
