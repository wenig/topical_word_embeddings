#!/usr/bin/env python2
#-*- coding: UTF-8 -*-
#Author: Yang Liu <largelymfs@gmail.com>
#Description:Topical Word Embeddings TWE-3(This version is not the one used in the experiment, which needs the twe-1's result as an initial value)

import gensim
import sys

def gen(filename, tmp="tmp"):
    content2id = {}
    cnt = 0
    fout =open("%s/data.tmp" % tmp,"w")
    max_word_number = 0
    max_topic_number =0
    with open(filename) as  f:
        for l in f:
            words = l.strip().split()
            for w in words:
                word_number, topic_number = w.split(':')
                word_number = int(word_number)
                topic_number = int(topic_number)
                if topic_number > max_topic_number:
                    max_topic_number = topic_number
                if word_number > max_word_number:
                    max_word_number = word_number
                if (word_number, topic_number) not in content2id:
                    content2id[(word_number, topic_number)] = cnt
                    now_id = cnt
                    cnt+=1
                else:
                    now_id = content2id[(word_number, topic_number)]
                print >>fout, word_number, topic_number, now_id,
            print >>fout
    fout.close()
    max_word_number +=1
    max_topic_number +=1
    return max_topic_number, max_word_number

class MyCorpus(object):
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        with open(self.filename) as f:
            for l in f:
                words = l.strip().split()
                length = len(words)/3
                res = [(int(words[3*i]),int(words[3 * i + 1]),words[3*i+2]) for i in range(length)]
                yield res

def load_wordmap(filename):
    id2word = {}
    with open(filename) as f:
        f.readline()
        for l in f:
            word, number = l.strip().split()
            number = int(number)
            id2word[number ]= word
    return id2word

def train_twe3(wordmapfile, tassignfile, tmp="tmp", output="output"):
    topic_number, word_number = gen(tassignfile)
    sentences = MyCorpus("%s/data.tmp" % tmp)
    id2word = load_wordmap(wordmapfile)

    print "Begin Training..."
    w = gensim.models.Word2Vec(sentences, window=5, size=400, workers=4,
                               word_number=word_number, topic_number=topic_number,
                               topic_size=400)
    print "Finish"
    print "Saving the word vector..."
    w.save_word_vector("%s/word_vector.txt" % output, id2word)
    print "Saving the topic vector..."
    w.save_topic_vector("%s/topic_vector.txt" % output)

if __name__=="__main__":
    if len(sys.argv)!=3:
        print "Usage : python train.py wordmap_filename, tassign_filename"
        sys.exit(1)
    wordmapfile = sys.argv[1]
    tassignfile = sys.argv[2]
    train_twe3(wordmapfile, tassignfile)
    

