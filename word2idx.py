import numpy as np
import json
import codecs
import jieba as jb
import sys
import random as rd


def dododo(word_dic,file):
    if 'jieba' in word_dic:
        bak='_idx'
    elif 'nioseg' in word_dic:
        bak='_idx_nioseg'
    elif 'ipx' in word_dic:
        bak='_idx_ipx'
    word_dic = json.load(codecs.open(word_dic, 'r', 'utf-8'))
    with codecs.open(file, 'r', 'utf-8') as corpus:
        with codecs.open(file + bak, 'w', 'utf-8') as out:
            for line in corpus:
                line = line.strip().split('\t')
                tmp = ''
                for each in line[:3]:
                    for word in jb.cut(each):
                    # for word in each:
                        try:
                            tmp += str(word_dic[word]) + ' '
                        except:
                            print word
                            tmp += str(0) + ' '
                            pass
                    tmp = tmp.strip()
                    tmp += '\t'
                if len(line) == 4:
                    tmp += str(line[-1])
                else:
                    tmp += str(0)
                if len(tmp.split('\t')) == 4:
                    out.write(tmp + '\n')
                else:
                    print(line)

if __name__=='__main__':
    dododo('word_dic_jieba',sys.argv[1])



