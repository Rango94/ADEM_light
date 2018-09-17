import numpy as np
import json
import codecs
import jieba as jb
import sys
import random as rd


if __name__=='__main__':

    word_dic = json.load(codecs.open('word_dic', 'r', 'utf-8'))

    with codecs.open(sys.argv[1], 'r', 'utf-8') as corpus:
        with codecs.open(sys.argv[1] + '_idx', 'w', 'utf-8') as out:
            for line in corpus:
                line = line.strip().split('\t')
                tmp = ''
                for each in line[:3]:
                    for word in jb.cut(each):
                        try:
                            tmp += str(word_dic[word]) + ' '
                        except:
                            tmp+=str(0)+' '
                            pass
                    tmp = tmp.strip()
                    tmp += '\t'
                if len(line)==4:
                    tmp += str(line[-1])
                else:
                    tmp += str(0)
                if len(tmp.split('\t'))==4:
                    out.write(tmp + '\n')
                else:
                    print(line)


