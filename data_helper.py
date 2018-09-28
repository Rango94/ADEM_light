#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_helper.py
# @Author: nanzhi.wang
# @Date  : 2018/9/17
import sys
import os
import jieba as jb
import json
import numpy as np
import codecs
import random as rd
import math

class data_helper:

    def __init__(self, config ,train_file='', val_file='',test_file=''):

        cate=config['cate']
        self.weight=config['weight']
        data_flag=config['data']
        dic_file=config['seg']

        if data_flag=='8':
            self.data_pre='../DATA_8/'
        elif data_flag=='9':
            self.data_pre = '../DATA_9/'
        elif data_flag=='origin':
            self.data_pre = '../DATA_origin/'
        elif data_flag=='all':
            self.data_pre='../DATA/'

        if dic_file=='ipx':
            self.max_len=30
        else:
            self.max_len=18

        if cate=='two':
            self.cateflag=True
        else:
            self.cateflag=False

        if dic_file=='jieba_single':
            self.DIC_FILE = self.data_pre + 'word_dic_single'
        elif dic_file=='jieba':
            self.DIC_FILE = 'word_dic_jieba'
        elif dic_file=='nio':
            self.DIC_FILE = 'word_dic_nioseg'
        elif dic_file=='ipx':
            self.DIC_FILE = 'word_dic_ipx'


        if train_file=='':
            self.TRAIN_FILE = self.data_pre+'corpus_normal_random_train_idx_'+dic_file
        else:
            self.TRAIN_FILE = train_file

        if val_file=='':
            self.VAL_FILE = self.data_pre+'corpus_normal_random_val_idx_'+dic_file
        else:
            self.VAL_FILE=val_file

        if test_file=='':
            self.TEST_FILE = self.data_pre+'corpus_normal_random_test_idx_'+dic_file
        else:
            self.TEST_FILE =test_file

        self.corpus_train = codecs.open(self.TRAIN_FILE, 'r', 'utf-8')
        self.corpus_val = codecs.open(self.VAL_FILE, 'r', 'utf-8')
        self.corpus_test = codecs.open(self.TEST_FILE, 'r', 'utf-8')

        self.cont_dic = {0: 221604, 1: 54029, 2: 245302, 3: 245664, 4: 56014}

        #计算权重，权重和频率成反比。
        for key in self.cont_dic:
            self.cont_dic[key] = math.pow(float(54029) / float(self.cont_dic[key]), 0.5)

        self.builddic()
        self.word_dic=json.load(codecs.open(self.DIC_FILE,'r','utf-8'))
        self.vocab_size = len(self.word_dic)

    #随机生成负样本
    def generate_negative_sample(self):
        context, true_response, model_response=[[self.get_random() for i in range(rd.randint(1, self.max_len))] for _ in range(3)]
        return context, true_response, model_response,0,0.5

    def get_random(self):
        idx=rd.randint(1,self.vocab_size)
        while rd.random()<math.pow((float(idx)/float(self.vocab_size)),0.3):
            idx = rd.randint(1, self.vocab_size)
        return idx

    #随机生成句子级别的负样本
    def generate_negative_sample_seq(self):
        while True:
            line = self.corpus_train.readline()
            if line == '':
                self.corpus_train.close()
                self.corpus_train=codecs.open(self.TRAIN_FILE, 'r', 'utf-8')
                line = self.corpus_train.readline()
            line = line.strip().split('\t')
            if line[3] != '0' and line[3] != '1' and line[3] != '2' and line[3] != '3' and line[3] != '4':
                continue
            context_tmp = [int(i) for i in line[rd.randint(0,2)].split(' ')]
            if context_tmp and rd.random()>0.3:
                break

        while True:
            line = self.corpus_train.readline()
            if line == '':
                self.corpus_train.close()
                self.corpus_train = codecs.open(self.TRAIN_FILE, 'r', 'utf-8')
                line = self.corpus_train.readline()
            line = line.strip().split('\t')
            if line[3] != '0' and line[3] != '1' and line[3] != '2' and line[3] != '3' and line[3] != '4':
                continue
            true_response_tmp = [int(i) for i in line[rd.randint(0,2)].split(' ')]
            if true_response_tmp and rd.random() > 0.3:
                break

        while True:
            line = self.corpus_train.readline()
            if line == '':
                self.corpus_train.close()
                self.corpus_train = codecs.open(self.TRAIN_FILE, 'r', 'utf-8')
                line = self.corpus_train.readline()
            line = line.strip().split('\t')
            if line[3] != '0' and line[3] != '1' and line[3] != '2' and line[3] != '3' and line[3] != '4':
                continue
            model_response_tmp = [int(i) for i in line[rd.randint(0,2)].split(' ')]
            if model_response_tmp and rd.random() > 0.3:
                break

        return context_tmp,true_response_tmp,model_response_tmp,0,0.5

    #获取一个batch
    def next_batch(self,size):
        context=[]
        true_response=[]
        model_response=[]
        human_score=[]
        context_mask=[]
        model_response_mask=[]
        true_response_mask=[]
        grads_wt=[]
        while size>0:
            if rd.random() > 0.7:
                continue    #不想挨着顺序取

            if rd.random() > 0.9:
                # 随机生成负样本，该负样本为全随机
                context_tmp, true_response_tmp, model_response_tmp, human_score_, grads_wt_ = self.generate_negative_sample()
                human_score.append(human_score_)
                grads_wt.append(grads_wt_)
            else:
                if rd.random() > 0.9:
                    #随机生成负样本，该负样本为句子级别的拼凑
                    context_tmp, true_response_tmp, model_response_tmp, human_score_, grads_wt_ = self.generate_negative_sample_seq()
                    human_score.append(human_score_)
                    grads_wt.append(grads_wt_)
                else:
                    line = self.corpus_train.readline()
                    if line == '':
                        self.corpus_train.close()
                        self.corpus_train = codecs.open(self.TRAIN_FILE, 'r', 'utf-8')
                        line = self.corpus_train.readline()

                    line = line.strip().split('\t')
                    if line[3] != '0' and line[3] != '1' and line[3] != '2' and line[3] != '3' and line[3] != '4':
                        continue

                    context_tmp = [int(i) for i in line[0].split(' ')]
                    true_response_tmp = [int(i) for i in line[1].split(' ')]
                    model_response_tmp = [int(i) for i in line[2].split(' ')]

                    #决定二分类还是回归，默认是回归，但实际上，模型的loss计算只写了回归的部分，如果要做逻辑回归的话，得重写模型的loss
                    if self.cateflag:
                        human_score.append(0 if int(line[3]) <= 2 else 1)
                    else:
                        human_score.append(int(line[3]))

                    #样本权重，只有训练数据会添加样本权重
                    if self.weight:
                        grads_wt.append(self.cont_dic[int(line[3])])
                    else:
                        grads_wt.append(1)

            context_mask.append(min(self.max_len, len(context_tmp)))
            true_response_mask.append(min(self.max_len, len(true_response_tmp)))
            model_response_mask.append(min(self.max_len, len(model_response_tmp)))

            while len(context_tmp)<self.max_len:
                context_tmp.append(0)
            while len(context_tmp)>self.max_len:
                context_tmp.pop(-1)
            while len(true_response_tmp)<self.max_len:
                true_response_tmp.append(0)
            while len(true_response_tmp)>self.max_len:
                true_response_tmp.pop(-1)
            while len(model_response_tmp)<self.max_len:
                model_response_tmp.append(0)
            while len(model_response_tmp)>self.max_len:
                model_response_tmp.pop(-1)
            context.append(np.array(context_tmp))
            true_response.append(np.array(true_response_tmp))
            model_response.append(np.array(model_response_tmp))
            size-=1


        return np.array(context),np.array(true_response),\
               np.array(model_response),np.array(context_mask),\
               np.array(true_response_mask),np.array(model_response_mask),\
               np.array(human_score),np.array(grads_wt)

    #获取测试数据
    def get_test_data(self):
        context = []
        true_response = []
        model_response = []
        human_score = []
        context_mask = []
        model_response_mask = []
        true_response_mask = []
        grads_wt = []
        while True:
            line = self.corpus_test.readline()
            if line=='':
                break
            line = line.strip().split('\t')
            if line[3] != '0' and line[3] != '1' and line[3] != '2' and line[3] != '3' and line[3] != '4':
                continue

            context_tmp = [int(i) for i in line[0].split(' ')]
            true_response_tmp = [int(i) for i in line[1].split(' ')]
            model_response_tmp = [int(i) for i in line[2].split(' ')]


            context_mask.append(min(self.max_len,len(context_tmp)))
            true_response_mask.append(min(self.max_len,len(true_response_tmp)))
            model_response_mask.append(min(self.max_len,len(model_response_tmp)))

            if self.cateflag:
                human_score.append(0 if int(line[3]) <= 2 else 1)
            else:
                human_score.append(int(line[3]))

            grads_wt.append(1)
            while len(context_tmp) < self.max_len:
                context_tmp.append(0)
            while len(context_tmp) > self.max_len:
                context_tmp.pop(-1)
            while len(true_response_tmp) < self.max_len:
                true_response_tmp.append(0)
            while len(true_response_tmp) > self.max_len:
                true_response_tmp.pop(-1)
            while len(model_response_tmp) < self.max_len:
                model_response_tmp.append(0)
            while len(model_response_tmp) > self.max_len:
                model_response_tmp.pop(-1)

            context.append(np.array(context_tmp))
            true_response.append(np.array(true_response_tmp))
            model_response.append(np.array(model_response_tmp))


        self.corpus_test.close()
        self.corpus_test = codecs.open(self.TEST_FILE, 'r', 'utf-8')
        return np.array(context), np.array(true_response), np.array(model_response), np.array(context_mask), np.array(
            true_response_mask), np.array(model_response_mask), np.array(human_score), np.array(grads_wt)

    #获取验证数据
    def get_val_data(self):
        context = []
        true_response = []
        model_response = []
        human_score = []
        context_mask = []
        model_response_mask = []
        true_response_mask = []
        grads_wt=[]
        while True:
            if rd.random()<0.8:
                line = self.corpus_val.readline()
                if line == '' :
                    break
                line = line.strip().split('\t')

                if line[3] != '0' and line[3] != '1' and line[3] != '2' and line[3] != '3' and line[3]!='4':
                    continue

                context_tmp = [int(i) for i in line[0].split(' ')]
                true_response_tmp = [int(i) for i in line[1].split(' ')]
                model_response_tmp = [int(i) for i in line[2].split(' ')]

                if self.cateflag:
                    human_score.append(0 if int(line[3]) <= 2 else 1)
                else:
                    human_score.append(int(line[3]))

                grads_wt.append(1)

            else:
                context_tmp, true_response_tmp, model_response_tmp, human_score_, grads_wt_ = self.generate_negative_sample()
                human_score.append(human_score_)
                grads_wt.append(grads_wt_)

            context_mask.append(min(self.max_len, len(context_tmp)))
            true_response_mask.append(min(self.max_len, len(true_response_tmp)))
            model_response_mask.append(min(self.max_len, len(model_response_tmp)))

            while len(context_tmp) < self.max_len:
                context_tmp.append(0)
            while len(context_tmp) > self.max_len:
                context_tmp.pop(-1)
            while len(true_response_tmp) < self.max_len:
                true_response_tmp.append(0)
            while len(true_response_tmp) > self.max_len:
                true_response_tmp.pop(-1)
            while len(model_response_tmp) < self.max_len:
                model_response_tmp.append(0)
            while len(model_response_tmp) > self.max_len:
                model_response_tmp.pop(-1)

            context.append(np.array(context_tmp))
            true_response.append(np.array(true_response_tmp))
            model_response.append(np.array(model_response_tmp))
        self.corpus_val.close()
        self.corpus_val=codecs.open(self.VAL_FILE, 'r', 'utf-8')
        return np.array(context), np.array(true_response), np.array(model_response), np.array(context_mask), np.array(
            true_response_mask), np.array(model_response_mask), np.array(human_score),np.array(grads_wt)


    #写这个方法是为了看一下模型在训练集上的收敛情况
    def get_train_data(self,size=15000):
        rd1=size/200000
        context = []
        true_response = []
        model_response = []
        human_score = []
        context_mask = []
        model_response_mask = []
        true_response_mask = []
        grads_wt = []
        while size > 0:
            if rd1>rd.random():
                continue
            line = self.corpus_train.readline()

            if line == '':
                self.corpus_train.close()
                self.corpus_train=codecs.open(self.TRAIN_FILE, 'r', 'utf-8')
                line = self.corpus_train.readline()

            line = line.strip().split('\t')
            if line[3] != '0' and line[3] != '1' and line[3] != '2' and line[3] != '3' and line[3] != '4':
                continue

            context_tmp = [int(i) for i in line[0].split(' ')]
            true_response_tmp = [int(i) for i in line[1].split(' ')]
            model_response_tmp = [int(i) for i in line[2].split(' ')]

            context_mask.append(min(self.max_len,len(context_tmp)))
            true_response_mask.append(min(self.max_len,len(true_response_tmp)))
            model_response_mask.append(min(self.max_len,len(model_response_tmp)))

            if self.cateflag:
                human_score.append(0 if int(line[3])<=2 else 1)
            else:
                human_score.append(int(line[3]))

            grads_wt.append(1)
            while len(context_tmp) < self.max_len:
                context_tmp.append(0)
            while len(context_tmp) > self.max_len:
                context_tmp.pop(-1)
            while len(true_response_tmp) < self.max_len:
                true_response_tmp.append(0)
            while len(true_response_tmp) > self.max_len:
                true_response_tmp.pop(-1)
            while len(model_response_tmp) < self.max_len:
                model_response_tmp.append(0)
            while len(model_response_tmp) > self.max_len:
                model_response_tmp.pop(-1)

            context.append(np.array(context_tmp))
            true_response.append(np.array(true_response_tmp))
            model_response.append(np.array(model_response_tmp))
            size -= 1
        return np.array(context), np.array(true_response), \
               np.array(model_response), np.array(context_mask), \
               np.array(true_response_mask), np.array(model_response_mask), \
               np.array(human_score), np.array(grads_wt)

    #从特定文件中获取样本，注意这个文件必须是索引文件，不能是源文件。
    def get_specific_data(self,file):
        context = []
        true_response = []
        model_response = []
        context_mask = []
        model_response_mask = []
        true_response_mask = []
        corpus_specific=codecs.open(file, 'r', 'utf-8')
        while True:
            line = corpus_specific.readline()
            if line == '':
                break
            line = line.strip().split('\t')
            if len(line)<3:
                continue
            context_tmp = [int(i) for i in line[0].split(' ')]
            true_response_tmp = [int(i) for i in line[1].split(' ')]
            model_response_tmp = [int(i) for i in line[2].split(' ')]

            context_mask.append(min(self.max_len, len(context_tmp)))
            true_response_mask.append(min(self.max_len, len(true_response_tmp)))
            model_response_mask.append(min(self.max_len, len(model_response_tmp)))

            while len(context_tmp) < self.max_len:
                context_tmp.append(0)
            while len(context_tmp) > self.max_len:
                context_tmp.pop(-1)
            while len(true_response_tmp) < self.max_len:
                true_response_tmp.append(0)
            while len(true_response_tmp) > self.max_len:
                true_response_tmp.pop(-1)
            while len(model_response_tmp) < self.max_len:
                model_response_tmp.append(0)
            while len(model_response_tmp) > self.max_len:
                model_response_tmp.pop(-1)

            context.append(np.array(context_tmp))
            true_response.append(np.array(true_response_tmp))
            model_response.append(np.array(model_response_tmp))
        corpus_specific.close()
        return np.array(context), np.array(true_response), np.array(model_response), np.array(context_mask), np.array(
            true_response_mask), np.array(model_response_mask)


    def builddic(self):
        if not os.path.exists(self.DIC_FILE):
            dic={}
            dic['##stop']=0
            i=1
            for line in codecs.open(self.data_pre+'corpus_normal_random','r','utf-8'):
                line=line.strip().split('\t')[:3]
                for each in line:
                    for word in jb.cut(each):
                        if word not in dic:
                            dic[word]=i
                            i+=1
            with codecs.open(self.DIC_FILE,'w','utf-8') as fo:
                json.dump(dic,fo)
        else:
            return 0

if __name__=='__main__':
    config = {'score_style': 'mine',
              'normal': True,
              'LR': 0.2,
              'cate': 'mlut',
              'weight': True,
              'data': 'all',
              'seg': 'jieba',
              'prewordembedding': True,
              }
    dh=data_helper(config)

    context_input_, refrence_input_, model_input_, context_sequence_length_, \
    refrence_sequence_length_, model_sequence_length_, human_score_, grad_ys_ = dh.get_val_data()
    print(context_input_.shape, refrence_input_.shape, model_input_.shape,
          context_sequence_length_.shape, refrence_sequence_length_.shape,
          model_sequence_length_.shape, human_score_.shape)
