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
        weight=config['weight']
        data_flag=config['data']
        dic_file=config['seg']

        self.train_file_list=['../DATA_8/corpus_normal_random_train_idx',
                         '../DATA_9/corpus_normal_random_train_idx',
                         '../DATA_orgin/corpus_normal_random_train_idx',]

        self.val_file_list = ['../DATA_8/corpus_normal_random_val_idx',
                           '../DATA_9/corpus_normal_random_val_idx',
                           '../DATA_orgin/corpus_normal_random_val_idx',]

        self.test_file_list = ['../DATA_8/corpus_normal_random_test_idx',
                         '../DATA_9/corpus_normal_random_test_idx',
                         '../DATA_orgin/corpus_normal_random_test_idx',]

        if config['seg']=='nio':
            self.train_file_list=[filename+'_nioseg' for filename in self.train_file_list]
            self.test_file_list = [filename + '_nioseg' for filename in self.test_file_list]
            self.val_file_list = [filename + '_nioseg' for filename in self.val_file_list]
        self.train_fo_list=[codecs.open(filename,'r','utf-8') for filename in self.train_file_list]
        if data_flag=='8':
            self.data_pre='../DATA_8/'
        elif data_flag=='9':
            self.data_pre = '../DATA_9/'
        elif data_flag=='orgin':
            self.data_pre = '../DATA_orgin/'
        elif data_flag=='all':
            self.data_pre='all'
        else:
            self.data_pre = '../DATA_total/'

        self.train_cont=rd.randint(0,2)
        self.test_cont = 0
        self.val_cont = 0

        self.test_idx=0
        self.max_len=18
        self.weight=weight

        if cate=='two':
            self.cateflag=True
        else:
            self.cateflag=False

        if dic_file=='jieba_single':
            self.DIC_FILE = self.data_pre + 'word_dic_single'
        elif dic_file=='jieba':
            self.DIC_FILE = 'word_dic'
        elif dic_file=='nio':
            self.DIC_FILE = 'word_dic_nioseg'


        if train_file=='':
            self.TRAIN_FILE = self.data_pre+'corpus_normal_random_train_idx'
        else:
            self.TRAIN_FILE = train_file

        if val_file=='':
            self.VAL_FILE = self.data_pre+'corpus_normal_random_val_idx'
        else:
            self.VAL_FILE=val_file

        if test_file=='':
            self.TEST_FILE = self.data_pre+'corpus_normal_random_test_idx'
        else:
            self.TEST_FILE =test_file

        self.cont_dic_8 = {0: 142121, 1: 33358, 2: 67408, 3: 107272, 4: 12162}
        for key in self.cont_dic_8:
            self.cont_dic_8[key] = math.pow(float(12162) / float(self.cont_dic_8[key]), 0.5)

        self.cont_dic_9 = {0: 65763, 1: 15531, 2: 46452, 3: 49359, 4: 11183}
        for key in self.cont_dic_9:
            self.cont_dic_9[key] = math.pow(float(11183) / float(self.cont_dic_9[key]),0.5)

        self.cont_dic_orgin = {0: 36322, 1: 10477, 2: 21376, 3: 110334, 4: 36847}
        for key in self.cont_dic_orgin:
            self.cont_dic_orgin[key] = math.pow(float(10477) / float(self.cont_dic_orgin[key]),0.5)

        self.weight_dic={0:self.cont_dic_8,1:self.cont_dic_9,2:self.cont_dic_orgin}

        self.builddic()
        self.word_dic=json.load(codecs.open(self.DIC_FILE,'r','utf-8'))
        if data_flag!='all':
            self.corpus_train=codecs.open(self.TRAIN_FILE, 'r','utf-8')
            self.corpus_val=codecs.open(self.VAL_FILE, 'r', 'utf-8')
            self.corpus_test = codecs.open(self.TEST_FILE, 'r', 'utf-8')
        else:

            self.change('val')
            self.change('test')

        self.vocab_size=len(self.word_dic)


    def change(self,file_flag):
        if self.data_pre=='all':
            if file_flag=='train':
                self.TRAIN_FILE=self.train_file_list[self.train_cont % len(self.train_file_list)]
                self.corpus_train = codecs.open(self.TRAIN_FILE, 'r', 'utf-8')
                self.train_cont+=1
                return True

            if file_flag=='test':
                if self.test_cont<len(self.test_file_list):
                    self.TEST_FILE = self.test_file_list[self.test_cont % len(self.test_file_list)]
                    self.corpus_test = codecs.open(self.TEST_FILE, 'r', 'utf-8')
                    self.test_cont+=1
                    return True
                else:
                    self.test_cont=0
                    self.TEST_FILE = self.test_file_list[self.test_cont % len(self.test_file_list)]
                    self.test_cont +=1
                    self.corpus_test = codecs.open(self.TEST_FILE, 'r', 'utf-8')
                    return False

            if file_flag=='val':
                if self.val_cont<len(self.val_file_list):
                    self.VAL_FILE = self.val_file_list[self.val_cont % len(self.val_file_list)]
                    self.corpus_val = codecs.open(self.VAL_FILE, 'r', 'utf-8')
                    self.val_cont+=1
                    return True
                else:
                    self.val_cont=0
                    self.VAL_FILE = self.val_file_list[self.val_cont % len(self.val_file_list)]
                    self.val_cont+=1
                    self.corpus_val = codecs.open(self.VAL_FILE, 'r', 'utf-8')
                    return False

    def generate_negative_sample(self):
        context, true_response, model_response=[[self.get_random() for i in range(rd.randint(1, 18))] for _ in range(3)]
        human_score=0
        grads_wt=0.5
        return context, true_response, model_response,human_score,grads_wt

    def get_random(self):
        idx=rd.randint(1,self.vocab_size)
        while rd.random()<math.pow((float(idx)/float(self.vocab_size)),0.3):
            idx = rd.randint(1, self.vocab_size)
        return idx


    def generate_negative_sample_seq(self):
        context_tmp=[]
        true_response_tmp=[]
        model_response_tmp=[]
        while True:
            self.train_cont = rd.randint(0, 2)
            self.TRAIN_FILE = self.train_file_list[self.train_cont]
            self.corpus_train = self.train_fo_list[self.train_cont]
            line = self.corpus_train.readline()

            if line == '':
                self.corpus_train.close()
                self.corpus_train = codecs.open(self.train_file_list[self.train_cont], 'r', 'utf-8')
                self.train_fo_list[self.train_cont] = self.corpus_train
                self.TRAIN_FILE = self.train_file_list[self.train_cont]
                line = self.corpus_train.readline()
            line = line.strip().split('\t')
            if line[3] != '0' and line[3] != '1' and line[3] != '2' and line[3] != '3' and line[3] != '4':
                continue
            context_tmp = [int(i) for i in line[rd.randint(0,2)].split(' ')]
            if context_tmp and rd.random()>0.3:
                break

        while True:
            self.train_cont = rd.randint(0, 2)
            self.TRAIN_FILE = self.train_file_list[self.train_cont]
            self.corpus_train = self.train_fo_list[self.train_cont]
            line = self.corpus_train.readline()

            if line == '':
                self.corpus_train.close()
                self.corpus_train = codecs.open(self.train_file_list[self.train_cont], 'r', 'utf-8')
                self.train_fo_list[self.train_cont] = self.corpus_train
                self.TRAIN_FILE = self.train_file_list[self.train_cont]
                line = self.corpus_train.readline()
            line = line.strip().split('\t')
            if line[3] != '0' and line[3] != '1' and line[3] != '2' and line[3] != '3' and line[3] != '4':
                continue
            true_response_tmp = [int(i) for i in line[rd.randint(0,2)].split(' ')]
            if true_response_tmp and rd.random() > 0.3:
                break

        while True:
            self.train_cont = rd.randint(0, 2)
            self.TRAIN_FILE = self.train_file_list[self.train_cont]
            self.corpus_train = self.train_fo_list[self.train_cont]
            line = self.corpus_train.readline()

            if line == '':
                self.corpus_train.close()
                self.corpus_train = codecs.open(self.train_file_list[self.train_cont], 'r', 'utf-8')
                self.train_fo_list[self.train_cont] = self.corpus_train
                self.TRAIN_FILE = self.train_file_list[self.train_cont]
                line = self.corpus_train.readline()
            line = line.strip().split('\t')
            if line[3] != '0' and line[3] != '1' and line[3] != '2' and line[3] != '3' and line[3] != '4':
                continue
            model_response_tmp = [int(i) for i in line[rd.randint(0,2)].split(' ')]
            if model_response_tmp and rd.random() > 0.3:
                break

        return context_tmp,true_response_tmp,model_response_tmp,0,0.5


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
            if rd.random()>0.7:
                continue
            if rd.random()<0.8:
                if rd.random()<0.8:
                    self.train_cont=rd.randint(0,2)
                    self.TRAIN_FILE = self.train_file_list[self.train_cont]
                    self.corpus_train = self.train_fo_list[self.train_cont]
                    line=self.corpus_train.readline()

                    if line=='':
                        self.corpus_train.close()
                        self.corpus_train=codecs.open(self.train_file_list[self.train_cont],'r','utf-8')
                        self.train_fo_list[self.train_cont] = self.corpus_train
                        self.TRAIN_FILE=self.train_file_list[self.train_cont]
                        line=self.corpus_train.readline()

                    line=line.strip().split('\t')
                    if line[3]!='0' and line[3]!='1' and line[3]!='2' and line[3]!='3' and line[3]!='4':
                        continue

                    context_tmp=[int(i) for i in line[0].split(' ')]
                    true_response_tmp = [int(i) for i in line[1].split(' ')]
                    model_response_tmp = [int(i) for i in line[2].split(' ')]

                    # human_score.append(int(line[3]))
                    if self.cateflag:
                        human_score.append(0 if int(line[3]) <= 2 else 1)
                    else:
                        human_score.append(int(line[3]))

                    if self.weight:
                        grads_wt.append(self.weight_dic[(self.train_cont - 1) % 3][int(line[3])])
                    else:
                        grads_wt.append(1)
                else:
                    context_tmp, true_response_tmp, model_response_tmp,human_score_,grads_wt_=self.generate_negative_sample_seq()
                    human_score.append(human_score_)
                    grads_wt.append(grads_wt_)

            else:
                #随机生成负样本，该副样本为全随机
                context_tmp, true_response_tmp, model_response_tmp, human_score_, grads_wt_=self.generate_negative_sample()
                human_score.append(human_score_)
                grads_wt.append(grads_wt_)

            if self.test_idx != 0:
                context_mask.append(self.test_idx)
                true_response_mask.append(self.test_idx)
                model_response_mask.append(self.test_idx)
            else:
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

            if line == '' :
                if not self.change('test'):
                    break
                else:
                    continue

            line = line.strip().split('\t')
            if line[3] != '0' and line[3] != '1' and line[3] != '2' and line[3] != '3' and line[3] != '4':
                continue

            context_tmp = [int(i) for i in line[0].split(' ')]
            true_response_tmp = [int(i) for i in line[1].split(' ')]
            model_response_tmp = [int(i) for i in line[2].split(' ')]

            if self.test_idx!=0:
                context_mask.append(self.test_idx)
                true_response_mask.append(self.test_idx)
                model_response_mask.append(self.test_idx)
            else:
                context_mask.append(min(self.max_len,len(context_tmp)))
                true_response_mask.append(min(self.max_len,len(true_response_tmp)))
                model_response_mask.append(min(self.max_len,len(model_response_tmp)))

            # human_score.append(int(line[3]))
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
                    if not self.change('val'):
                        break
                    else:
                        continue
                line = line.strip().split('\t')

                if line[3] != '0' and line[3] != '1' and line[3] != '2' and line[3] != '3' and line[3]!='4':
                    continue

                context_tmp = [int(i) for i in line[0].split(' ')]
                true_response_tmp = [int(i) for i in line[1].split(' ')]
                model_response_tmp = [int(i) for i in line[2].split(' ')]

                # human_score.append(int(line[3]))
                if self.cateflag:
                    human_score.append(0 if int(line[3]) <= 2 else 1)
                else:
                    human_score.append(int(line[3]))

                grads_wt.append(1)

            else:
                context_tmp, true_response_tmp, model_response_tmp, human_score_, grads_wt_ = self.generate_negative_sample()
                human_score.append(human_score_)
                grads_wt.append(grads_wt_)

            if self.test_idx != 0:
                context_mask.append(self.test_idx)
                true_response_mask.append(self.test_idx)
                model_response_mask.append(self.test_idx)
            else:
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

            self.train_cont = rd.randint(0, 2)
            self.TRAIN_FILE = self.train_file_list[self.train_cont]
            self.corpus_train = self.train_fo_list[self.train_cont]
            line = self.corpus_train.readline()

            if line == '':
                self.corpus_train.close()
                self.train_fo_list[self.train_cont] = codecs.open(self.train_file_list[self.train_cont],
                                                                  'r', 'utf-8')
                self.TRAIN_FILE = self.train_file_list[self.train_cont]
                self.corpus_train = self.train_fo_list[self.train_cont]
                line = self.corpus_train.readline()

            line = line.strip().split('\t')
            if line[3] != '0' and line[3] != '1' and line[3] != '2' and line[3] != '3' and line[3] != '4':
                continue

            context_tmp = [int(i) for i in line[0].split(' ')]
            true_response_tmp = [int(i) for i in line[1].split(' ')]
            model_response_tmp = [int(i) for i in line[2].split(' ')]

            if self.test_idx!=0:
                context_mask.append(self.test_idx)
                true_response_mask.append(self.test_idx)
                model_response_mask.append(self.test_idx)
            else:
                context_mask.append(min(self.max_len,len(context_tmp)))
                true_response_mask.append(min(self.max_len,len(true_response_tmp)))
                model_response_mask.append(min(self.max_len,len(model_response_tmp)))

            # human_score.append(int(line[3]))

            if self.cateflag:
                human_score.append(0 if int(line[3])<=2 else 1)
            else:
                human_score.append(int(line[3]))

            grads_wt.append(1)
            # grads_wt.append(1)
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
