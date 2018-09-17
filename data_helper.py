import sys
import os
import jieba as jb
import json
import numpy as np
import codecs
import random as rd
import math

class data_helper:

    def __init__(self, data_flag='' ,dic_file='',train_file='', val_file='',test_file='',cate='two',weight=True):

        self.train_file_list=['../DATA_8/corpus_normal_random_train_idx',
                         '../DATA_9/corpus_normal_random_train_idx',
                         '../DATA_orgin/corpus_normal_random_train_idx',]

        self.val_file_list = ['../DATA_8/corpus_normal_random_val_idx',
                           '../DATA_9/corpus_normal_random_val_idx',
                           '../DATA_orgin/corpus_normal_random_val_idx',]

        self.test_file_list = ['../DATA_8/corpus_normal_random_test_idx',
                         '../DATA_9/corpus_normal_random_test_idx',
                         '../DATA_orgin/corpus_normal_random_test_idx',]

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

        self.train_cont=0
        self.test_cont = 0
        self.val_cont = 0

        self.test_idx=0
        self.max_len=18
        self.weight=weight

        if cate=='two':
            self.cateflag=True
        else:
            self.cateflag=False

        if dic_file=='':
            self.DIC_FILE = self.data_pre + 'word_dic_single'
        else:
            self.DIC_FILE = 'word_dic'

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
            self.change('train')
            self.change('val')
            self.change('test')

        self.vocab_size=len(self.word_dic)


    def change(self,file_flag):
        if self.data_pre=='all':
            print(file_flag)
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
            line=self.corpus_train.readline()
            if line=='':
                self.corpus_train.close()
                self.change('train')
                self.corpus_train=codecs.open(self.TRAIN_FILE, 'r', 'utf-8')
                line=self.corpus_train.readline()
            line=line.strip().split('\t')
            if line[3]!='0' and line[3]!='1' and line[3]!='2' and line[3]!='3' and line[3]!='4':
                continue

            context_tmp=[int(i) for i in line[0].split(' ')]
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

            if self.weight:
                grads_wt.append(self.weight_dic[(self.train_cont-1)%3][int(line[3])])
            else:
                grads_wt.append(1)

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
            line = self.corpus_train.readline()
            if rd1>rd.random():
                continue
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


    def get_old_test_data(self):
        corpus_test = codecs.open('./DATA_8/olddata/corpus_normal_random_test_1.2', 'r','utf-8')
        out=[]
        for line in corpus_test:
            s=line.replace('\n','').split('\t')[-1]
            if s!='0' and s!='1' and s!='2' and s!='3' and s!='4':
                continue
            out.append(int(s))
        return np.array(out)


if __name__=='__main__':
    data_helper()
