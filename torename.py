#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : torename.py
# @Author: nanzhi.wang
# @Date  : 2018/9/21 下午5:15

from ADEM_light_train import *

if sys.argv[1] == '1':
    config = {'score_style': 'mine', 'normal': True, 'LR': 0.2, 'cate': 'mlut', 'weight': True, 'data': 'all',
              'seg': 'jieba', 'prewordembedding': False}
elif sys.argv[1] == '2':
    config = {'score_style': 'mine', 'normal': True, 'LR': 0.2, 'cate': 'mlut', 'weight': True, 'data': 'all',
              'seg': 'nio', 'prewordembedding': False}
elif sys.argv[1] == '3':
    config = {'score_style': 'mine', 'normal': True, 'LR': 0.2, 'cate': 'mlut', 'weight': True, 'data': 'all',
              'seg': 'jieba', 'prewordembedding': True}
elif sys.argv[1] == '4':
    config = {'score_style': 'mine', 'normal': True, 'LR': 0.2, 'cate': 'mlut', 'weight': True, 'data': 'all',
              'seg': 'nio', 'prewordembedding': True}


config_new=config.copy()
config_new['attflag']=False
Rename(config,config_new)