#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : ADEM_light_train.py
# @Author: nanzhi.wang
# @Date  : 2018/9/17

import sys

from ADEM_light import *
from data_helper import *

#将配置文件展开为string
def tostring(config):
    keys_ = config.keys()
    keys_.sort()
    return '-'.join([key + '_' + str(config[key]) for key in keys_])

#根据config更改模型名字
def Rename(root,config_old,config_new):
    model_name_old = tostring(config_old)
    model_name_new = tostring(config_new)
    if model_name_new==model_name_old:
        return 0
    else:
        for file in os.listdir(root):
            print(file)
            if model_name_old in file:
                print '将%s改为%s'%(file,file.replace(model_name_old,model_name_new))
                os.rename(root+'/'+file, root+'/'+file.replace(model_name_old,model_name_new))


def main():
    '''
    设置data_helper和部分模型的超参数，利用这个配置文件给模型文件命名
    参数解释：
        score_style:
            取值:'mine';'adem'
            说明：模型计算最终得分的方式，'mine'方式将三个输入拼接为了一个向量，通过两层全连接神经网络最终输出结果；'adem'采用论文计算方式，使用矩阵变换计算三个输入之间的相似度作为最终结果。

        normal:
            取值:True;False
            说明：是否添加正则项

        LR:
            取值:数字
            说明:学习率

        cate:
            取值:'mlut';'two'
            说明:0-4的回归问题 或者 2分类问题

        weight:
            取值:True;False
            说明:是否为数据添加权重，由于各个类别分布不均匀，据定是否为数据添加权重，权重与各类别的频率成反比。

        data:
            取值:'8';'9';'origin';'all'
            说明:一共有三批不同的数据，默认用all

        seg:
            取值:'nio';'jieba';'ipx'
            说明:分词类型，分别为nio自己的分词，jieba 和 单字

        prewordembedding:
            取值:True;False
            说明:是否使用预训练词向量

        attflag:
            取值:True;False
            说明:是否使用attention机制
    '''
    config = {'score_style': 'adem',
              'normal': True,
              'LR': 1,
              'cate': 'mlut',
              'weight': True,
              'data': 'all',
              'seg': 'jieba',
              'prewordembedding': False,
              'attflag': True}

    model_name = tostring(config)
    CHECKPOINT_PATH = '../MODEL/' + model_name + '_ckpt'

    #判断是否有同名模型存在
    if os.path.exists(CHECKPOINT_PATH + '.index'):
        exists_flag = True
    else:
        exists_flag = False

    dh = data_helper(config=config)

    #设置模型超参数，之所以要设置两个超参数是为了利用上一个超参数来命名模型
    config_network = {
        'HIDDEN_SIZE': 256,
        'NUM_LAYERS': 2,
        'SRC_VOCAB_SIZE': dh.vocab_size,
        'KEEP_PROB': 0.8,
        'MAX_GRAD_NORM': 5,
        'word_embedding_file': 'word_dic_jieba_embedding.pk' if config[
                                                                    'seg'] == 'jieba' else 'word_dic_nioseg_embedding.pk',
        'max_len':dh.max_len
    }

    train_model=ADEM_model(config, config_network)
    saver=tf.train.Saver()
    step=0
    max_loop=9999999
    min_=9999
    marks=[]

    #控制tensorflow的显存占比
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.per_process_gpu_memory_fraction = 0.9
    config_tf.gpu_options.allow_growth = True

    with tf.Session(config=config_tf) as sess:
        tf.global_variables_initializer().run()
        #如果模型存在，那么载入后再做持续训练
        if exists_flag:
            print('loading trained model')
            saver = tf.train.Saver()
            saver.restore(sess, CHECKPOINT_PATH)

        for i in range(max_loop):
            context_input_, refrence_input_, model_input_, context_sequence_length_, \
            refrence_sequence_length_,model_sequence_length_,human_score_,grad_ys_=dh.next_batch(300)

            step=train_model.train_on_batch(sess,step,feed_dict={train_model.context_input:context_input_,
                                                                  train_model.context_sequence_length:context_sequence_length_,
                                                                  train_model.model_response_input:model_input_,
                                                                  train_model.refrence_response_input:refrence_input_,
                                                                  train_model.model_sequence_length:model_sequence_length_,
                                                                  train_model.refrence_sequence_length:refrence_sequence_length_,
                                                                  train_model.human_score:human_score_,
                                                                  train_model.grad_ys:grad_ys_
                                                                       })
            if i % 100 == 0 and i != 0:
                context_input_, refrence_input_, model_input_, context_sequence_length_, \
                refrence_sequence_length_, model_sequence_length_, human_score_,grad_ys_= dh.get_val_data()
                model_score=train_model.predict_on_batch(sess,feed_dict={train_model.context_input:context_input_,
                                                                  train_model.context_sequence_length:context_sequence_length_,
                                                                  train_model.model_response_input:model_input_,
                                                                  train_model.refrence_response_input:refrence_input_,
                                                                  train_model.model_sequence_length:model_sequence_length_,
                                                                  train_model.refrence_sequence_length:refrence_sequence_length_,
                                                                       })
                loss = train_model.mean_square_error(human_score_, model_score)
                if loss < min_:
                    min_ = loss
                    saver.save(sess, CHECKPOINT_PATH)
                marks.append([i, loss])
                for k in marks:
                    print(k)
                print(config)

if __name__=='__main__':
    main()
