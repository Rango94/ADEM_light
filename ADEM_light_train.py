#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : ADEM_light_train.py
# @Author: nanzhi.wang
# @Date  : 2018/9/17

import sys

from ADEM_light import *
from data_helper import *

def tostring(config):
    keys_ = config.keys()
    keys_.sort()
    return '-'.join([key + '_' + str(config[key]) for key in keys_])

def Rename(config_old,config_new):
    model_name_old = tostring(config_old)
    model_name_new = tostring(config_new)
    if model_name_new==model_name_old:
        return 0
    else:
        for file in os.listdir('../MODEL/without_attention'):
            print(file)
            if model_name_old in file:
                print '将%s改为%s'%(file,file.replace(model_name_old,model_name_new))
                os.rename('../MODEL/without_attention/'+file, '../MODEL/without_attention/'+file.replace(model_name_old,model_name_new))

config={'score_style':'mine','normal':True,'LR':1,'cate':'mlut','weight':True,'data':'all','seg':'jieba','prewordembedding':False,'attflag':True}

if len(sys.argv)==2:
    if sys.argv[1]=='1':
        config={'score_style':'mine','normal':True,'LR':0.2,'cate':'mlut','weight':True,'data':'all','seg':'jieba','prewordembedding':False,'attflag':True}
    elif sys.argv[1]=='2':
        config={'score_style':'mine','normal':True,'LR':0.2,'cate':'mlut','weight':True,'data':'all','seg':'nio','prewordembedding':False,'attflag':True}
    elif sys.argv[1]=='3':
        config={'score_style':'mine','normal':True,'LR':0.2,'cate':'mlut','weight':True,'data':'all','seg':'jieba','prewordembedding':True,'attflag':True}
    elif sys.argv[1] == '4':
        config={'score_style':'mine','normal':True,'LR':0.2,'cate':'mlut','weight':True,'data':'all','seg':'nio','prewordembedding':True,'attflag':True}

if len(sys.argv)==2 and (sys.argv[1]=='all' or sys.argv[1]=='8' or sys.argv[1]=='9' or sys.argv[1]=='orgin'):
    config['data']=sys.argv[1]

def main():
    model_name = tostring(config)
    CHECKPOINT_PATH = '../MODEL/' + model_name + '_ckpt'
    if os.path.exists(CHECKPOINT_PATH + '.index'):
        exists_flag = True
    else:
        exists_flag = False

    dh = data_helper(config=config)

    config_network = {
        'HIDDEN_SIZE': 128,
        'NUM_LAYERS': 1,
        'SRC_VOCAB_SIZE': dh.vocab_size,
        'BARCH_SIZE': 100,
        'NUM_EPOCH': 5,
        'KEEP_PROB': 0.8,
        'MAX_GRAD_NORM': 5,
        'word_embedding_file': 'word_dic_jieba_embedding.pk' if config[
                                                                    'seg'] == 'jieba' else 'word_dic_nioseg_embedding.pk'
    }
    train_model=ADEM_model(config, config_network)
    saver=tf.train.Saver()
    step=0
    max_loop=9999999
    min_=9999
    marks=[]

    config_tf = tf.ConfigProto()
    config_tf.gpu_options.per_process_gpu_memory_fraction = 0.5
    config_tf.gpu_options.allow_growth = True

    with tf.Session(config=config_tf) as sess:

        tf.global_variables_initializer().run()

        if exists_flag:
            print('loading trained model')
            saver = tf.train.Saver()
            saver.restore(sess, CHECKPOINT_PATH)

        for i in range(max_loop):
            context_input_, refrence_input_, model_input_, context_sequence_length_, \
            refrence_sequence_length_,model_sequence_length_,human_score_,grad_ys_=dh.next_batch(300)

            step=train_model.train_on_batch(sess,step,feed_dict_={'context_input':context_input_,
                                                                       'context_sequence_length':context_sequence_length_,
                                                                       'model_input':model_input_,
                                                                       'refrence_input':refrence_input_,
                                                                       'model_sequence_length':model_sequence_length_,
                                                                       'refrence_sequence_length':refrence_sequence_length_,
                                                                       'human_score':human_score_,
                                                                       'grad_ys':grad_ys_
                                                                       })

            if i % 100 == 0 and i!=0:
                context_input_, refrence_input_, model_input_, context_sequence_length_, \
                refrence_sequence_length_, model_sequence_length_, human_score_,grad_ys_= dh.get_val_data()
                print(context_input_.shape, refrence_input_.shape, model_input_.shape,
                      context_sequence_length_.shape, refrence_sequence_length_.shape,
                      model_sequence_length_.shape, human_score_.shape)
                model_score=train_model.predict_on_batch(sess,feed_dict_={'context_input':context_input_,
                                                                       'context_sequence_length':context_sequence_length_,
                                                                       'model_input':model_input_,
                                                                       'refrence_input':refrence_input_,
                                                                       'model_sequence_length':model_sequence_length_,
                                                                       'refrence_sequence_length':refrence_sequence_length_,
                                                                       'human_score':human_score_,
                                                                       'grad_ys':grad_ys_
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
