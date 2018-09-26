#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : ADEM_light.py
# @Author: nanzhi.wang
# @Date  : 2018/9/17
'''
policy grad
'''
import tensorflow as tf
import math
import numpy as np
import cPickle
import jieba as jb
import json
import codecs

class ADEM_model(object):

    def __init__(self, config, config_network):
        self.config=config
        self.config_network=config_network
        self.HIDDEN_SIZE=config_network['HIDDEN_SIZE']
        self.NUM_LAYERS=config_network['NUM_LAYERS']
        self.SRC_VOCAB_SIZE=config_network['SRC_VOCAB_SIZE']
        self.BARCH_SIZE=config_network['BARCH_SIZE']
        self.NUM_EPOCH=config_network['NUM_EPOCH']
        self.KEEP_PROB=config_network['KEEP_PROB']
        self.MAX_GRAD_NORM=config_network['MAX_GRAD_NORM']
        self.attention_flag=config['attflag']
        self.max_len=18
        self.normal=config['normal']

        self.build_graph()


    def build_graph(self):
        self.build_placeholder()
        self.build_variable()
        self.forward()

        if self.config['score_style']=='mine':
            self.compute_score_mine()
        elif self.config['score_style']=='adem':
            self.compute_score_adem()
        else:
            print('flag error')
            return 0

        self.compute_loss()
        self._train()

    def build_placeholder(self):
        self.context_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='context_input')
        self.model_response_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='context_input')
        self.refrence_response_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='context_input')

        self.context_sequence_length = tf.placeholder(dtype=tf.int32, shape=[None], name='context_sequence_length')
        self.model_sequence_length = tf.placeholder(dtype=tf.int32, shape=[None], name='context_sequence_length')
        self.refrence_sequence_length = tf.placeholder(dtype=tf.int32, shape=[None], name='context_sequence_length')

        self.grad_ys = tf.placeholder(dtype=tf.float32, shape=[None], name='grad_ys')
        self.human_score = tf.placeholder(dtype=tf.float32, shape=[None], name='human_score')


    def build_variable(self):
        initializer = tf.random_uniform_initializer(-0.5, 0.5)
        with tf.variable_scope('nmt_model', reuse=None, initializer=initializer):
            if not self.config['prewordembedding']:
                self.embedding = tf.get_variable('emb', [self.SRC_VOCAB_SIZE, self.HIDDEN_SIZE])
            else:
                print('loading pretrained word embedding')
                embedding_np = cPickle.load(
                    open('../PRE_WORD_EMBEDDING/' + self.config_network['word_embedding_file'], "rb"))
                print('loaded')
                self.HIDDEN_SIZE = len(embedding_np[0])
                self.embedding = tf.get_variable('emb',
                                                 initializer=tf.convert_to_tensor(embedding_np, dtype=tf.float32),
                                                 trainable=True)

            self.context_cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.BasicLSTMCell(self.HIDDEN_SIZE) for _ in range(self.NUM_LAYERS)])
            self.model_cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.BasicLSTMCell(self.HIDDEN_SIZE) for _ in range(self.NUM_LAYERS)])
            self.refrence_cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.BasicLSTMCell(self.HIDDEN_SIZE) for _ in range(self.NUM_LAYERS)])

            with tf.variable_scope('attention'):
                self.attention_weight = tf.get_variable(name='att_weight', shape=[self.HIDDEN_SIZE * self.NUM_LAYERS,
                                                                                  self.HIDDEN_SIZE * self.NUM_LAYERS])

            with tf.variable_scope('output_layer'):
                self.w1 = tf.get_variable(name='w1', shape=[self.HIDDEN_SIZE * 3 * self.NUM_LAYERS, 256])
                self.bais1 = tf.get_variable(name='bais1', shape=[1, 256])
                self.w2 = tf.get_variable(name='w2', shape=[256, 256])
                self.bais2 = tf.get_variable(name='bais2', shape=[1, 256])
                self.w3 = tf.get_variable(name='w3', shape=[256, 1])
                self.bais3 = tf.get_variable(name='bais3', shape=[1])

                self.N = tf.Variable(name='n', initial_value=tf.random_normal(
                    shape=[self.HIDDEN_SIZE * self.NUM_LAYERS, self.HIDDEN_SIZE * self.NUM_LAYERS]))
                self.M = tf.Variable(name='m', initial_value=tf.random_normal(
                    shape=[self.HIDDEN_SIZE * self.NUM_LAYERS, self.HIDDEN_SIZE * self.NUM_LAYERS]))
                self.Alpha = tf.Variable(
                    name='alpha', initial_value=tf.random_uniform([1], maxval=5.0))
                self.Beta = tf.Variable(
                    name='beta', initial_value=tf.random_uniform([1], maxval=5.0))


    def forward(self):

        context_emb=tf.nn.embedding_lookup(self.embedding, self.context_input)
        model_response_emb=tf.nn.embedding_lookup(self.embedding, self.model_response_input)
        refrence_response_emb = tf.nn.embedding_lookup(self.embedding, self.refrence_response_input)

        context_emb=tf.nn.dropout(context_emb,self.KEEP_PROB)
        model_response_emb=tf.nn.dropout(model_response_emb,self.KEEP_PROB)
        refrence_response_emb=tf.nn.dropout(refrence_response_emb,self.KEEP_PROB)

        with tf.variable_scope('context_encoder'):
            context_output,context_state=tf.nn.dynamic_rnn(self.context_cell, context_emb, sequence_length=self.context_sequence_length, dtype=tf.float32)

        with tf.variable_scope('model_encoder'):
            model_output,model_state=tf.nn.dynamic_rnn(self.model_cell, model_response_emb, sequence_length=self.model_sequence_length, dtype=tf.float32)

        with tf.variable_scope('refrence_encoder'):
            refrence_output,refrence_state = tf.nn.dynamic_rnn(self.refrence_cell, refrence_response_emb, sequence_length=self.refrence_sequence_length,dtype=tf.float32)

        with tf.variable_scope('handle_output'):

            if self.attention_flag:
                self.context_state_h = self.get_h(context_state)
                self.model_state_h = self.attetion(self.context_state_h,model_output,self.model_sequence_length)
                self.refrence_state_h = self.attetion(self.context_state_h,refrence_output,self.refrence_sequence_length)
            else:
                self.context_state_h=self.get_h(context_state)
                self.model_state_h=self.get_h(model_state)
                self.refrence_state_h=self.get_h(refrence_state)

    def compute_score_mine(self):
        reshaped = tf.concat([self.context_state_h, self.model_state_h], 1)
        reshaped = tf.concat([reshaped, self.refrence_state_h], 1)
        a=tf.nn.relu(tf.matmul(reshaped,self.w1))+self.bais1
        b=tf.nn.relu(tf.matmul(a,self.w2))+self.bais2
        self.model_score=tf.matmul(b,self.w3)

    def compute_score_adem(self):
        self.model_score = (tf.reduce_sum((self.context_state_h * tf.matmul(self.model_state_h, self.M)), axis=1) +
                 tf.reduce_sum((self.refrence_state_h * tf.matmul(self.model_state_h, self.N)), axis=1) -
                 self.Alpha) / self.Beta

    def compute_loss(self):
        human_score = tf.reshape(self.human_score, shape=[-1, 1])
        model_score = tf.reshape(self.model_score, shape=[-1, 1])
        grad_ys = tf.reshape(self.grad_ys, shape=[-1, 1])
        [human_score, model_score] = self.cast_to_float32([human_score, model_score])

        if self.normal:
            if self.config['score_style']=='adem':
                regularization = self.matrix_l1_norm(self.M) + self.matrix_l1_norm(self.N)
                gamma = tf.constant(0.15, name='gamma')
                self.loss = tf.reduce_sum(tf.square(human_score - model_score) * grad_ys) + (gamma * regularization)
            elif self.config['score_style']=='mine':
                regularization = self.matrix_l1_norm(self.w1) + self.matrix_l1_norm(self.w2)
                gamma = tf.constant(0.15, name='gamma')
                self.loss = tf.reduce_sum(tf.square(human_score - model_score) * grad_ys) + (gamma * regularization)
            else:
                self.loss = tf.reduce_sum(tf.square(human_score - model_score) * grad_ys)
        else:
            self.loss = tf.reduce_sum(tf.square(human_score - model_score) * grad_ys)


    def _train(self):
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.MAX_GRAD_NORM)
        opt = tf.train.AdadeltaOptimizer(learning_rate=self.config['LR'])
        self.train_op = opt.apply_gradients(zip(grads, trainable_variables))

    def train_on_batch(self, session, step, feed_dict_):
        feed_dict = {self.context_input: feed_dict_['context_input'],
                     self.context_sequence_length: feed_dict_['context_sequence_length'],
                     self.model_response_input: feed_dict_['model_input'],
                     self.refrence_response_input: feed_dict_['refrence_input'],
                     self.model_sequence_length: feed_dict_['model_sequence_length'],
                     self.refrence_sequence_length: feed_dict_['refrence_sequence_length'],
                     self.human_score: feed_dict_['human_score'],
                     self.grad_ys: feed_dict_['grad_ys']
                     }
        loss, _ = session.run([self.loss, self.train_op], feed_dict=feed_dict)
        if step % 100 == 0 and step != 0:
            print('After %d steps,per token loss is %.3f' % (step, loss))
        step += 1
        return step

    def predict_on_batch(self, session, feed_dict_):
        feed_dict = {self.context_input: feed_dict_['context_input'],
                     self.context_sequence_length: feed_dict_['context_sequence_length'],
                     self.model_response_input: feed_dict_['model_input'],
                     self.refrence_response_input: feed_dict_['refrence_input'],
                     self.model_sequence_length: feed_dict_['model_sequence_length'],
                     self.refrence_sequence_length: feed_dict_['refrence_sequence_length']
                     }
        model_score = session.run(self.model_score, feed_dict=feed_dict)
        return model_score

    '''
    根据context final state分别计算其与refrence response和model response的attention，将attention的结果作为最终编码。
    采用的相似度计算方法：
    Alpha = h ✖ M ✖ context_state.T
    步骤分解如下：
        # tmp = tf.matmul(context_state, self.attention_weight)
        # tmp = tf.matmul(outputs, tf.reshape(tmp, shape=[-1, self.HIDDEN_SIZE, 1]))
        # tmp = tf.reshape(tmp, [-1, 18])
        # tmp = tmp * tf.sequence_mask(length, self.max_len, dtype=tf.float32)
        # tmp = tf.nn.softmax(tmp)
        # tmp = tf.matmul(tf.reshape(tmp, shape=[-1, 1, 18]), outputs)
    '''
    def attetion(self, context_state, outputs, length):
        return tf.reshape(
            tf.matmul(
                tf.reshape(
                    tf.nn.softmax(
                        tf.reshape(
                            tf.matmul(
                                outputs,tf.reshape(
                                    tf.matmul(
                                        context_state, self.attention_weight), shape=[-1, self.HIDDEN_SIZE, 1])), [-1, 18]) * tf.sequence_mask(
                            length, self.max_len, dtype=tf.float32)), shape=[-1, 1, 18]), outputs),shape=[-1, self.HIDDEN_SIZE])


    def matrix_l1_norm(self,matrix):
        matrix = tf.cast(matrix, tf.float32)
        abs_matrix = tf.abs(matrix)
        row_max = tf.reduce_max(abs_matrix, axis=1)
        return tf.reduce_sum(row_max)

    def get_h(self,state):
        c,h=state[0]
        h=tf.reshape(h,[-1,self.HIDDEN_SIZE*self.NUM_LAYERS])
        return h

    def cast_to_float32(self,tensor_list):
        for num, tensor in enumerate(tensor_list):
            tensor_list[num] = tf.cast(tensor, tf.float32)
        return tensor_list

    def mean_square_error(self,human_score,predict_score):
        sum=0
        n=0
        len_=len(human_score)
        human_score=np.reshape(human_score,[len_])
        predict_score=np.reshape(predict_score,[len_])
        for i,j in zip(human_score,predict_score):
            print (i,j)
            n+=1
            if n>50:
                break
        for i,j in zip(human_score,predict_score):
            sum+=math.pow(i-j,2)
        return math.pow(sum/len(human_score),0.5)


    #载入字典文件，输入为字典文件名。
    def set_word_dic(self,word_dic_file):
        self.word_dic = json.load(codecs.open(word_dic_file, 'r', 'utf-8'))
        if 'jieba' in word_dic_file:
            self.dic_flag='jieba'
        elif 'ipx' in word_dic_file:
            self.dic_flag='ipx'
        else:
            self.dic_flag='nio'
        print '字典载入完毕'

    #输入为文字，即时给出预测值，测试用，只有jieba分词，使用之前先使用上一个方法载入字典文件
    def predict_on_line(self,sess,line):
        if len(line.split('\t'))!=3:
            print '格式错误'
        else:
            if self.dic_flag=='jieba':
                context = []
                true_response = []
                model_response = []
                context_mask = []
                model_response_mask = []
                true_response_mask = []
                line = line.strip().split('\t')

                context_tmp = [self.word_dic[i] for i in jb.cut(line[0])]
                true_response_tmp = [self.word_dic[i]  for i in jb.cut(line[1])]
                model_response_tmp = [self.word_dic[i]  for i in jb.cut(line[2])]
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

                return self.predict_on_batch(sess,feed_dict_={'context_input':np.array(context),
                                                              'refrence_input':np.array(true_response),
                                                              'model_input':np.array(model_response),
                                                              'context_sequence_length':np.array(context_mask),
                                                              'refrence_sequence_length':np.array(true_response_mask),
                                                              'model_sequence_length':np.array(model_response_mask)})



