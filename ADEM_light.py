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

        self.build_graph()

    def build_graph(self):

        self.context_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='context_input')
        self.model_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='context_input')
        self.refrence_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='context_input')

        self.context_sequence_length = tf.placeholder(dtype=tf.int32, shape=[None], name='context_sequence_length')
        self.model_sequence_length = tf.placeholder(dtype=tf.int32, shape=[None], name='context_sequence_length')
        self.refrence_sequence_length = tf.placeholder(dtype=tf.int32, shape=[None], name='context_sequence_length')

        self.grad_ys = tf.placeholder(dtype=tf.float32, shape=[None], name='grad_ys')
        self.human_score = tf.placeholder(dtype=tf.float32, shape=[None], name='human_score')

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
                self.E = tf.constant(math.e, name='E')
                self.attention_weight = tf.get_variable(name='att_weight', shape=[self.HIDDEN_SIZE * self.NUM_LAYERS,
                                                                                  self.HIDDEN_SIZE * self.NUM_LAYERS])
                self.att_context = tf.get_variable(name='att_context', shape=[1,self.HIDDEN_SIZE * self.NUM_LAYERS])
                self.att_model_res = tf.get_variable(name='model_res', shape=[1,self.HIDDEN_SIZE * self.NUM_LAYERS])
                self.att_refrence_res = tf.get_variable(name='refrence_res', shape=[1,self.HIDDEN_SIZE * self.NUM_LAYERS])

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

        context_state_h, model_state_h, refrence_state_h=self.forward(self.context_input, self.context_sequence_length, self.model_input,
                self.model_sequence_length,self.refrence_input,
                self.refrence_sequence_length)

        if self.config['score_style']=='mine':
            self.model_score=self.compute_score_mine(context_state_h, model_state_h, refrence_state_h)
        elif self.config['score_style']=='adem':
            self.model_score = self.compute_score_adem(context_state_h, model_state_h, refrence_state_h)
        else:
            print('flag error')
            return 0

        self.loss=self.compute_loss(self.model_score,self.human_score,self.grad_ys,self.config['normal'])

        self.train_op=self._train(self.loss)


    def forward(self, context_input, context_size, model_response_input,
                model_response_size,refrence_response_input,
                refrence_response_size):

        context_emb=tf.nn.embedding_lookup(self.embedding, context_input)
        model_response_emb=tf.nn.embedding_lookup(self.embedding, model_response_input)
        refrence_response_emb = tf.nn.embedding_lookup(self.embedding, refrence_response_input)

        context_emb=tf.nn.dropout(context_emb,self.KEEP_PROB)
        model_response_emb=tf.nn.dropout(model_response_emb,self.KEEP_PROB)
        refrence_response_emb=tf.nn.dropout(refrence_response_emb,self.KEEP_PROB)

        with tf.variable_scope('context_encoder'):
            context_output,context_state=tf.nn.dynamic_rnn(self.context_cell, context_emb, sequence_length=context_size, dtype=tf.float32)

        with tf.variable_scope('model_encoder'):
            model_output,model_state=tf.nn.dynamic_rnn(self.model_cell, model_response_emb, sequence_length=model_response_size, dtype=tf.float32)

        with tf.variable_scope('refrence_encoder'):
            refrence_output,refrence_state = tf.nn.dynamic_rnn(self.refrence_cell, refrence_response_emb, sequence_length=refrence_response_size,dtype=tf.float32)

        with tf.variable_scope('handle_output'):

            if self.attention_flag:
                context_state_h = self.attetion(self.att_context,context_output,context_size)
                model_state_h = self.attetion(self.att_model_res,model_output,model_response_size)
                refrence_state_h = self.attetion(self.att_refrence_res,refrence_output,refrence_response_size)
            else:
                context_state_h=self.get_h(context_state)
                model_state_h=self.get_h(model_state)
                refrence_state_h=self.get_h(refrence_state)

        return context_state_h,model_state_h,refrence_state_h


    def compute_loss(self, model_score,human_score, grad_ys,normal):

        human_score = tf.reshape(human_score, shape=[-1, 1])
        model_score = tf.reshape(model_score, shape=[-1, 1])
        grad_ys = tf.reshape(grad_ys, shape=[-1, 1])

        [human_score, model_score] = self.cast_to_float32([human_score, model_score])

        if normal:
            if self.config['score_style']=='adem':
                regularization = self.matrix_l1_norm(self.M) + self.matrix_l1_norm(self.N)
                gamma = tf.constant(0.15, name='gamma')
                loss = tf.reduce_sum(tf.square(human_score - model_score) * grad_ys) + (gamma * regularization)
            elif self.config['score_style']=='mine':
                regularization = self.matrix_l1_norm(self.w1) + self.matrix_l1_norm(self.w2)
                gamma = tf.constant(0.15, name='gamma')
                loss = tf.reduce_sum(tf.square(human_score - model_score) * grad_ys) + (gamma * regularization)
            else:
                loss = tf.reduce_sum(tf.square(human_score - model_score) * grad_ys)
        else:
            loss = tf.reduce_sum(tf.square(human_score - model_score) * grad_ys)
        return loss


    def _train(self,loss):
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(loss, trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.MAX_GRAD_NORM)
        opt = tf.train.AdadeltaOptimizer(learning_rate=self.config['LR'])
        train_op = opt.apply_gradients(zip(grads, trainable_variables))
        return train_op

    def train_on_batch(self, session, step, feed_dict_):

        feed_dict = {self.context_input: feed_dict_['context_input'],
                     self.context_sequence_length: feed_dict_['context_sequence_length'],
                     self.model_input: feed_dict_['model_input'],
                     self.refrence_input: feed_dict_['refrence_input'],
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
                     self.model_input: feed_dict_['model_input'],
                     self.refrence_input: feed_dict_['refrence_input'],
                     self.model_sequence_length: feed_dict_['model_sequence_length'],
                     self.refrence_sequence_length: feed_dict_['refrence_sequence_length']
                     # self.human_score: feed_dict_['human_score']
                     }
        model_score = session.run(self.model_score, feed_dict=feed_dict)
        return model_score

    '''
    采用的相似度计算方法：
    a=z0*M*h
    
    '''

    # tmp = tf.matmul(att_, self.attention_weight)
    #
    # tmp = tf.matmul(tf.reshape(state, shape=[-1, self.HIDDEN_SIZE]), tf.reshape(tmp, shape=[self.HIDDEN_SIZE, 1]))
    #
    # tmp = tf.reshape(tmp, [-1, 18])
    #
    # tmp = tmp * self.genarate_mask(length, 18)
    #
    # tmp = tf.nn.softmax(tmp)
    #
    # tmp = tf.matmul(tf.reshape(tmp, shape=[-1, 1, 18]), state)


    def attetion(self,att_,state,length):
        return tf.reshape(
            tf.matmul(
            tf.reshape(
            tf.nn.softmax(
            tf.reshape(
            tf.matmul(
            tf.reshape(
            state, shape=[-1, self.HIDDEN_SIZE]),
            tf.reshape(
            tf.matmul(
            att_, self.attention_weight), shape=[self.HIDDEN_SIZE, 1])), [-1, self.max_len]) *
            tf.sequence_mask(
            length, self.max_len, dtype=tf.float32)), shape=[-1, 1, self.max_len]), state),shape=[-1,self.HIDDEN_SIZE])



    def genarate_mask(self,length,max_len):
        return



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

    def compute_score_mine(self, context_state_h, model_state_h, refrence_state_h):
        reshaped = tf.concat([context_state_h, model_state_h], 1)
        reshaped = tf.concat([reshaped, refrence_state_h], 1)
        a=tf.nn.relu(tf.matmul(reshaped,self.w1))+self.bais1
        b=tf.nn.relu(tf.matmul(a,self.w2))+self.bais2
        model_score=tf.matmul(b,self.w3)
        return model_score


    def compute_score_adem(self, context_state_h, model_state_h, refrence_state_h):
        model_score = (tf.reduce_sum((context_state_h * tf.matmul(model_state_h, self.M)), axis=1) +
                 tf.reduce_sum((refrence_state_h * tf.matmul(model_state_h, self.N)), axis=1) -
                 self.Alpha) / self.Beta
        return model_score

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


