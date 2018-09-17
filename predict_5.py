
#coding:UTF-8
import tensorflow as tf

from ADEM.model_adem_with_encoder import *
from ADEM_light_code.sta_helper import *
from DATA_orgin.data_helper import *
#这个带有encoder的ADEM模型才是我们需要的，因为我们在上下文表示和答案表示那部分并没有训练模型
import matplotlib.pyplot as plt

CHECKPOINT_PATH='./model/mult_True_old_adem_ckpt'


class Model():

    def __init__(self):
        #上下文encoder
        #train_file='./DATA_orgin/corpus_normal_random_train_idx_1.2',test_file='./DATA_orgin/corpus_normal_random_test_idx_1.2'

        self.train_file='./DATA_orgin/corpus_normal_random_train_idx'

        self.test_file='./DATA_orgin/corpus_normal_random_test_idx'

        self.dp = data_helper(train_file=self.train_file, test_file=self.test_file,cate='mult',weight=True)

        self.context_encoder = {
            'name': 'lstm_context_encoder',
            'params': {'utterence_level_state_size': 128,
                       'utterence_level_keep_proba': 1,
                       'utterence_level_num_layers': 1,
                       'context_level_state_size': 128,
                       'context_level_keep_proba': 1,
                       'context_level_num_layers': 1}}

        #模型预测值encoder
        self.model_response_encoder = {
            'name': 'lstm_context_encoder',
            'params': {'utterence_level_state_size': 128,
                       'utterence_level_keep_proba': 1,
                       'utterence_level_num_layers': 1,
                       'context_level_state_size': 128,
                       'context_level_keep_proba': 1,
                       'context_level_num_layers': 1}}

        #参考答案encoder
        self.reference_response_encoder = {
            'name': 'lstm_context_encoder',
            'params': {'utterence_level_state_size': 128,
                       'utterence_level_keep_proba': 1,
                       'utterence_level_num_layers': 1,
                       'context_level_state_size': 128,
                       'context_level_keep_proba': 1,
                       'context_level_num_layers': 1}}

        self.embedding_lut_path = 'word_embedding.pk'
        self.vocab_size = self.dp.vocab_size
        self.embedding_size = 100
        self.learn_embedding = True
        self.learning_rate = 0.2
        self.max_grad_norm = 5


    def looup(self,human_score,pretict_score):
        print('\n',end='')
        k=0
        for i,j in zip(human_score,pretict_score):
            if k>50:
                break
            k+=1
            print(i,j)

    def wrtie_in_file(self,filename,predict_s,batch_size):
        outfo = open(filename + '_inc_pre', 'w', encoding='utf-8')
        filename='./DATA_orgin/'+filename
        n=0
        for line in open(filename,'r',encoding='utf-8'):
            print(line.strip()+'\t'+str(predict_s[n]))
            outfo.write(line.strip()+'\t'+str(predict_s[n])+'\n')
            n+=1
            if n==batch_size:
                break


    def five_to_2(self,human_score,predict_score):

        for idx,i in enumerate(human_score):
            human_score[idx]=0 if i<=2 else 1

        for idx,i in enumerate(predict_score):
            predict_score[idx]=max(0,min((i-0)/4,1))

        self.looup(human_score, predict_score)
        baseline_rd, baseline_mean = baseline_auc(human_score.copy())
        print('baseline_rd:', baseline_rd)
        print('baseline_mean:', baseline_mean)

        auc = Auc(human_score.copy(), predict_score.copy(), True)
        print('auc', auc)
        print('preall:', score_int(predict_score.copy(), human_score.copy(), yuzhi=auc[1]))
        recall_and_pre(human_score.copy(), predict_score.copy(), yuzhi=auc[1], num=2)
        # 排序&&归一
        human_score, predict_score = sort(human_score.copy(), predict_score.copy())

        # 趋势图
        x = np.array([i for i in range(len(human_score))])
        y1 = human_score
        y2 = predict_score
        plt.scatter(x, y1, s=1)
        plt.scatter(x, y2, s=1)
        plt.show()

        plt.scatter(y2, y1, s=1)
        plt.xlabel('model_score')
        plt.ylabel('human_score')
        plt.show()

        predict_score = normal(predict_score)
        x = np.array([i for i in range(len(human_score))])
        y1 = human_score
        y2 = predict_score
        plt.scatter(x, y1, s=1)
        plt.scatter(x, y2, s=1)
        plt.show()


    def prediction(self):

        tf.reset_default_graph()

        with tf.Session() as sess:
            model = ADEMWithEncoder(
                self.vocab_size, self.embedding_size,
                self.context_encoder, self.model_response_encoder,
                self.reference_response_encoder, self.embedding_lut_path,
                self.learn_embedding, self.learning_rate, self.max_grad_norm,False)
            saver = tf.train.Saver()
            saver.restore(sess,CHECKPOINT_PATH)

            batch_size=10000
            context, reference_response, model_response, context_mask, reference_response_mask, model_response_mask, human_score,grads_wt = self.dp.get_test_data()
            # print(context, reference_response, model_response, context_mask.shape, reference_response_mask.shape, model_response_mask.shape, human_score.shape)
            predict_score,loss=model.predict_on_single_batch(
                sess,
                context=context,
                model_response=model_response,
                reference_response=reference_response,
                context_mask=context_mask,
                model_response_mask=model_response_mask,
                reference_response_mask=reference_response_mask,
                human_score=human_score,
                grads_wt=grads_wt
            )

            '''
            0 1.8121926
            4 2.6640408
            2 1.8455334
            3 4.638817
            0 3.0323699
            3 1.9427468
            3 1.4852211
            3 1.5767825
            3 1.5828276
            0 3.826309
            2 0.3906038
            3 1.5975298
            3 2.796331
            3 2.8464959
            3 2.8151202
            3 2.7620142
            3 2.1617484
            3 2.4619799
            4 3.3620286
            2 2.3893974
            3 2.5811157
            4 3.1808114
            3 3.7451615
            4 2.2440903
            4 3.2216673
            3 2.7560394
            3 2.9316342
            1 1.5007926
            3 1.5497361
            0 0.9809722
            0 0.489291
            4 2.696309
            4 4.0056906
            3 2.7586572
            0 2.4177597
            2 2.5925455
            3 4.170372
            3 3.6090806
            2 1.9310045
            3 2.8816924
            3 2.3652754
            4 1.2664548
            3 3.4385424
            4 1.8087312
            2 2.4913275
            3 2.107152
            3 3.0900922
            4 2.4833841
            3 2.0976334
            3 1.8983685
            0 5.1366525
            baseline_rd: 2.006419666219381
            baseline_mean: 1.3118255063297246
            mean_square_error 1.5002180547527022
            '''
            # self.five_to_2(human_score,predict_score)
            # self.wrtie_in_file('corpus_normal_random_test_1.2',predict_score,batch_size)

            self.looup(human_score,predict_score)
            baseline_rd, baseline_mean = baseline(human_score)

            print('baseline_rd:', baseline_rd)
            print('baseline_mean:', baseline_mean)
            print('mean_square_error', mean_square_error(predict_score, human_score))
            print('preall:', score_int(predict_score, human_score))
            recall_and_pre(human_score, predict_score)
            # 排序&&归一
            human_score, predict_score = sort(human_score, predict_score)

            # 趋势图
            x = np.array([i for i in range(len(human_score))])
            y1 = human_score
            y2 = predict_score
            plt.scatter(x, y1, s=2)
            plt.scatter(x, y2, s=2)
            plt.show()

            plt.scatter(y2, y1, s=2)
            plt.xlabel('model_score')
            plt.ylabel('human_score')
            plt.show()

            predict_score = normal(predict_score)
            x = np.array([i for i in range(len(human_score))])
            y1 = human_score
            y2 = predict_score
            plt.scatter(x, y1, s=2)
            plt.scatter(x, y2, s=2)
            plt.show()

a=Model()
a.prediction()


