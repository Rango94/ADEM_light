#coding:UTF-8
from sta_helper import *
import matplotlib.pyplot as plt
from ADEM_light import *
from data_helper import *
import codecs

def write_in_file(file,score,word_dic):
    with codecs.open(file,'r','utf-8') as fo:
        with codecs.open(file+'_marked_'+word_dic,'w','utf-8') as out:
            for idx,line in enumerate(fo):
                out.write('\t'.join(line.rstrip().split('\t')[:3])+'\t'+str(min(max(0,score[idx]),4))+'\n')
    os.system("sort -t $'\t' -k4 -g -r ../DATA_test/"+file+"_marked_"+word_dic+" > ../DATA_test/tmp")
    os.system("mv ../DATA_test/tmp ../DATA_test/"+file+"_marked_"+word_dic)

def predict_on_file(sess,file,word_dic):
    context_input, refrence_input, model_input, context_sequence_length, \
    refrence_sequence_length, model_sequence_length= dp.get_specific_data(file+'_idx_'+word_dic)
    print(context_input.shape, refrence_input.shape, model_input.shape,
          context_sequence_length.shape, refrence_sequence_length.shape,
          model_sequence_length.shape)

    predict_score = train_model.predict_on_batch(sess, feed_dict_={'context_input': context_input,
                                                                   'context_sequence_length': context_sequence_length,
                                                                   'model_input': model_input,
                                                                   'refrence_input': refrence_input,
                                                                   'model_sequence_length': model_sequence_length,
                                                                   'refrence_sequence_length': refrence_sequence_length,
                                                                   })

    std_score = np.zeros(len(predict_score))
    print(mean_square_error(predict_score,std_score))
    predict_score = np.reshape(predict_score, [len(predict_score)])
    write_in_file(file, predict_score,word_dic)


data_flag='all'

CHECKPOINT_PATH='../MODEL/LR_1-attflag_True-cate_mlut-data_'+data_flag+'-normal_True-prewordembedding_False-score_style_mine-seg_jieba-weight_True_ckpt'

# print(resolve_filename(CHECKPOINT_PATH))
config=resolve_filename(CHECKPOINT_PATH)
print(config)

# if config['data']=='orgin':
#     from DATA_orgin.data_helper import *
# elif config['data']=='8':
#     sys.path.append('./DATA_8')
#     from DATA_8.data_helper import *
# elif config['data']=='9':
#     sys.path.append('./DATA_9')
#     from DATA_9.data_helper import *
# elif config['data']=='total':
#     sys.path.append('./DATA_total')
#     from DATA_total.data_helper import *

dp=data_helper(config)

config_network={
   'HIDDEN_SIZE':128,
    'NUM_LAYERS':1,
    'SRC_VOCAB_SIZE':dp.vocab_size,
    'BARCH_SIZE':100,
    'NUM_EPOCH':5,
    'KEEP_PROB':0.8,
    'MAX_GRAD_NORM':5,
    'word_embedding_file':'word_dic_jieba_embedding.pk' if config['seg']=='jieba' else 'word_dic_nioseg_embedding.pk'
}




if __name__=='__main__':

    with tf.Session() as sess:
        train_model = ADEM_model(config, config_network)
        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        saver.restore(sess, CHECKPOINT_PATH)

        word_dic=config['seg'] if not config['seg'] =='nio' else 'nioseg'


        predict_on_file(sess,'../DATA_test/random_test_data',word_dic)

        predict_on_file(sess, '../DATA_test/external_test_data', word_dic)

        # context_input, refrence_input, model_input, context_sequence_length, \
        # refrence_sequence_length, model_sequence_length, human_score, grad_ys = dp.get_test_data()
        #
        # print(context_input.shape, refrence_input.shape, model_input.shape,
        #       context_sequence_length.shape, refrence_sequence_length.shape,
        #       model_sequence_length.shape, human_score.shape)
        #
        # predict_score = train_model.predict_on_batch(sess, feed_dict_={'context_input': context_input,
        #                                                                'context_sequence_length': context_sequence_length,
        #                                                                'model_input': model_input,
        #                                                                'refrence_input': refrence_input,
        #                                                                'model_sequence_length': model_sequence_length,
        #                                                                'refrence_sequence_length': refrence_sequence_length,
        #                                                                'human_score': human_score,
        #                                                                })
        #
        # len_ = len(human_score)
        # human_score = np.reshape(human_score, [len_])
        # predict_score = np.reshape(predict_score, [len_])
        #
        # looup(human_score, predict_score)
        # baseline_rd, baseline_mean = baseline(human_score)
        #
        # print('baseline_rd:', baseline_rd)
        # print('baseline_mean:', baseline_mean)
        # rmse=mean_square_error(predict_score, human_score)
        # print('mean_square_error', rmse)
        # print('preall:', score_int(predict_score, human_score))
        # recall_and_pre(human_score, predict_score)
        # # 排序&&归一
        # human_score, predict_score = sort(human_score, predict_score)
        #
        # # 趋势图
        # x = np.array([i for i in range(len(human_score))])
        #
        # y1 = human_score
        # y2 = predict_score
        # plt.scatter(x, y1, s=2)
        # plt.scatter(x, y2, s=2)
        # plt.title('data:' + data_flag + ' | rmse:' + '%.4f' % (rmse))
        # plt.show()
        #
        # plt.scatter(y2, y1, s=2)
        # plt.xlabel('model_score')
        # plt.ylabel('human_score')
        # plt.title('data:' + data_flag + ' | rmse:' + '%.4f' % (rmse))
        # plt.show()
        #
        # predict_score = normal(predict_score)
        # x = np.array([i for i in range(len(human_score))])
        # y1 = human_score
        # y2 = predict_score
        # plt.scatter(x, y1, s=2)
        # plt.scatter(x, y2, s=2)
        # plt.title('data:' + data_flag + ' | rmse:' + '%.4f' % (rmse))
        # plt.show()
