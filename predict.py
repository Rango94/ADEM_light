#coding:UTF-8
from sta_helper import *
#线上机器默认是没有matplotlib，如果要使用需要自行安装或者线下使用。
import matplotlib.pyplot as plt
from ADEM_light import *
from data_helper import *
import codecs

'''
线上机器没有matplotlib，要画图需要线下使用该程序
三种预测方式，batch上预测，从文件读取数据预测，对单条数据预测
'''

def predict_on_batch_demo(CHECKPOINT_PATH):

    config=resolve_filename(CHECKPOINT_PATH)
    dh=data_helper(config)

    config_network = {
        'HIDDEN_SIZE': 256,
        'NUM_LAYERS': 2,
        'SRC_VOCAB_SIZE': dh.vocab_size,
        'KEEP_PROB': 1,
        'MAX_GRAD_NORM': 5,
        'word_embedding_file': 'word_dic_jieba_embedding.pk' if config[
                                                                    'seg'] == 'jieba' else 'word_dic_nioseg_embedding.pk',
        'max_len': dh.max_len
    }

    with tf.Session() as sess:
        train_model = ADEM_model(config, config_network)
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver.restore(sess, CHECKPOINT_PATH)


        context_input, refrence_input, model_input, context_sequence_length, \
        refrence_sequence_length, model_sequence_length, human_score, grad_ys = dh.get_test_data()

        predict_score = train_model.predict_on_batch(sess, feed_dict={train_model.context_input:context_input,
                                                                  train_model.context_sequence_length:context_sequence_length,
                                                                  train_model.model_response_input:model_input,
                                                                  train_model.refrence_response_input:refrence_input,
                                                                  train_model.model_sequence_length:model_sequence_length,
                                                                  train_model.refrence_sequence_length:refrence_sequence_length,
                                                                       })

        len_ = len(human_score)
        human_score = np.reshape(human_score, [len_])
        predict_score = np.reshape(predict_score, [len_])

        looup(human_score, predict_score)
        baseline_rd, baseline_mean = baseline(human_score)

        print('baseline_rd:', baseline_rd)
        print('baseline_mean:', baseline_mean)
        rmse=RMSE(predict_score, human_score)
        print('mean_square_error', rmse)
        print('preall:', score_int(predict_score, human_score))
        recall_and_pre(human_score, predict_score)

        human_score, predict_score = sort(human_score, predict_score)

        # 趋势图
        x = np.array([i for i in range(len(human_score))])

        y1 = human_score
        y2 = predict_score
        plt.scatter(x, y1, s=2)
        plt.scatter(x, y2, s=2)
        plt.title('data:' + config['data'] + ' | rmse:' + '%.4f' % (rmse))
        plt.show()

        plt.scatter(y2, y1, s=2)
        plt.xlabel('model_score')
        plt.ylabel('human_score')
        plt.title('data:' + config['data'] + ' | rmse:' + '%.4f' % (rmse))
        plt.show()

        predict_score = normal(predict_score)
        x = np.array([i for i in range(len(human_score))])
        y1 = human_score
        y2 = predict_score
        plt.scatter(x, y1, s=2)
        plt.scatter(x, y2, s=2)
        plt.title('data:' + config['data'] + ' | rmse:' + '%.4f' % (rmse))
        plt.show()


def predict_on_line_demo(CHECKPOINT_PATH,line=u'我最近好累啊\t要好好休息，知道吗\t知道你过得不好我就放心了'):

    config = resolve_filename(CHECKPOINT_PATH)
    dh = data_helper(config)

    config_network = {
        'HIDDEN_SIZE': 256,
        'NUM_LAYERS': 2,
        'SRC_VOCAB_SIZE': dh.vocab_size,
        'KEEP_PROB': 1,
        'MAX_GRAD_NORM': 5,
        'word_embedding_file': 'word_dic_jieba_embedding.pk' if config[
                                                                    'seg'] == 'jieba' else 'word_dic_nioseg_embedding.pk',
        'max_len': dh.max_len
    }
    with tf.Session() as sess:
        train_model = ADEM_model(config, config_network)
        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        saver.restore(sess, CHECKPOINT_PATH)

        train_model.set_word_dic('word_dic_jieba')
        print(train_model.predict_on_line(sess=sess, line=line))



def predict_on_file_demo(CHECKPOINT_PATH,file_list):
    config = resolve_filename(CHECKPOINT_PATH)
    dh = data_helper(config)
    config_network = {
        'HIDDEN_SIZE': 256,
        'NUM_LAYERS': 2,
        'SRC_VOCAB_SIZE': dh.vocab_size,
        'KEEP_PROB': 1,
        'MAX_GRAD_NORM': 5,
        'word_embedding_file': 'word_dic_jieba_embedding.pk' if config[
                                                                    'seg'] == 'jieba' else 'word_dic_nioseg_embedding.pk',
        'max_len': dh.max_len
    }
    with tf.Session() as sess:
        train_model = ADEM_model(config, config_network)
        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        saver.restore(sess, CHECKPOINT_PATH)

        word_dic=config['seg'] if not config['seg'] =='nio' else 'nioseg'

        for file in file_list:
            context_input, refrence_input, model_input, context_sequence_length, \
            refrence_sequence_length, model_sequence_length = dh.get_specific_data(file + '_idx_' + word_dic)
            predict_score = train_model.predict_on_batch(sess, feed_dict={train_model.context_input:context_input,
                                                                          train_model.context_sequence_length:context_sequence_length,
                                                                          train_model.model_response_input:model_input,
                                                                          train_model.refrence_response_input:refrence_input,
                                                                          train_model.model_sequence_length:model_sequence_length,
                                                                          train_model.refrence_sequence_length:refrence_sequence_length,
                                                                           })
            std_score = np.zeros(len(predict_score))
            print(RMSE(predict_score, std_score))
            predict_score = np.reshape(predict_score, [len(predict_score)])
            write_in_file(file, predict_score, word_dic)

def write_in_file(file,score,word_dic):
    with codecs.open(file,'r','utf-8') as fo:
        with codecs.open(file+'_marked_'+word_dic,'w','utf-8') as out:
            for idx,line in enumerate(fo):
                out.write('\t'.join(line.rstrip().split('\t')[:3])+'\t'+str(min(max(0,score[idx]),4))+'\n')
    os.system("sort -t $'\t' -k4 -g -r ../DATA_test/"+file+"_marked_"+word_dic+" > ../DATA_test/tmp")
    os.system("mv ../DATA_test/tmp ../DATA_test/"+file+"_marked_"+word_dic)


if __name__=='__main__':
    predict_on_file_demo('../MODEL/LR_1-attflag_True-cate_mlut-data_all-normal_True-prewordembedding_False-score_style_mine-seg_jieba-weight_True_ckpt',['../DATA_test/external_test_data'])