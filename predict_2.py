#coding:UTF-8

from sta_helper import *
import matplotlib.pyplot as plt
from ADEM_light import *


def write_in_file(file,score):
    with open(file,'r',encoding='utf-8') as fo:
        with open(file+'_marked','w',encoding='utf-8') as out:
            for idx,line in enumerate(fo):
                out.write('\t'.join(line.rstrip().split('\t')[:3])+'\t'+str(score[idx])+'\n\n')


data_flag='8'
CHECKPOINT_PATH='./MODEL/cate_mlut-weight_True-normal_False-score_style_mine-LR_0.2-data_'+data_flag+'_ckpt'

# print(resolve_filename(CHECKPOINT_PATH))
config=resolve_filename(CHECKPOINT_PATH)
# config={'score_style':'mine',
#             'normal':False,
#         'LR':0.2,
#         'cate':'mlut',
#         'weight':True,
#         'data':data_flag
#         }
# print(config)

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

from ADEM_light_code.data_helper import *


dp=data_helper(data_flag=data_flag,test_file='./DATA_orgin/external_test_data_idx',cate=config['cate'],weight=config['weight'])

config_1={
   'HIDDEN_SIZE':128,
    'NUM_LAYERS':1,
    'SRC_VOCAB_SIZE':dp.vocab_size,
    'BARCH_SIZE':100,
    'NUM_EPOCH':5,
    'KEEP_PROB':1,
    'MAX_GRAD_NORM':5,
}


if __name__=='__main__':
    with tf.Session() as sess:
        train_model = ADEM_model(config, config_1)
        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        saver.restore(sess, CHECKPOINT_PATH)

        context_input, refrence_input, model_input, context_sequence_length, \
        refrence_sequence_length, model_sequence_length, human_score, grad_ys = dp.get_test_data()

        print(context_input.shape, refrence_input.shape, model_input.shape,
              context_sequence_length.shape, refrence_sequence_length.shape,
              model_sequence_length.shape, human_score.shape)

        predict_score = train_model.predict_on_batch(sess, feed_dict_={'context_input': context_input,
                                                                       'context_sequence_length': context_sequence_length,
                                                                       'model_input': model_input,
                                                                       'refrence_input': refrence_input,
                                                                       'model_sequence_length': model_sequence_length,
                                                                       'refrence_sequence_length': refrence_sequence_length,
                                                                       'human_score': human_score,
                                                                       })
        len_ = len(human_score)
        human_score = np.reshape(human_score, [len_])
        predict_score = np.reshape(predict_score, [len_])

        write_in_file('./DATA_orgin/external_test_data',predict_score)

        looup(human_score, predict_score)
        baseline_rd, baseline_mean = baseline(human_score)

        print('baseline_rd:', baseline_rd)
        print('baseline_mean:', baseline_mean)
        rmse=mean_square_error(predict_score, human_score)
        print('mean_square_error', rmse)
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
        plt.title('data:' + data_flag + ' | rmse:' + '%.4f' % (rmse))
        plt.show()

        plt.scatter(y2, y1, s=2)
        plt.xlabel('model_score')
        plt.ylabel('human_score')
        plt.title('data:' + data_flag + ' | rmse:' + '%.4f' % (rmse))
        plt.show()

        predict_score = normal(predict_score)
        x = np.array([i for i in range(len(human_score))])
        y1 = human_score
        y2 = predict_score
        plt.scatter(x, y1, s=2)
        plt.scatter(x, y2, s=2)
        plt.title('data:' + data_flag + ' | rmse:' + '%.4f' % (rmse))
        plt.show()
