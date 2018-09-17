#coding:UTF-8
import sys

from ADEM_light import *
import os

config={'score_style':'mine',
            'normal':False,
        'LR':0.2,
        'cate':'mlut',
        'weight':True,
        'data':'all'
        }

if len(sys.argv)==2:
    config['data']=sys.argv[1]

model_name='-'.join([key+'_'+str(config[key]) for key in config])

CHECKPOINT_PATH = '../MODEL/' + model_name + '_ckpt'

if os.path.exists(CHECKPOINT_PATH+'.index'):
    exists_flag=True
else:
    exists_flag=False


# if config['data']=='orgin':
#     sys.path.append('./DATA_orgin')
#     from data_helper import *
# elif config['data']=='8':
#     sys.path.append('./DATA_8')
#     from data_helper import *
# elif config['data']=='9':
#     sys.path.append('./DATA_9')
#     from data_helper import *
# elif config['data']=='total':
#     sys.path.append('./DATA_total')
#     from data_helper import *

from data_helper import *

dp = data_helper(data_flag=config['data'],dic_file='word_dic',cate=config['cate'], weight=config['weight'])

config_1={
   'HIDDEN_SIZE':128,
    'NUM_LAYERS':1,
    'SRC_VOCAB_SIZE':dp.vocab_size,
    'BARCH_SIZE':100,
    'NUM_EPOCH':5,
    'KEEP_PROB':0.8,
    'MAX_GRAD_NORM':5,
}

def main():

    train_model=ADEM_model(config, config_1)

    saver=tf.train.Saver()
    step=0
    max_loop=9999999
    min_=9999
    marks=[]

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        if exists_flag:
            print('loading trained model')
            saver = tf.train.Saver()
            saver.restore(sess, CHECKPOINT_PATH)

        for i in range(max_loop):
            context_input_, refrence_input_, model_input_, context_sequence_length_, \
            refrence_sequence_length_,model_sequence_length_,human_score_,grad_ys_=dp.next_batch(300)

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
                refrence_sequence_length_, model_sequence_length_, human_score_,grad_ys_= dp.get_val_data()
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
                print('DATA:',config['data'],dp.train_file_list[(dp.train_cont-1)%3] if config['data']=='all' else '')

if __name__=='__main__':
    main()
