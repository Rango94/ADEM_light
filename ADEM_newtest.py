import tensorflow as tf
import sys
sys.path.append('./DATA_orgin')
from ADEM_light_code.data_helper import *

class ADEM:

    def __init__(self):
        self.NUM_CHANNELS=1
        self.WORD_LEN=200
        self.FEATURE_SHAPE=[12,3665]


        self.CONV1_DEEP=4
        self.CONV1_WIDE=self.WORD_LEN
        self.CONV1_HIGH=1

        self.CONV2_DEEP=4
        self.CONV2_WIDE=self.WORD_LEN
        self.CONV2_HIGH=2


        self.CONV3_DEEP=4
        self.CONV3_WIDE=self.WORD_LEN
        self.CONV3_HIGH=3


        self.CONV4_DEEP=4
        self.CONV4_WIDE=self.WORD_LEN
        self.CONV4_HIGH=4

        self.FC_SIZ=24
        self.NUM_LABELS=21


    def inference_single(self,input_tensor):
        #embedding层，目前定义的输入数据shape是[batch_size,12,3665]
        #需要将其映射为[batch_size,12,200]
        #由于cnn的特性，我们需要将它reshape成[batch_size,12,200,1]



        with tf.variable_scope('layer0-embedding'):
            input_tensor = tf.reshape(input_tensor, [-1, self.FEATURE_SHAPE[1]])
            embedding_weights = tf.get_variable('weight', [self.FEATURE_SHAPE[1], self.WORD_LEN],
                                                initializer=tf.random_uniform_initializer(maxval=0.5 / self.WORD_LEN,
                                                                                          minval=-0.5 / self.WORD_LEN))
            embedding_baises = tf.get_variable('bais', [self.WORD_LEN], initializer=tf.constant_initializer(0.0))
            embedding = tf.matmul(input_tensor, embedding_weights) + embedding_baises
            embedding = tf.reshape(embedding, [-1, 12, self.WORD_LEN, self.NUM_CHANNELS])

        with tf.variable_scope('layer1-conv1'):
            conv1_weights=tf.get_variable('wegiht',[self.CONV1_HIGH,self.CONV1_WIDE,self.NUM_CHANNELS,self.CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1_biases=tf.get_variable('bais',[self.CONV1_DEEP],initializer=tf.constant_initializer(0.0))
            conv1=tf.nn.conv2d(embedding,conv1_weights,strides=[1,1,self.WORD_LEN,1],padding='SAME')
            relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

        with tf.variable_scope('layer1-conv2'):
            conv2_weights=tf.get_variable('wegiht',[self.CONV2_HIGH,self.CONV2_WIDE,self.NUM_CHANNELS,self.CONV2_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases=tf.get_variable('bais',[self.CONV2_DEEP],initializer=tf.constant_initializer(0.0))
            conv2=tf.nn.conv2d(embedding,conv2_weights,strides=[1,1,self.WORD_LEN,1],padding='SAME')
            relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

        with tf.variable_scope('layer1-conv3'):
            conv3_weights=tf.get_variable('wegiht',[self.CONV3_HIGH,self.CONV3_WIDE,self.NUM_CHANNELS,self.CONV3_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv3_biases=tf.get_variable('bais',[self.CONV3_DEEP],initializer=tf.constant_initializer(0.0))
            conv3=tf.nn.conv2d(embedding,conv3_weights,strides=[1,1,self.WORD_LEN,1],padding='SAME')
            relu3=tf.nn.relu(tf.nn.bias_add(conv3,conv3_biases))

        with tf.variable_scope('layer1-conv4'):
            conv4_weights=tf.get_variable('wegiht',[self.CONV4_HIGH,self.CONV4_WIDE,self.NUM_CHANNELS,self.CONV4_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv4_biases=tf.get_variable('bais',[self.CONV4_DEEP],initializer=tf.constant_initializer(0.0))
            conv4=tf.nn.conv2d(embedding,conv4_weights,strides=[1,1,self.WORD_LEN,1],padding='SAME')
            relu4=tf.nn.relu(tf.nn.bias_add(conv4,conv4_biases))

        with tf.name_scope('layer2-pool1'):
            conv1_shape=relu1.get_shape().as_list()
            pool1=tf.nn.max_pool(relu1,ksize=[1,conv1_shape[1],conv1_shape[2],1],strides=[1,conv1_shape[1],conv1_shape[2],1],padding='SAME')

        with tf.name_scope('layer2-pool2'):
            conv2_shape = relu1.get_shape().as_list()
            pool2=tf.nn.max_pool(relu2,ksize=[1,conv2_shape[1],conv2_shape[2],1],strides=[1,conv1_shape[1],conv1_shape[2],1],padding='SAME')

        with tf.name_scope('layer2-pool3'):
            conv3_shape = relu1.get_shape().as_list()
            pool3=tf.nn.max_pool(relu3,ksize=[1,conv3_shape[1],conv3_shape[2],1],strides=[1,conv1_shape[1],conv1_shape[2],1],padding='SAME')

        with tf.name_scope('layer2-pool4'):
            conv4_shape = relu1.get_shape().as_list()
            pool4=tf.nn.max_pool(relu4,ksize=[1,conv4_shape[1],conv4_shape[2],1],strides=[1,conv1_shape[1],conv1_shape[2],1],padding='SAME')

        pool_shape=pool1.get_shape().as_list()
        nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
        reshaped1=tf.reshape(pool1,[-1,nodes])

        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped2 = tf.reshape(pool2, [-1, nodes])

        pool_shape = pool3.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped3 = tf.reshape(pool3, [-1, nodes])

        pool_shape = pool4.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped4 = tf.reshape(pool4, [-1, nodes])

        reshaped=tf.concat([reshaped1,reshaped2],1)
        reshaped=tf.concat([reshaped,reshaped3],1)
        reshaped=tf.concat([reshaped,reshaped4],1)
        return reshaped


    def inference(self,context_src,true_response_src,model_response_src,train,regularizer):

        with tf.variable_scope('context'):
            context=self.inference_single(context_src)
        with tf.variable_scope('true_response_src'):
            true_response=self.inference_single(true_response_src)
        with tf.variable_scope('model_response_src'):
            model_response=self.inference_single(model_response_src)

        input_tensor=tf.concat([context,true_response],1)
        input_tensor=tf.concat([input_tensor,model_response],1)

        shape=input_tensor.get_shape().aslist()
        print(input_tensor.get_shape)
        with tf.variable_scope('layer3-fc1'):
            fc1_weights=tf.get_variable('weight',[shape[0],self.FC_SIZ],initializer=tf.truncated_normal_initializer(stddev=0.1))
            if regularizer!=None:
                tf.add_to_collection('losses',regularizer(fc1_weights))
            fc1_biases=tf.get_variable('bias',[self.FC_SIZ],initializer=tf.constant_initializer(0.1))
            fc1=tf.nn.tanh(tf.matmul(input_tensor,fc1_weights)+fc1_biases)
            if train:fc1=tf.nn.dropout(fc1,0.5)

        with tf.variable_scope('layer4-fc2'):
            fc2_weights=tf.get_variable('weight',[self.FC_SIZ,self.NUM_LABELS],initializer=tf.truncated_normal_initializer(stddev=0.1))
            if regularizer!=None:
                tf.add_to_collection('losses',regularizer(fc2_weights))
            fc2_biases=tf.get_variable('bias',[self.NUM_LABELS],initializer=tf.constant_initializer(0.1))
            logit=tf.matmul(fc1,fc2_weights)+fc2_biases
        return logit


MOVING_AVERAGE_DECAY=0.99

BATCH_SIZE=100

LEARNING_RATE_BASE=1.0
LEARNING_RATE_DECAY=0.99

MODEL_SAVE_PATH='/model/'
MODEL_NAME='model'
REGULARIZER_RATE=0.001

adem=ADEM()

def train(dh):
    x=tf.placeholder(tf.float32, [None, adem.FEATURE_SHAPE[0], adem.FEATURE_SHAPE[1]], name='x-input')
    y_=tf.placeholder(tf.float32, [None,adem.NUM_LABELS], name='y-input')
    Regularzer=tf.contrib.layers.l2_regularizer(REGULARIZER_RATE)
    y=adem.inference(x, True, Regularzer)
    global_step=tf.Variable(0,trainable=False)

    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op=variable_averages.apply(tf.trainable_variables())

    cross_entropy=tf.nn.softmax_cross_entropy_with_logits_v2(logits=y,labels=y_)
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,dh.train_file_num*5000/BATCH_SIZE,LEARNING_RATE_DECAY)
    train_step=tf.train.AdadeltaOptimizer(learning_rate).minimize(loss,global_step=global_step)

    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op=tf.no_op(name='train')

    # saver=tf.train.Saver()

    TRAINING_STEPS = 30000

    dh.set_batch_size(BATCH_SIZE)

    X_test,Y_test=dh.get_test_data()

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(TRAINING_STEPS):

            xs,ys=dh.next_batch()

            # print(sess.run(y,feed_dict={x:xs,y_:ys}))
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i%10==0:
                print('After %d training steps, loss on training batch is %g.' %(step,loss_value))
                # saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
                y_out=sess.run(y,feed_dict={x:X_test,y_:Y_test}).tolist()
                # print(y_out[0],y_out[10],y_out[50])
                y_out_r=Y_test.tolist()
                r=0
                tt=0
                for idx,i in enumerate(y_out):
                    if i.index(max(i))==y_out_r[idx].index(max(y_out_r[idx])):
                        r+=1
                    tt+=1
                print(r,tt,r/tt)
def main(argv=None):
    dh=data_helper()
    train(dh)
    #1

tf.app.run()













