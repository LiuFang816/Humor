# -*- coding: utf-8 -*-
import tensorflow as tf
class CNNConfig(object):
    filter_sizes=[3,4,5]
    num_filters=128
    embedding_size=128
    seq_length=600
    num_classes=2
    vocab_size=5000
    kernel_size=5
    dropout_keep_prob=0.5
    l2_reg_lambda=0.8
    learning_rate=1e-3
    batch_size=32
    num_epochs=50
    print_per_batch=100
    save_per_batch=10

class TextCNN(object):
    def __init__(self,config):
        self.config=config
        self.input=tf.placeholder(tf.int32,[None,None],name='input')
        self.label=tf.placeholder(tf.int32,[None,self.config.num_classes])
        self.keep_prob=tf.placeholder(tf.float32,name='keep_prob')
        self.l2_loss=tf.constant(0.0)
        self.cnn()
    def cnn(self):
        with tf.device('/cpu:0'):
            embedding=tf.get_variable('embedding',[self.config.vocab_size,self.config.embedding_size])
            embedding_inputs=tf.nn.embedding_lookup(embedding,self.input)
            embedding_inputs=tf.expand_dims(embedding_inputs,-1)
        with tf.name_scope('cnn'):
            pool_outputs=[]
            for i,filter_size in enumerate(self.config.filter_sizes):
                with tf.name_scope('conv-maxpool-%s'%filter_size):
                    filter_shape=[filter_size,self.config.embedding_size,1,self.config.num_filters]
                    W=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='W')
                    b=tf.Variable(tf.constant(0.1,shape=[self.config.num_filters]),name='b')
                    conv=tf.nn.conv2d(embedding_inputs,W,strides=[1,1,1,1],padding='VALID',name='conv')
                    h=tf.nn.relu(tf.nn.bias_add(conv,b),name='relu')
                    # seq_len=tf.constant(self.seq_len,tf.int32)
                    pooled=tf.nn.max_pool(h,[1,self.config.seq_length-filter_size+1,1,1],[1,1,1,1],padding='VALID',name='pool')# [batch_size,1,1,num_filters]
                    pool_outputs.append(pooled)

            self.h_pool=tf.concat(pool_outputs,3)# concat features from different filters
            num_filter_total=self.config.num_filters*len(self.config.filter_sizes)
            self.h_pool_flat=tf.reshape(self.h_pool,[-1,num_filter_total])

            with tf.name_scope('dropout'):
                self.h_drop=tf.nn.dropout(self.h_pool_flat,self.config.dropout_keep_prob)

            with tf.name_scope('output'):
                W=tf.get_variable('W',shape=[num_filter_total,self.config.num_classes],initializer=tf.contrib.layers.xavier_initializer())
                b=tf.Variable(tf.constant(0.1,shape=[self.config.num_classes]),name='b')
                self.l2_loss+=tf.nn.l2_loss(W)
                self.l2_loss+=tf.nn.l2_loss(b)
                self.scores=tf.nn.xw_plus_b(self.h_drop,W,b,name='scores')
                self.predictions=tf.argmax(self.scores,1,name='predictions')

            with tf.name_scope('loss'):
                losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.label)
                self.loss=tf.reduce_mean(losses)+self.config.l2_reg_lambda*self.l2_loss
                self.optimizer=tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

            with tf.name_scope('Accuracy'):
                correct_predictions=tf.equal(self.predictions,tf.argmax(self.label,1))
                self.accuracy=tf.reduce_mean(tf.cast(correct_predictions,'float'),name='accuracy')



