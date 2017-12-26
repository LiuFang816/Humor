# -*- coding: utf-8 -*-
import tensorflow as tf
class RNNConfig(object):
    embedding_size=64
    # seq_length=600
    num_classes=2
    vocab_size=5000
    num_layer=2
    hidden_size=128
    rnn='gru'
    dropout_keep_prob=0.8
    learning_rate=1e-3
    batch_size=8
    num_epochs=10
    print_per_batch=100
    save_per_batch=10

class TextRNN(object):
    def __init__(self,config):
        self.config=config
        self.input=tf.placeholder(tf.int32,[None,None],name='input')
        self.label=tf.placeholder(tf.int32,[None,self.config.num_classes])
        self.keep_prob=tf.placeholder(tf.float32,name='keep_prob')
        self.rnn()
    def rnn(self):
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size,state_is_tuple=True)
        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.config.hidden_size)
        def drop_out():
            if self.config.rnn=='lstm':
                cell=lstm_cell()
            else:
                cell=gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.config.dropout_keep_prob)
        with tf.device('/cpu:0'):
            embedding=tf.get_variable('embedding',[self.config.vocab_size,self.config.embedding_size])
            embedding_inputs=tf.nn.embedding_lookup(embedding,self.input)
        with tf.name_scope('rnn'):
            cells=[drop_out() for _ in range(self.config.num_layer)]
            rnn_cell=tf.contrib.rnn.MultiRNNCell(cells,state_is_tuple=True)
            outputs,_=tf.nn.dynamic_rnn(rnn_cell,embedding_inputs,dtype=tf.float32)
            last=outputs[:,-1,:]#取最后一个timestep的输出作为结果
        with tf.name_scope('score'):
            #全连接层
            fc=tf.layers.dense(last,self.config.hidden_size,name='fc1')
            fc=tf.contrib.layers.dropout(fc,self.keep_prob)
            fc=tf.nn.relu(fc)
            #输出层
            self.logits=tf.layers.dense(fc,self.config.num_classes,name='fc2')
            self.predict_class=tf.argmax(tf.nn.softmax(self.logits),1)
        with tf.name_scope('optimize'):
            cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.label)
            self.loss=tf.reduce_mean(cross_entropy)
            self.optimizer=tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        with tf.name_scope('accuracy'):
            correct_pre=tf.equal(tf.argmax(self.label,1),self.predict_class)
            self.acc=tf.reduce_mean(tf.cast(correct_pre,tf.float32))