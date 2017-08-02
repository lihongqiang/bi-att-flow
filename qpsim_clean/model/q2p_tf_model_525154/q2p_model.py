from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import LSTMCell

class Config:
    def __init__(self, tokenizer, use_cpu=False, keep_prob=1.0, l2_reg_lambda=0.01):
        self.keep_prob = keep_prob
        self.vocab_size = tokenizer.n_symbols
        self.vocab_dim = tokenizer.vocab_dim
        self.l2_reg_lambda = l2_reg_lambda
        self.use_cpu = use_cpu
        self.q_maxlen = 25
        self.p_maxlen = 100

class Model:
  def __init__(self, is_training, config, var_init):
    vocab_size = config.vocab_size
    vocab_dim = config.vocab_dim
    l2_loss = tf.constant(0.0)
    l2_reg_lambda = config.l2_reg_lambda
    lstm_dim = 200
    nb_classes = 2
    filter_sizes = [1,2,3,4]
    num_filters = 200
    num_filters_total = num_filters * len(filter_sizes)
    q_maxlen = config.q_maxlen
    p_maxlen = config.p_maxlen

    self.input1 = tf.placeholder(np.int32, [None, q_maxlen])
    self.input2 = tf.placeholder(np.int32, [None, p_maxlen])
    self.targets = tf.placeholder(np.float32, [None, nb_classes])
    self.length1  = tf.cast(tf.reduce_sum(tf.sign(self.input1), reduction_indices=1), tf.int32)
    self.length2  = tf.cast(tf.reduce_sum(tf.sign(self.input2), reduction_indices=1), tf.int32)

    # embedding
    gpu_str = "/gpu:1"
    #if config.use_cpu:
    #   gpu_str = "/cpu:0"

    with tf.device(gpu_str):
        self.embedding = tf.get_variable("embedding", [vocab_size, vocab_dim], tf.float32, trainable=True, initializer=var_init)

        with tf.variable_scope('query'):
            q1_fw_lstm_cell = LSTMCell(num_units=lstm_dim, state_is_tuple=True,reuse = tf.get_variable_scope().reuse,activation=tf.tanh)
            q1_bw_lstm_cell = LSTMCell(num_units=lstm_dim, state_is_tuple=True,reuse = tf.get_variable_scope().reuse,activation=tf.tanh)

            q1_emb = tf.nn.embedding_lookup(self.embedding, self.input1)
            #q1_emb = [tf.squeeze(input_, [1]) for input_ in tf.split(axis=1, num_or_size_splits=q_maxlen, value=q1_emb)]

            #q1_fb_outputs, q1_fw_state, q1_bw_state = tf.contrib.rnn.stack_bidirectional_rnn([q1_fw_lstm_cell], [q1_bw_lstm_cell], q1_emb, sequence_length=self.length1, dtype=tf.float32)
            #q1_outputs = tf.transpose(tf.stack(q1_fb_outputs), perm=[1,0,2])
            q1_outputs,q1_state = tf.nn.bidirectional_dynamic_rnn(q1_fw_lstm_cell,q1_bw_lstm_cell,q1_emb,sequence_length=self.length1,dtype=tf.float32)
            q1_outputs = tf.concat(axis=-1,values=q1_outputs)

            q1_outputs_expanded = tf.expand_dims(q1_outputs, -1)
            # Create a convolution + maxpool layer for each filter size
            q1_pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("q1-conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, lstm_dim*2, 1, num_filters]
                    q1_W = tf.get_variable("W-%s" % i, filter_shape, tf.float32, initializer=var_init)
                    q1_b = tf.get_variable("b-%s" % i, [num_filters], tf.float32, initializer=var_init)
                    q1_conv = tf.nn.conv2d(
                        q1_outputs_expanded,
                        q1_W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv-%s" % i)
                    # Apply nonlinearity
                    q1_h = tf.nn.relu(tf.nn.bias_add(q1_conv, q1_b), name="relu-%s" % i)
                    # Maxpooling over the outputs
                    q1_pooled = tf.nn.max_pool(
                        q1_h,
                        ksize=[1, q_maxlen - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool-%s" % i)
                    q1_pooled_outputs.append(q1_pooled)

            # Combine all the pooled features
            q1_h_pool = tf.concat(axis=3, values=q1_pooled_outputs)
            q1_h_pool_flat = tf.reshape(q1_h_pool, [-1, num_filters_total])
            self.q1_vect = q1_h_pool_flat

        with tf.variable_scope('passage'):
            q2_fw_lstm_cell = LSTMCell(num_units=lstm_dim, state_is_tuple=True,reuse = tf.get_variable_scope().reuse,activation=tf.tanh)
            q2_bw_lstm_cell = LSTMCell(num_units=lstm_dim, state_is_tuple=True,reuse = tf.get_variable_scope().reuse,activation=tf.tanh)

            q2_emb = tf.nn.embedding_lookup(self.embedding, self.input2)
            #q2_emb = [tf.squeeze(input_, [1]) for input_ in tf.split(axis=1, num_or_size_splits=p_maxlen, value=q2_emb)]

            #q2_fb_outputs, q2_fw_state, q2_bw_state = tf.contrib.rnn.stack_bidirectional_rnn([q2_fw_lstm_cell], [q2_bw_lstm_cell], q2_emb, sequence_length=self.length2, dtype=tf.float32)
            #q2_outputs = tf.transpose(tf.stack(q2_fb_outputs), perm=[1,0,2])
            q2_outputs,q2_state = tf.nn.bidirectional_dynamic_rnn(q2_fw_lstm_cell,q2_bw_lstm_cell,q2_emb,sequence_length=self.length2,dtype=tf.float32)
            q2_outputs = tf.concat(axis=-1,values=q2_outputs)

            q2_outputs_expanded = tf.expand_dims(q2_outputs, -1)

            # Create a convolution + maxpool layer for each filter size
            q2_pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("q2-conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, lstm_dim*2, 1, num_filters]
                    q2_W = tf.get_variable("W-%s" % i, filter_shape, tf.float32, initializer=var_init)
                    q2_b = tf.get_variable("b-%s" % i, [num_filters], tf.float32, initializer=var_init)
                    q2_conv = tf.nn.conv2d(
                        q2_outputs_expanded,
                        q2_W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv-%s" % i)
                    # Apply nonlinearity
                    q2_h = tf.nn.relu(tf.nn.bias_add(q2_conv, q2_b), name="relu-%s" % i)
                    # Maxpooling over the outputs
                    q2_pooled = tf.nn.max_pool(
                        q2_h,
                        ksize=[1, p_maxlen - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool-%s" % i)
                    q2_pooled_outputs.append(q2_pooled)

            # Combine all the pooled features
            q2_h_pool = tf.concat(axis=3, values=q2_pooled_outputs)
            q2_h_pool_flat = tf.reshape(q2_h_pool, [-1, num_filters_total])
            self.q2_vect = q2_h_pool_flat

        # merge
        merged = tf.multiply(self.q1_vect, self.q2_vect)

        if is_training and config.keep_prob < 1:
            merged = tf.nn.dropout(merged, config.keep_prob)

        # dense
        softmax_w = tf.get_variable("softmax_w", [num_filters_total, nb_classes], tf.float32, initializer=var_init)
        softmax_b = tf.get_variable("softmax_b", [nb_classes], tf.float32, initializer=var_init)
        logits = tf.matmul(merged, softmax_w) + softmax_b

        self.probs = tf.nn.softmax(logits) 

        l2_loss += tf.nn.l2_loss(softmax_w)
        l2_loss += tf.nn.l2_loss(softmax_b)

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.targets)
        loss = tf.reduce_mean(loss) + l2_reg_lambda * l2_loss

        self.train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(), optimizer='Adam', learning_rate=0.001)
        self.cost = loss

        self.predictions = tf.argmax(logits, 1)
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.targets, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
