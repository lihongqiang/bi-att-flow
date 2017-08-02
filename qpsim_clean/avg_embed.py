from __future__ import print_function
import numpy as np
seed = 1337
np.random.seed(seed)  # for reproducibility

from six.moves import cPickle

import tensorflow as tf
import os
import argparse
import random
import shutil

import deepctxt_util
from deepctxt_util import DCTokenizer

from q2p_model import Config, Model

def create_philly_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Philly arguments parser")
    parser.add_argument('-datadir', type=str, nargs='?',
                       help='path of the data directory')
    parser.add_argument('-logdir', type=str, nargs='?',
                       help='path of the log directory')
    parser.add_argument('-outputdir', type=str, 
                       help='path of the model directory')                      
    parser.add_argument('-modelname', type=str, 
                       help='model name')                      
    parser.add_argument('-batchsize', type=int, 
                       help='batch size')
    parser.add_argument('-maxepoch', type=int, 
                       help='max epoch')
    parser.add_argument('-keepprob', type=float, 
                       help='keep prob')    
    parser.add_argument('-l2reg', type=float, 
                       help='l2 reg lambda')                       
    parser.add_argument('-rinit', type=float, 
                       help='random init var')                       
    return parser

def main(unused_argv):
    model_path = "./model"
    data_path = "/home/t-tiayi/msproj/data"
    log_path = "./log"
    model_name = "q2p_tf"
    max_epoch = 2
    batch_size = 1000
    keep_prob = 1.0
    l2_reg_lambda = 0.01
    r_init = 0.001
    test_after_steps = 360
    load_model_path = "model/q2p_tf_model_181903/q2p_tf.ckpt-1"
    input_file_path = "/home/t-tiayi/msproj/data/EQnA_QnARepro_L4_ranker_test2.tsv"
    output_file_path = "EQnA_QnARepro_L4_ranker_test2.tsv.avg_vec"


    #Philly parameter override
    pparser = create_philly_parser()
    philly_args, _ = pparser.parse_known_args()
    print(philly_args)
    if philly_args.datadir:
        data_path = philly_args.datadir
    if philly_args.logdir:
        log_path = philly_args.logdir
    if philly_args.outputdir:
        model_path = philly_args.outputdir
    if philly_args.modelname:
        model_name = philly_args.modelname
    if philly_args.batchsize:
        batch_size = philly_args.batchsize    
    if philly_args.maxepoch:
        max_epoch = philly_args.maxepoch    
    if philly_args.keepprob:
        keep_prob = philly_args.keepprob            
    if philly_args.l2reg:
        l2_reg_lambda = philly_args.l2reg    
    if philly_args.rinit:
        r_init = philly_args.rinit   

    print("===================================================")
    print("max_epoch %d" % max_epoch)
    print("batch_size: %d" % batch_size)
    print("keep_prob: %f" % keep_prob)
    print("l2_reg_lambda: %f" % l2_reg_lambda)
    print("r_init: %f" % r_init)
    print("===================================================")

    print('Loading tokenizer...')
    tokenizer = DCTokenizer()
    tokenizer.load(os.path.join('/home/t-tiayi/msproj/data', 'glove.6B.300d.txt'), vocab_size=400000)
    #tokenizer.load(os.path.join('/home/wendw/data/conceptnet/', 'conceptnet.vec.txt'), vocab_size=300000)
    #tokenizer.load_bin(os.path.join('/home/wendw/data/w2v/', 'GoogleNews-vectors-negative300.bin'), vocab_size=100000)

    config = Config(tokenizer=tokenizer, keep_prob=keep_prob, l2_reg_lambda=l2_reg_lambda)
    q_maxlen = config.q_maxlen
    p_maxlen = config.p_maxlen

    print('Loading data... (Train)')
    (Q1, Q2, y_train) = deepctxt_util.load_raw_data_x1_x2_y(path=input_file_path)
    
    print('Done')
    print()

    print('Converting data...')
    Q1_train = tokenizer.texts_to_sequences(Q1, q_maxlen)
    Q2_train = tokenizer.texts_to_sequences(Q2, p_maxlen)
    print('Done')
    print()

    Y_train = deepctxt_util.to_categorical(y_train, 2).astype(np.float32)

    print("Pad sequences (samples x time)")
    Q1_train = deepctxt_util.pad_sequences(Q1_train, maxlen=q_maxlen, padding='post', truncating='post')
    Q2_train = deepctxt_util.pad_sequences(Q2_train, maxlen=p_maxlen, padding='post', truncating='post')
    print('Q1_train shape:', Q1_train.shape)
    print('Q2_train shape:', Q2_train.shape)
    print('y_train shape:', Y_train.shape)
    print()

    print('Build model...')
    # session
    config_tf = tf.ConfigProto(allow_soft_placement = True, gpu_options = tf.GPUOptions(allow_growth = True))
    with tf.Session(config=config_tf) as sess:
        initializer = tf.random_uniform_initializer(-r_init, r_init) 
        p_acc = tf.placeholder(dtype=tf.float32,shape=[])
        p_loss = tf.placeholder(dtype=tf.float32,shape=[])

        with tf.variable_scope("model", reuse=False):
            m_train = Model(is_training=True, config=config, var_init=initializer)
            tf.summary.scalar("Training loss", m_train.cost)
            tf.summary.scalar("Training accuracy", m_train.accuracy)

            tr_acc = tf.summary.scalar("train-acc",p_acc)
            tr_loss = tf.summary.scalar("train-loss",p_loss)
            tr_merge = tf.summary.merge(inputs=[tr_acc, tr_loss], collections=tf.get_collection(tf.GraphKeys.SUMMARIES))

        with tf.variable_scope("model", reuse=True):
            m_test = Model(is_training=False, config=config, var_init=initializer)
            tf.summary.scalar("Testing loss", m_test.cost)
            tf.summary.scalar("Testing accuracy", m_test.accuracy)

            tf.add_to_collection("_Q_", m_test.input1)
            tf.add_to_collection("_P_", m_test.input2)
            tf.add_to_collection("_score_", m_test.probs)

            
            te_acc = tf.summary.scalar("test-acc",p_acc)
            te_loss = tf.summary.scalar("test-loss",p_loss)
            te_merge = tf.summary.merge(inputs=[te_acc, te_loss], collections=tf.get_collection(tf.GraphKeys.SUMMARIES))

        print("initialize variables")
        embed_q = tf.nn.embedding_lookup(m_train.embedding,m_train.input1)
        embed_p = tf.nn.embedding_lookup(m_train.embedding,m_train.input2)
        embed_q = tf.reduce_mean(embed_q,axis=1)
        embed_p = tf.reduce_mean(embed_p,axis=1)

        sess.run(tf.global_variables_initializer())

        sess.run(m_train.embedding.assign(tokenizer.embedding_weights))
        saver = tf.train.Saver(tf.global_variables())
        #saver.restore(sess, load_model_path)
        with open(output_file_path,"w",encoding="utf-8") as output_file:
            for i in range(0,len(Y_train),batch_size):
                e_q,e_p = sess.run([embed_q,embed_p],feed_dict={m_train.input1:Q1_train[i:i+batch_size],m_train.input2:Q2_train[i:i+batch_size]})
                for j in range(len(e_q)):
                    eq = " ".join([str(f) for f in e_q[j]])
                    ep = " ".join([str(f) for f in e_p[j]])
                    output_file.write("{}\t{}\t{}\t{}\t{}\n".format(Q1[i+j],Q2[i+j],y_train[i+j],eq,ep))

if __name__ == "__main__":
    tf.app.run()
