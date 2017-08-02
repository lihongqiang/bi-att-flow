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
        #tf.initialize_all_variables().run()
        sess.run(tf.global_variables_initializer())

        sess.run(m_train.embedding.assign(tokenizer.embedding_weights))
        saver = tf.train.Saver()
        saver.restore(sess,load_model_path)
        #for n in sess.graph.get_operations():
        #    if not n.name.startswith("model/"):continue
        #    print(n.name)
        graph=sess.graph
        softmax_w = sess.run(graph.get_tensor_by_name("model/softmax_w:0"))
        softmax_b = sess.run(graph.get_tensor_by_name("model/softmax_b:0"))
        var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for t in var:
            print(t.name)
        with open("dense_w_b_181903","w") as outfile:
            softmax_w_shape = softmax_w.shape
            outfile.write("{}\t{}\n".format(softmax_w_shape[0],softmax_w_shape[1]))
            for i in range(softmax_w_shape[0]):
                outfile.write("{}".format(softmax_w[i,0]))
                for j in range(1,softmax_w_shape[1]):
                    outfile.write("\t{}".format(softmax_w[i,j]))
                outfile.write("\n")
            softmax_b_shape = softmax_b.shape
            outfile.write("{}\n".format(softmax_b_shape[0]))
            outfile.write("{}".format(softmax_b[0]))
            for i in range(1,softmax_b_shape[0]):
                outfile.write("\t{}".format(softmax_b[i]))
            outfile.write("\n")

if __name__ == "__main__":
    tf.app.run()
