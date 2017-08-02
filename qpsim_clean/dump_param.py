#encoding:utf-8
from __future__ import print_function
import numpy as np

import sys
from six.moves import cPickle
import deepctxt_util
from deepctxt_util import DCTokenizer
import tensorflow as tf

from q2p_model import Config, Model

###############################################################################
# PARAMETERS 
###############################################################################
query_field = "query"
passage_field = "passage"
output_field = "similarity"
#model_name = "./model_5epoch_batch1k/q2p_tf.ckpt-3"
model_name = "/home/t-tiayi/test/za/model/baseline/q2p_tf.ckpt-1"

#input_filename = "/home/wendw/data/qgen_msra/eqna.qa.prod.test.QG.1best-output.forjudge.tsv"
#output_filename = "/home/wendw/data/qgen_msra/eqna.qa.prod.test.QG.1best-output.forjudge.tsv.withqpsimscore"

#input_filename = "/home/wendw/data/MALTA_L3Train/exp1/exp1_SemanticQueryBatch1.tsv"
#output_filename = "/home/wendw/data/MALTA_L3Train/exp1/exp1_SemanticQueryBatch1.tsv.withqpsimscore"

#input_filename = "/home/t-tiayi/qpsim_data/EQnA_QnARepro_L4_ranker_test1.tsv"
#output_filename = "EQnA_QnARepro_L4_ranker_test1.tsv"+".baseline"

#input_filename = sys.argv[1]
#output_filename = sys.argv[2]
params = {}

if len(sys.argv) > 3:
    params_str = sys.argv[3]
    for param in params_str.split(','):
        (key, value) = param.split('=')
        params[key] = value

if "query_field" in params:
    query_field = params["query_field"]

if "passage_field" in params:
    passage_field = params["passage_field"]

if "output_field" in params:
    output_field = params["output_field"]

if "model_name" in params:
    model_name = params["model_name"]

print("###############################################################")
#print("Input:" + input_filename)
#print("Output:" + output_filename)
print("Param: query_field="+ query_field)
print("Param: passage_field="+ passage_field)
print("Param: output_field="+ output_field)
print("Param: model_name=" + model_name)
print("###############################################################")

###############################################################################
# FUNCTIONS 
###############################################################################
def process_data(tokenizer, q_maxlen, p_maxlen, model, queries, passages):
    X1 = queries
    X2 = passages

    X1_test = tokenizer.texts_to_sequences(X1, q_maxlen)
    X1_test = deepctxt_util.pad_sequences(X1_test, maxlen=q_maxlen, padding='post', truncating='post')
    X2_test = tokenizer.texts_to_sequences(X2, p_maxlen)
    X2_test = deepctxt_util.pad_sequences(X2_test, maxlen=p_maxlen, padding='post', truncating='post')

    probs = sess.run([model.probs], feed_dict={model.input1:X1_test, model.input2:X2_test})
    return probs[0]

###############################################################################
# MAIN
###############################################################################

print('Loading tokenizer')
tokenizer = DCTokenizer()
#tokenizer.load("/home/wendw/data/glove/glove.6B.200d.txt", vocab_size=400000)
tokenizer.load("/home/t-tiayi/qpsim_data/glove.6B.300d.txt", vocab_size=400000)
#tokenizer.load("/home/wendw/data/glove/glove.840B.300d.txt", vocab_size=400000)
print('Done')

config = Config(tokenizer, use_cpu=True)

query_index = 0
passage_index = 1

lines = []
queries = []
passages = []

config_tf = tf.ConfigProto(allow_soft_placement = True, gpu_options = tf.GPUOptions(allow_growth = True))
with tf.Session(config=config_tf) as sess:
    with tf.variable_scope("model", reuse=False):
        model = Model(is_training=False, config=config, var_init=tf.random_uniform_initializer(-0.3, 0.3))


    #saver = tf.train.import_meta_graph(model_name)
    saver = tf.train.Saver()
    saver.restore(sess,model_name)
    graph = tf.get_default_graph()
    softmax_w = sess.run(graph.get_tensor_by_name("model/softmax_w:0"))
    softmax_b = sess.run(graph.get_tensor_by_name("model/softmax_b:0"))
    for t in tf.get_operations():
        print(t.name)
    with open("dense_w_b","w") as outfile:
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
    
