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
#input_filename = "/home/t-tiayi/msproj/data/fasttest.tsv"
#input_filename = "/home/t-honli/data/s2vec/test.tsv"
#input_filename = "/home/t-honli/data/EQnA/train-S2V.tsv"
input_filename = "/home/t-honli/data/EQnA/dev-S2V.tsv"
#output_filename = "/home/t-honli/data/EQnA/train-vectors.tsv"
output_filename = "/home/t-honli/data/EQnA/dev-vectors.tsv"
query_field = "query"
passage_field = "passage"
output_field = "similarity"
model_name = "./model/q2p_tf_model_181903/q2p_tf.ckpt-1"

if len(sys.argv) == 3:
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
#input_filename = sys.argv[1]
#output_filename = sys.argv[2]
#model_name = sys.argv[3]

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
print("Input:" + input_filename)
print("Output:" + output_filename)
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

    q1_vect,q2_vect = sess.run([model.q1_vect,model.q2_vect], feed_dict={model.input1:X1_test, model.input2:X2_test})
    return q1_vect,q2_vect

def vec2str(vec):
    return " ".join([str(f) for f in vec])
###############################################################################
# MAIN
###############################################################################

print('Loading tokenizer')
tokenizer = DCTokenizer()
tokenizer.load("/home/t-tiayi/msproj/data/glove.6B.300d.txt", vocab_size=400000)
print('Done')

config = Config(tokenizer, use_cpu=True)

f_out = open(output_filename, "w", encoding='utf-8')
f_in = open(input_filename, "r", encoding='utf-8')

#header_line = f_in.readline().replace("\n", "")
#header = header_line.split('\t')

#f_out.write(header_line + "\t" + output_field + "\n")

#query_index = header.index(query_field)
#passage_index = header.index(passage_field)
query_index = 0
passage_index = 1

lines = []
queries = []
passages = []

config_tf = tf.ConfigProto(allow_soft_placement = True, gpu_options = tf.GPUOptions(allow_growth = True))
with tf.Session(config=config_tf) as sess:
    with tf.variable_scope("model", reuse=False):
        model = Model(is_training=False, config=config, var_init=tf.random_uniform_initializer(-0.3, 0.3))

    print("initialize variables")
    tf.initialize_all_variables().run()

    saver = tf.train.Saver(tf.all_variables())
    saver.restore(sess, model_name)

    line_count = 0
    for l in f_in:
        line = l.replace("\n", "")
        fields = line.split('\t')
        if len(fields) <= 1:
            continue

        lines.append(line)
        queries.append(fields[query_index])
        passages.append(fields[passage_index])

        line_count += 1
        if len(lines) >= 1000:
            print("processing " + str(line_count))
            q1_vec,q2_vec = process_data(tokenizer, config.q_maxlen, config.p_maxlen, model, queries, passages)
            for i in range(0, len(lines)):
                q1 = vec2str(q1_vec[i])
                q2 = vec2str(q2_vec[i])
                f_out.write(lines[i] + "\t" + "{}\t{}".format(q1,q2) + "\n")

            lines = []
            queries = []
            passages = []

    print("processing " + str(line_count))
    q1_vec,q2_vec = process_data(tokenizer, config.q_maxlen, config.p_maxlen, model, queries, passages)
    for i in range(0, len(lines)):
        q1 = vec2str(q1_vec[i])
        q2 = vec2str(q2_vec[i])
        f_out.write(lines[i] + "\t" + "{}\t{}".format(q1,q2) + "\n")
f_in.close()
f_out.close()


