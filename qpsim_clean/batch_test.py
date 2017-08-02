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
input_filename = "/home/t-tiayi/msproj/data/Fake_fasttest.tsv"
#output_filename = "/home/wendw/data/EQnA_L4_train_test1_test2_20161223.tsv.output3"
output_filename = "Fake_fastest.result2"
query_field = "query"
passage_field = "passage"
output_field = "similarity"
#model_name = "./model_5epoch_batch1k/q2p_tf.ckpt-3"
model_name = "./model/q2p_tf_model_181903/q2p_tf.ckpt-1"

#input_filename = "/home/wendw/data/qgen_msra/eqna.qa.prod.test.QG.1best-output.forjudge.tsv"
#output_filename = "/home/wendw/data/qgen_msra/eqna.qa.prod.test.QG.1best-output.forjudge.tsv.withqpsimscore"

#input_filename = "/home/wendw/data/MALTA_L3Train/exp1/exp1_SemanticQueryBatch1.tsv"
#output_filename = "/home/wendw/data/MALTA_L3Train/exp1/exp1_SemanticQueryBatch1.tsv.withqpsimscore"


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

    probs = sess.run([model.probs], feed_dict={model.input1:X1_test, model.input2:X2_test})
    return probs[0]

###############################################################################
# MAIN
###############################################################################

print('Loading tokenizer')
tokenizer = DCTokenizer()
#tokenizer.load("/home/wendw/data/glove/glove.6B.200d.txt", vocab_size=400000)
tokenizer.load("/home/t-tiayi/msproj/data/glove.6B.300d.txt", vocab_size=400000)
#tokenizer.load("/home/wendw/data/glove/glove.840B.300d.txt", vocab_size=400000)
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
            probs = process_data(tokenizer, config.q_maxlen, config.p_maxlen, model, queries, passages)
            probs_f = process_data(tokenizer, config.q_maxlen, config.p_maxlen, model, queries, passages[::-1])
            new_passages = np.copy(passages)
            np.random.shuffle(new_passages)
            probs_np = process_data(tokenizer, config.q_maxlen, config.p_maxlen, model, queries, new_passages)
            for i in range(0, len(lines)):
                #f_out.write(lines[i] + "\t" + str(int(probs[i][1]*1000000)) + "\n")
                f_out.write(lines[i] + "\t" + str(probs[i][1]) + "\n")
                print(probs[i],probs_f[i],probs_np[i])

            lines = []
            queries = []
            passages = []

    print("processing " + str(line_count))
    probs = process_data(tokenizer, config.q_maxlen, config.p_maxlen, model, queries, passages)
    probs_f = process_data(tokenizer, config.q_maxlen, config.p_maxlen, model, queries, passages[::-1])
    new_passages = np.copy(passages)
    np.random.shuffle(new_passages)
    probs_np = process_data(tokenizer, config.q_maxlen, config.p_maxlen, model, queries, new_passages)
    for i in range(0, len(lines)):
        #f_out.write(lines[i] + "\t" + str(int(probs[i][1]*1000000)) + "\n")
        f_out.write(lines[i] + "\t" + str(probs[i][1]) + "\n")
        print(probs[i],probs_f[i],probs_np[i])

f_in.close()
f_out.close()


