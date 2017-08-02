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
input_filename = "/home/t-tiayi/msproj/data/fasttest.tsv"
output_filename = "res"
text_field = "query"
output_field = "query_vector"
query_or_passage = "query"
model_name = "./model/q2p_tf_model_181903/q2p_tf.ckpt-1"

#input_filename = sys.argv[1]
#output_filename = sys.argv[2]
params = {}

if len(sys.argv) > 3:
    params_str = sys.argv[3]
    for param in params_str.split(','):
        (key, value) = param.split('=')
        params[key] = value

if "text_field" in params:
    text_field = params["text_field"]

if "output_field" in params:
    output_field = params["output_field"]

if "model_name" in params:
    model_name = params["model_name"]

if "query_or_passage" in params:
    query_or_passage = params["query_or_passage"]


print("###############################################################")
print("Input:" + input_filename)
print("Output:" + output_filename)
print("Param: text_field="+ text_field)
print("Param: output_field="+ output_field)
print("Param: model_name=" + model_name)
print("Param: query_or_passage=" + query_or_passage)
print("###############################################################")

###############################################################################
# FUNCTIONS 
###############################################################################
def process_data(tokenizer, texts, model, config, query_or_passage):
    if query_or_passage == "query":
        X1_test = tokenizer.texts_to_sequences(texts, config.q_maxlen)
        X1_test = deepctxt_util.pad_sequences(X1_test, maxlen=config.q_maxlen, padding='post', truncating='post')
        result = sess.run([model.q1_vect], feed_dict={model.input1:X1_test})
    else:
        X1_test = tokenizer.texts_to_sequences(texts, config.p_maxlen)
        X1_test = deepctxt_util.pad_sequences(X1_test, maxlen=config.p_maxlen, padding='post', truncating='post')
        result = sess.run([model.q2_vect], feed_dict={model.input2:X1_test})

    vec_strs = []
    for vec in result[0]:
        vec_str = ','.join([str(n) for n in vec])
        vec_strs.append(vec_str)
    return vec_strs

###############################################################################
# MAIN
###############################################################################

print('Loading tokenizer')
tokenizer = DCTokenizer()
tokenizer.load("/home/t-tiayi/msproj/data/glove.6B.300d.txt")
print('Done')

config = Config(tokenizer, use_cpu=True)

f_out = open(output_filename, "w", encoding='utf-8')
f_in = open(input_filename, "r", encoding='utf-8')

header_line = f_in.readline().replace("\n", "")
header = header_line.split('\t')

f_out.write(header_line + "\t" + output_field + "\n")

header = ["query","passage","label"]
text_index = header.index(text_field)

lines = []
texts = []

with tf.Session() as sess:

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
        texts.append(fields[text_index])

        line_count += 1
        if len(lines) >= 1000:
            print("processing " + str(line_count))
            results = process_data(tokenizer, texts, model, config, query_or_passage)
            for i in range(0, len(lines)):
                f_out.write(lines[i] + "\t" + results[i] + "\n")

            lines = []
            texts = []

    print("processing " + str(line_count))
    if len(lines) > 0:
        results = process_data(tokenizer, texts, model, config, query_or_passage)
        for i in range(0, len(lines)):
            f_out.write(lines[i] + "\t" + results[i] + "\n")

f_in.close()
f_out.close()
