import argparse
import json
import math
import os
import shutil
from pprint import pprint

import tensorflow as tf
from tqdm import tqdm
import numpy as np

from basic.evaluator import ForwardEvaluator, MultiGPUF1Evaluator
from basic.graph_handler import GraphHandler
from basic.model import get_multi_gpu_models
from basic.trainer import MultiGPUTrainer
from basic.read_data import read_data, get_squad_data_filter, update_config
from my.tensorflow import get_num_params


def main(config):
    set_dirs(config)
    with tf.device(config.device):
        if config.mode == 'train':
            _train(config)
        elif config.mode == 'test':
            _test(config)
        elif config.mode == 'forward':
            _forward(config)
        else:
            raise ValueError("invalid value for 'mode': {}".format(config.mode))


def set_dirs(config):
    # create directories
    assert config.load or config.mode == 'train', "config.load must be True if not training"
    if not config.load and os.path.exists(config.out_dir):
        shutil.rmtree(config.out_dir)

    config.save_dir = os.path.join(config.out_dir, "save")
    print ('print save_dir', config.save_dir)
    
    config.log_dir = os.path.join(config.out_dir, "log")
    config.eval_dir = os.path.join(config.out_dir, "eval")
    config.answer_dir = os.path.join(config.out_dir, "answer")
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    if not os.path.exists(config.answer_dir):
        os.mkdir(config.answer_dir)
    if not os.path.exists(config.eval_dir):
        os.mkdir(config.eval_dir)


def _config_debug(config):
    if config.debug:
        config.num_steps = 2
        config.eval_period = 1
        config.log_period = 1
        config.save_period = 1
        config.val_num_batches = 2
        config.test_num_batches = 2


def _train(config):
    data_filter = get_squad_data_filter(config)
    
    # 训练的时候，构建词的编号，存储再shared.json里面，load时加载shared文件
    print ('train config.load ', config.load)
    train_data = read_data(config, 'train', config.load, data_filter=data_filter)
    
    # 对dev的数据，不构建词的编号
    print ('dev config.load True')
    dev_data = read_data(config, 'dev', True, data_filter=data_filter)
    update_config(config, [train_data])

    _config_debug(config)

    print ('config retrain:', config.retrain)
    # 如果不加载已有模型的
    if not config.retrain:
        # 训练数据中的word:vec dict, 通过查找golve获得
        word2vec_dict = train_data.shared['lower_word2vec'] if config.lower_word else train_data.shared['word2vec']

        # read data过程中对train data的单词编号
        word2idx_dict = train_data.shared['word2idx']

        # 训练数据中所有词的 id:vec dict
        idx2vec_dict = {word2idx_dict[word]: vec for word, vec in word2vec_dict.items() if word in word2idx_dict}

        # 构建word embedding矩阵，遍历train和dev在update_config中更新单词总个数,如果单词在训练数据集中的id2vec字典中，设置设个id的vec，否则随机给一个
        emb_mat = np.array([idx2vec_dict[idx] if idx in idx2vec_dict
                            else np.random.multivariate_normal(np.zeros(config.word_emb_size), np.eye(config.word_emb_size))
                            for idx in range(config.word_vocab_size)])
        config.emb_mat = emb_mat

        print ('emb_mat: ', len(emb_mat))
    
    if config.use_sentence_emb:
        # set model config
        config.use_char_emb = False
        config.use_word_emb = False
        config.highway = False
        
        # sentence embedding add
        train_s2v = json.load(open(os.path.join(config.data_dir, "{}_sent_emb.json".format('train'))))
        dev_s2v = json.load(open(os.path.join(config.data_dir, "{}_sent_emb.json".format('dev'))))
        config.qvec = train_s2v['qvec'] + dev_s2v['qvec']
        config.cvec = train_s2v['cvec'] + dev_s2v['cvec']
        
        train_data.shared['question2id'] = train_data.shared['q2id']
        train_data.shared['context2id'] = train_data.shared['c2id']
        dev_data.shared['question2id'] = dict()
        dev_data.shared['context2id'] = dict()
        
        # dev offset 只增加一次
        for key, val in dev_data.shared['q2id'].items():
            dev_data.shared['question2id'][key] = val + len(train_s2v['qvec'])
            
        for key, val in dev_data.shared['c2id'].items():
            dev_data.shared['context2id'][key] = val + len(train_s2v['cvec'])
    

    # construct model graph and variables (using default graph)
    # pprint(config.__flags, indent=2)
    models = get_multi_gpu_models(config)
    model = models[0]
    print("num params: {}".format(get_num_params()))
    trainer = MultiGPUTrainer(config, models)
    evaluator = MultiGPUF1Evaluator(config, models, tensor_dict=model.tensor_dict if config.vis else None)
    graph_handler = GraphHandler(config, model)  # controls all tensors and variables in the graph, including loading /saving

    # Variables
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options = tf.GPUOptions(allow_growth = True)))
    graph_handler.initialize(sess)

    # Begin training  20000   q_size/(60 * 1) * 12
    num_steps = config.num_steps or int(math.ceil(train_data.num_examples / (config.batch_size * config.num_gpus))) * config.num_epochs
    #num_steps = int(math.ceil(train_data.num_examples / (config.batch_size * config.num_gpus))) * config.num_epochs
    global_step = 0
    for batches in tqdm(train_data.get_multi_batches(config.batch_size, config.num_gpus,
                                                     num_steps=num_steps, shuffle=True, cluster=config.cluster), total=num_steps):
        global_step = sess.run(model.global_step) + 1  # +1 because all calculations are done after step
        get_summary = global_step % config.log_period == 0
        loss, summary, train_op = trainer.step(sess, batches, get_summary=get_summary)
        if get_summary:
            graph_handler.add_summary(summary, global_step)

        # occasional saving
        if global_step % config.save_period == 0:
            graph_handler.save(sess, global_step=global_step)

        if not config.eval:
            continue
        # Occasional evaluation
        if global_step % config.eval_period == 0:
            num_steps = math.ceil(dev_data.num_examples / (config.batch_size * config.num_gpus))
            if 0 < config.val_num_batches < num_steps:
                num_steps = config.val_num_batches
            e_train = evaluator.get_evaluation_from_batches(
                sess, tqdm(train_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps), total=num_steps)
            )

            graph_handler.add_summaries(e_train.summaries, global_step)
            e_dev = evaluator.get_evaluation_from_batches(
                sess, tqdm(dev_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps), total=num_steps))
            graph_handler.add_summaries(e_dev.summaries, global_step)

            if config.dump_eval:
                graph_handler.dump_eval(e_dev)
            if config.dump_answer:
                graph_handler.dump_answer(e_dev)
    if global_step % config.save_period != 0:
        graph_handler.save(sess, global_step=global_step)


def _test_emb(config):
    
    
    data_filter = get_squad_data_filter(config)
    print ('train config.load ', config.load)
    train_data = read_data(config, 'train', config.load, data_filter=data_filter)
    print ('dev config.load ', config.load)
    dev_data = read_data(config, 'dev', True, data_filter=data_filter)
    update_config(config, [train_data, dev_data])

    _config_debug(config)

    word2vec_dict = train_data.shared['lower_word2vec'] if config.lower_word else train_data.shared['word2vec']
    word2idx_dict = train_data.shared['word2idx']
    idx2vec_dict = {word2idx_dict[word]: vec for word, vec in word2vec_dict.items() if word in word2idx_dict}
    
    # word embedding
    emb_mat = np.array([idx2vec_dict[idx] if idx in idx2vec_dict
                        else np.random.multivariate_normal(np.zeros(config.word_emb_size), np.eye(config.word_emb_size))
                        for idx in range(config.word_vocab_size)])
    config.emb_mat = emb_mat
    
    if config.online:
        test_data = read_data(config, 'online', True)
    else:
        test_data = read_data(config, 'test', True)
    update_config(config, [test_data])

    _config_debug(config)
    
    if config.use_glove_for_unk:
        word2vec_dict = test_data.shared['lower_word2vec'] if config.lower_word else test_data.shared['word2vec']
        new_word2idx_dict = test_data.shared['new_word2idx']
        idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
        new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
        config.new_emb_mat = new_emb_mat

    # pprint(config.__flags, indent=2)
    models = get_multi_gpu_models(config)
    model = models[0]
    evaluator = MultiGPUF1Evaluator(config, models, tensor_dict=models[0].tensor_dict if config.vis else None)
    graph_handler = GraphHandler(config, model)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options = tf.GPUOptions(allow_growth = True)))
    graph_handler.initialize(sess)
    num_steps = math.ceil(1.0 * test_data.num_examples / (config.batch_size * config.num_gpus)) # 2021 / 10 = 203
    
    # 这个地方可以自己设置test的num batch，就是不测试所有的batch，一般小于总大小
    if 0 < config.test_num_batches < num_steps:
        num_steps = config.test_num_batches

    
    e = None
    for multi_batch in tqdm(test_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps, cluster=config.cluster), total=num_steps):
        ei = evaluator.get_evaluation(sess, multi_batch)
        e = ei if e is None else e + ei
        if config.vis:
            eval_subdir = os.path.join(config.eval_dir, "{}-{}".format(ei.data_type, str(ei.global_step).zfill(6)))
            if not os.path.exists(eval_subdir):
                os.mkdir(eval_subdir)
            path = os.path.join(eval_subdir, str(ei.idxs[0]).zfill(8))
            graph_handler.dump_eval(ei, path=path)

    print(e)
    if config.dump_answer:
        print("dumping answer ...")
        graph_handler.dump_answer(e)
    if config.dump_eval:
        print("dumping eval ...")
        graph_handler.dump_eval(e)

def _test(config):
    if config.online:
        test_data = read_data(config, 'online', True)
    else:
        test_data = read_data(config, 'test', True)
    update_config(config, [test_data])

    _config_debug(config)

    # 测试的是否，选择是否使用glove来更新不在训练的model里面的词的词向量
    if config.use_glove_for_unk:
        
        # 测试数据的词向量，通过glove获取
        word2vec_dict = test_data.shared['lower_word2vec'] if config.lower_word else test_data.shared['word2vec']
        
        # 加载的model中的word2id 字典，每一个id对应了model存储的embedding matrix矩阵中的一个词向量，id对应在新词的序号
        new_word2idx_dict = test_data.shared['new_word2idx']
        
        # id vec， 这里的id对应在新词的序号
        idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
        
        # 这里的idx是新数据中不在model的word表中的词的序号，应该不是id2vec的索引号
        new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
        config.new_emb_mat = new_emb_mat
        
        print ('the number of words not in model :', len(new_emb_mat))


    # pprint(config.__flags, indent=2)
    models = get_multi_gpu_models(config)
    model = models[0]
    evaluator = MultiGPUF1Evaluator(config, models, tensor_dict=models[0].tensor_dict if config.vis else None)
    graph_handler = GraphHandler(config, model)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options = tf.GPUOptions(allow_growth = True)))
    graph_handler.initialize(sess)
    num_steps = math.ceil(1.0 * test_data.num_examples / (config.batch_size * config.num_gpus)) # 2021 / 10 = 203
    
    # 这个地方可以自己设置test的num batch，就是不测试所有的batch，一般小于总大小
    if 0 < config.test_num_batches < num_steps:
        num_steps = config.test_num_batches
    
    print (num_steps, config.test_num_batches)
    
    e = None
    for multi_batch in tqdm(test_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps, cluster=config.cluster), total=num_steps):
        ei = evaluator.get_evaluation(sess, multi_batch)
        e = ei if e is None else e + ei
        if config.vis:
            eval_subdir = os.path.join(config.eval_dir, "{}-{}".format(ei.data_type, str(ei.global_step).zfill(6)))
            if not os.path.exists(eval_subdir):
                os.mkdir(eval_subdir)
            path = os.path.join(eval_subdir, str(ei.idxs[0]).zfill(8))
            graph_handler.dump_eval(ei, path=path)

    print(e)
    if config.dump_answer:
        print("dumping answer ...")
        graph_handler.dump_answer(e)
    if config.dump_eval:
        print("dumping eval ...")
        graph_handler.dump_eval(e)

def _forward(config):
    assert config.load
    test_data = read_data(config, config.forward_name, True)
    update_config(config, [test_data])

    _config_debug(config)

    if config.use_glove_for_unk:
        word2vec_dict = test_data.shared['lower_word2vec'] if config.lower_word else test_data.shared['word2vec']
        new_word2idx_dict = test_data.shared['new_word2idx']
        idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
        new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
        config.new_emb_mat = new_emb_mat

    pprint(config.__flags, indent=2)
    models = get_multi_gpu_models(config)
    model = models[0]
    print("num params: {}".format(get_num_params()))
    evaluator = ForwardEvaluator(config, model)
    graph_handler = GraphHandler(config, model)  # controls all tensors and variables in the graph, including loading /saving

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), gpu_options = tf.GPUOptions(allow_growth = True))
    graph_handler.initialize(sess)

    num_batches = math.ceil(test_data.num_examples / config.batch_size)
    if 0 < config.test_num_batches < num_batches:
        num_batches = config.test_num_batches
    e = evaluator.get_evaluation_from_batches(sess, tqdm(test_data.get_batches(config.batch_size, num_batches=num_batches), total=num_batches))
    print(e)
    if config.dump_answer:
        print("dumping answer ...")
        graph_handler.dump_answer(e, path=config.answer_path)
    if config.dump_eval:
        print("dumping eval ...")
        graph_handler.dump_eval(e, path=config.eval_path)


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    return parser.parse_args()


class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def _run():
    args = _get_args()
    with open(args.config_path, 'r') as fh:
        config = Config(**json.load(fh))
        main(config)


if __name__ == "__main__":
    _run()
