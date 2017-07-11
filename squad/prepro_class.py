import json
import os
# data: q, cq, (dq), (pq), y, *x, *cx
# shared: x, cx, (dx), (px), word_counter, char_counter, word2vec
# no metadata
from collections import Counter

from tqdm import tqdm
import nltk.tokenize as nltk
from squad.utils import get_word_span, get_word_idx, process_tokens

import nltk.tokenize as nltk
import os

class PreproClass():
    
    def __init__(self):
        
        self.args = self.get_args()
        self.glove_path = os.path.join(self.args['glove_dir'], "glove.{}.{}d.txt".format(self.args['glove_corpus'], self.args['glove_vec_size']))
        self.glove =  self.getGloveDict()
    
    def getGloveDict(self):
        glove = {}
        # load glove
        sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
        total = sizes[self.args['glove_corpus']]
        with open(self.glove_path, 'r', encoding='utf-8') as fh:
            for line in tqdm(fh, total=total):
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                glove[word] = vector
        return glove

    def get_args(self):
        args = {}
        args['source_dir'] = '/home/t-honli/data/online'
        args['target_dir'] = '/home/t-honli/bi-att-flow/data/online'
        args['glove_dir'] = '/home/t-honli/data/glove'
        args['file_name'] = 'online'
        args['glove_corpus'] = '6B'
        args['glove_vec_size'] = 100
        args['tokenizer'] = "PTB"
        return args

    def save_online(self, data, shared, file_path):
        file_name = file_path.split('/')[-1]
        file_dir = os.path.join(self.args['target_dir'], file_name.split('.')[0])
        os.mkdir(file_dir)
        data_path = os.path.join(file_dir, "data_{}.json".format(self.args['file_name']))
        shared_path = os.path.join(file_dir, "shared_{}.json".format(self.args['file_name']))
        json.dump(data, open(data_path, 'w'))
        json.dump(shared, open(shared_path, 'w'))

    def get_word2vec(self, word_counter):
        word2vec_dict = {}
        for word in list(word_counter.keys()):
            if word in self.glove:
                word2vec_dict[word] = self.glove[word]
            elif word.capitalize() in self.glove:
                word2vec_dict[word.capitalize()] = self.glove[word.capitalize()]
            elif word.lower() in self.glove:
                word2vec_dict[word.lower()] = self.glove[word.lower()]
            elif word.upper() in self.glove:
                word2vec_dict[word.upper()] = self.glove[word.upper()]

        print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), self.glove_path))
        return word2vec_dict

    # data_file online.json
    def prepro_online(self, file_path, start_ratio=0.0, stop_ratio=1.0, in_path=None):
            
        #sent_tokenize = nltk.sent_tokenize
        def word_tokenize(tokens):
            return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


        #source_path = os.path.join(self.args['source_dir'], self.args['data_file'])
        source_path = file_path
        source_data = json.load(open(source_path, 'r'))

        q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
        na = []
        cy = []
        x, cx = [], []
        answerss = []
        p = []
        word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
        start_ai = int(round(len(source_data['data']) * start_ratio))
        stop_ai = int(round(len(source_data['data']) * stop_ratio))
        for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
            xp, cxp = [], []
            pp = []
            x.append(xp)    # [[[xi], []], []]所有article   [[[xi], [xi]]]
            cx.append(cxp)
            p.append(pp)
            for pi, para in enumerate(article['paragraphs']):   # 每个段落
                # wordss
                context = para['context']
                context = context.replace("''", '" ')
                context = context.replace("``", '" ')
                xi = list(map(word_tokenize, nltk.sent_tokenize(context)))       # paragraph context
                xi = [process_tokens(tokens) for tokens in xi]  # process tokens

                if pi == 0:
                    print (xi)

                # given xi, add chars
                cxi = [[list(xijk) for xijk in xij] for xij in xi]
                xp.append(xi)           # paragraph list (a article)
                cxp.append(cxi)
                pp.append(context)

                for xij in xi:
                    for xijk in xij:
                        word_counter[xijk] += len(para['qas'])
                        lower_word_counter[xijk.lower()] += len(para['qas'])
                        for xijkl in xijk:
                            char_counter[xijkl] += len(para['qas']) # context 中的每个token，权重为question的个数

                rxi = [ai, pi]
                assert len(x) - 1 == ai
                assert len(x[ai]) - 1 == pi
                for qa in para['qas']:
                    # get words
                    qi = word_tokenize(qa['question'])
                    qi = process_tokens(qi)
                    cqi = [list(qij) for qij in qi]
                    yi = []
                    cyi = []
                    answers = []
                    for answer in qa['answers']:
                        answer_text = answer['text']
                        answers.append(answer_text)
                        answer_start = answer['answer_start']
                        answer_stop = answer_start + len(answer_text)
                        # TODO : put some function that gives word_start, word_stop here
                        yi0, yi1 = get_word_span(context, xi, answer_start, answer_stop)    #第一个词和最后一个词的句子和词索引
                        # yi0 = answer['answer_word_start'] or [0, 0]
                        # yi1 = answer['answer_word_stop'] or [0, 1]
                        assert len(xi[yi0[0]]) > yi0[1]
                        assert len(xi[yi1[0]]) >= yi1[1]
                        # print ('yi, yi1:')
                        # print (yi0[0], yi0[1])
                        # print (yi1[0], yi1[1]-1)
                        w0 = xi[yi0[0]][yi0[1]]    # 获取答案的第一个词
                        w1 = xi[yi1[0]][yi1[1]-1]  # 获取答案的最后一个词
                        i0 = get_word_idx(context, xi, yi0)  # 获取第一个词在context中的start位置
                        i1 = get_word_idx(context, xi, (yi1[0], yi1[1]-1))  # 获取最后一个词在context中的start位置
                        cyi0 = answer_start - i0  # 减去偏移，从0开始， 获取第一个词的第一个字母的索引
                        cyi1 = answer_stop - i1 - 1  # 获取最后一个词的最后一个字母的索引

                        # print(answer_text, w0[cyi0:], w1[:cyi1+1])
                        # print ('answer_text:', answer_text)
                        # print ('w0:', w0)
                        # print ('w1:', w1)
                        assert answer_text[0] == w0[cyi0], (answer_text, w0, cyi0)  # 答案的第一个字母和第一个词的第一个字幕是否相同
                        assert answer_text[-1] == w1[cyi1]  # 答案的最后一个字母和最后一个词的最后一个字母是否相同
                        assert cyi0 < 32, (answer_text, w0)
                        assert cyi1 < 32, (answer_text, w1)

                        yi.append([yi0, yi1])
                        cyi.append([cyi0, cyi1])

                    if len(qa['answers']) == 0:
                        yi.append([(0, 0), (0, 1)])
                        cyi.append([0, 1])
                        na.append(True)
                    else:
                        na.append(False)

                    for qij in qi:
                        word_counter[qij] += 1
                        lower_word_counter[qij.lower()] += 1
                        for qijk in qij:
                            char_counter[qijk] += 1

                    q.append(qi)
                    cq.append(cqi)
                    y.append(yi)
                    cy.append(cyi)
                    rx.append(rxi)
                    rcx.append(rxi)
                    ids.append(qa['id'])
                    idxs.append(len(idxs))
                    answerss.append(answers)

        word2vec_dict = self.get_word2vec(word_counter)
        lower_word2vec_dict = self.get_word2vec(lower_word_counter)

        # add context here
        data = {'q': q, 'cq': cq, 'y': y, '*x': rx, '*cx': rcx, 'cy': cy,
                'idxs': idxs, 'ids': ids, 'answerss': answerss, '*p': rx, 'na': na}

        # q:    question token list
        # cq:   question token charchater list
        # y:    answer_start(sent_id, word_id), answer_stop+1(sent_id, word_id)
        # cy:   answer_start在token中的id， answer_stop在token中的id
        # rx:   [article_id, paragraph_id]
        # rcx:  [article_id, paragraph_id]
        # ids:  question id list
        # idxs: question id list(start from 0)

        shared = {'x': x, 'cx': cx, 'p': p,
                  'word_counter': word_counter, 'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
                  'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}

        # x:            context tokens list  [  art[  cont[   seq[] ]            ]]
        # cx:           context tokens character list
        # p:            context [["xxx, "xxx"], []]
        # word_counter: context+question word_count
        # lower_word_counter: 
        # char_counter: context+question word_ch_count
        # word2vec:
        # lower_word2vec:

        print("saving online... ")
        self.save_online(data, shared, file_path)

if __name__ == "__main__":
    main()