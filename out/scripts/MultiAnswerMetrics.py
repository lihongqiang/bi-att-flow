
def get_dev(dev_file = '/home/t-honli/data/EQnA/dev-v1.1.json', \
            bidaf_file = '/home/t-honli/bi-att-flow/out/EQnA/no_sent_token/01-08-2017/answer/test-020000.json', \
            prod_file = '/home/t-honli/data/EQnA/EQnA_Highlighting_Test_ProdModel.tsv'):
    
    # parse dev data function
    def ParseJson(data_json):
        paragraphs = data_json['data'][0]['paragraphs']
        idx_list = list()
        context_list = list()
        ques_list = list()
        ans_list = list()
        for para in paragraphs:
            context = para['context']
            idx = para['qas'][0]['id']
            ques = para['qas'][0]['question']
            ans = '|||'.join([ans['text'] for ans in para['qas'][0]['answers']])
            idx_list.append(idx)
            context_list.append(context)
            ques_list.append(ques)
            ans_list.append(ans)
        return idx_list, context_list, ques_list, ans_list

    # dev data
    import json

    dev_data = json.load(open(dev_file, "r"))
    import pandas as pd
    idx, context, ques, ans = ParseJson(dev_data)
    dev_pd = pd.DataFrame({'id':idx, 'context':context, 'question':ques, 'ground_truth':ans}, columns=['id', 'context', 'question', 'ground_truth'])

    # bidaf ans
    import json
    import os

    bidaf = json.load(open(bidaf_file, "r"))
    dev_pd['bidaf'] = dev_pd.apply(lambda row: '|||'.join([ phrase+':::'+score for phrase, score in  zip(str(bidaf[row['id']]).split('|||'), str(bidaf['scores'][row['id']]).split('|||'))          ]), axis=1)

    # prod model
    import pandas as pd
    col_names = ['id', 'Query', 'Url', 'Answer', 'AnswerTokenList', 'ParaseSpan', 'phrase', 'Label', 'Probability']
    prod = pd.read_csv(prod_file, header=None, sep='\t', names=col_names, dtype=str).fillna('')
    # 根据answer，query，url生成hash_id
    import re
    def getAnswerByTokenList(s):
        slist = ([ wd.strip() for wd in s.strip('[]').split('\",\"')])
        slist = ' '.join(slist)
        slist = re.sub(r'\\\"', '\"', slist)
        slist = re.sub(r'\\\'', '\'', slist)
        slist = slist.strip('\"')
        return slist
    import hashlib
    def GetHashCode(context):
        hash = hashlib.md5()
        hash.update(context.encode('utf-8'))
        return hash.hexdigest()
    prod['hash_id'] = prod.apply(lambda row: GetHashCode(getAnswerByTokenList(row['AnswerTokenList'].strip()) + ' ' + row['Query'].strip()), axis=1)
    # filter len >= 32
    prod = prod[prod.apply(lambda row: len(row['phrase'])<32, axis=1)]
    # 获取hash_id, [(phrase,score)]
    multi_phrase_dict = {}
    def getPhrase(row):
        if row['hash_id'] not in multi_phrase_dict:
            multi_phrase_dict[row['hash_id']] = list()
        row['Probability'] = '%.4f' % float(row['Probability'])
        multi_phrase_dict[row['hash_id']].append(row['phrase'] + ":::" + row['Probability'])
    prod.apply(getPhrase, axis=1)
    # sort
    for idx in list(multi_phrase_dict.keys()):
        multi_phrase_dict[idx].sort(key=lambda x: float(x.split(':')[-1]), reverse=True)
    dev_pd['prod'] = dev_pd.apply(lambda row: '|||'.join(multi_phrase_dict[row['id']]), axis=1)
    
    return dev_pd

# 计算ground truth和bidaf的集合的P/R, 宏平均
# P = intersection/bidaf
# R = intersection/ground_truth
# 采用cover的方式

# 是否选择多answer的阈值
# choose_multi_answer_threshold = 0.4
# 第二个及之后的answer的阈值
# multi_answer_threshold = 0.5
# 计算prod多answer的阈值
# prod_answer_threshold=0.6

def calPandR_Macro(dev_pd, choose_multi_answer_threshold = 0.4, \
                    multi_answer_threshold = 0.5, \
                    prod_answer_threshold=0.6):
    import string
    import re
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    # with tricks
    # 这里根据第一个slot的score来判断是否是多answer，threshold=0.4, （是否根据score>0.36选择后面的answer?）
    def getPandR(row):
        spans = row['bidaf'].split('|||')
        grounds = row['ground_truth'].split('|||')
        bidafs = []
        if float(spans[0].split(':::')[1]) > choose_multi_answer_threshold:
            bidafs.append(spans[0].split(':::')[0])
        else:
            for idx, span in enumerate(spans):
                ans, score = span.split(':::')
                if idx == 0:
                    bidafs.append(ans)
                else:
                    if float(score) > multi_answer_threshold:
                        bidafs.append(ans)
                    else:
                        break
        inter = 0
        for bidaf in bidafs:
            for ground in grounds:
                bidaf = normalize_answer(bidaf)
                ground = normalize_answer(ground)
                if bidaf in ground or ground in bidaf:
                    inter += 1
                    break
        return [1.0*inter/len(bidafs), 1.0*inter/len(grounds)]

    # without tricks, 至少有一个
    def getPandR_withoutTricks(row):
        grounds = row['ground_truth'].split('|||')
        bidafs = []
        for idx, span in enumerate(row['bidaf'].split('|||')):
            ans, score = span.split(':::')
            if idx == 0:
                bidafs.append(ans)
            else:
                if float(score) > multi_answer_threshold:
                    bidafs.append(ans)
                else:
                    break
        inter = 0
        for bidaf in bidafs:
            for ground in grounds:
                bidaf = normalize_answer(bidaf)
                ground = normalize_answer(ground)
                if bidaf in ground or ground in bidaf:
                    inter += 1
                    break
        return [1.0*inter/len(bidafs), 1.0*inter/len(grounds)]

    #threshold=0.6
    def getPandR_prod(row):
        grounds = row['ground_truth'].split('|||')
        prods = []
        for idx, span in enumerate(row['prod'].split('|||')):
            ans, score = span.split(':::')
            if idx == 0:
                prods.append(ans)
            else:
                if float(score) > prod_answer_threshold:
                    prods.append(ans)
                else:
                    break
        inter = 0
        for prod in prods:
            for ground in grounds:
                prod = normalize_answer(prod)
                ground = normalize_answer(ground)
                if prod in ground or ground in prod:
                    inter += 1
                    break
        return [1.0*inter/len(prods), 1.0*inter/len(grounds)]

    bidaf_P, bidaf_R = zip(*dev_pd.apply(getPandR, axis=1).values)
    P = sum(bidaf_P)/len(dev_pd)
    R = sum(bidaf_R)/len(dev_pd)
    print ('##############')
    print ('Macro Average:')
    print ('Tricks on choose multi answer: choose_multi_answer_threshold = {}, multi_answer_threshold = {}'.format(choose_multi_answer_threshold, multi_answer_threshold))
    print ('Bidaf P: ', P)
    print ('Bidaf R: ', R)
    print ('Bidaf F1: ', 2*P*R/(P+R))
    print ()

    bidaf_P, bidaf_R = zip(*dev_pd.apply(getPandR_withoutTricks, axis=1).values)
    P = sum(bidaf_P)/len(dev_pd)
    R = sum(bidaf_R)/len(dev_pd)
    print ('Normal: choose first answer, multi_answer_threshold = {}'.format(multi_answer_threshold))
    print ('Bidaf P: ', P)
    print ('Bidaf R: ', R)
    print ('Bidaf F1: ', 2*P*R/(P+R))
    print ()

    prod_P, prod_R = zip(*dev_pd.apply(getPandR_prod, axis=1).values)
    P = sum(prod_P)/len(dev_pd)
    R = sum(prod_R)/len(dev_pd)
    print ('Prod: choose first answer, multi_answer_threshold = {}'.format(prod_answer_threshold))
    print ('Bidaf P: ', P)
    print ('Bidaf R: ', R)
    print ('Bidaf F1: ', 2*P*R/(P+R))
    print ()


# 计算ground truth和bidaf的集合的P/R, 微平均
# P = intersection/bidaf
# R = intersection/ground_truth
# 采用cover的方式

# 是否选择多answer的阈值
# choose_multi_answer_threshold = 0.4
# 第二个及之后的answer的阈值
# multi_answer_threshold = 0.5
# 计算prod多answer的阈值
# prod_answer_threshold=0.6
inter_num = 0
ground_num = 0
bidaf_num = 0
    
def calPandR_Micro(dev_pd, choose_multi_answer_threshold = 0.4, \
                    multi_answer_threshold = 0.5, \
                    prod_answer_threshold=0.6):
    global inter_num, bidaf_num, ground_num
    import string
    import re
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    # with tricks
    # 这里根据第一个slot的score来判断是否是多answer，threshold=0.4, 最多选择3个answer（是否根据score>0.36选择后面的answer?）

    def getPandR(row):
        global inter_num, bidaf_num, ground_num
        spans = row['bidaf'].split('|||')
        grounds = row['ground_truth'].split('|||')
        bidafs = []
        if float(spans[0].split(':::')[1]) > choose_multi_answer_threshold:
            bidafs.append(spans[0].split(':::')[0])
        else:
            for idx, span in enumerate(spans):
                ans, score = span.split(':::')
                if idx == 0:
                    bidafs.append(ans)
                else:
                    if float(score) > multi_answer_threshold:
                        bidafs.append(ans)
                    else:
                        break
        inter = 0
        for bidaf in bidafs:
            for ground in grounds:
                bidaf = normalize_answer(bidaf)
                ground = normalize_answer(ground)
                if bidaf in ground or ground in bidaf:
                    inter += 1
                    break

        inter_num += inter
        bidaf_num += len(bidafs)
        ground_num += len(grounds)

        return [1.0*inter/len(bidafs), 1.0*inter/len(grounds)]

    # without tricks, 至少有一个
    def getPandR_withoutTricks(row):
        global inter_num, bidaf_num, ground_num
        grounds = row['ground_truth'].split('|||')
        bidafs = []
        for idx, span in enumerate(row['bidaf'].split('|||')):
            ans, score = span.split(':::')
            if idx == 0:
                bidafs.append(ans)
            else:
                if float(score) > multi_answer_threshold:
                    bidafs.append(ans)
                else:
                    break
        inter = 0
        for bidaf in bidafs:
            for ground in grounds:
                bidaf = normalize_answer(bidaf)
                ground = normalize_answer(ground)
                if bidaf in ground or ground in bidaf:
                    inter += 1
                    break

        inter_num += inter
        bidaf_num += len(bidafs)
        ground_num += len(grounds)
        return [1.0*inter/len(bidafs), 1.0*inter/len(grounds)]
    
    def getPandR_prod(row):
        global inter_num, bidaf_num, ground_num
       
        grounds = row['ground_truth'].split('|||')
        prods = []
        for idx, span in enumerate(row['prod'].split('|||')):
            ans, score = span.split(':::')
            if idx == 0:
                prods.append(ans)
            else:
                if float(score) > prod_answer_threshold:
                    prods.append(ans)
                else:
                    break
        inter = 0
        for prod in prods:
            for ground in grounds:
                prod = normalize_answer(prod)
                ground = normalize_answer(ground)
                if prod in ground or ground in prod:
                    inter += 1
                    break

        inter_num += inter
        bidaf_num += len(prods)
        ground_num += len(grounds)
        return [1.0*inter/len(prods), 1.0*inter/len(grounds)]

    inter_num = 0
    ground_num = 0
    bidaf_num = 0
    bidaf_P, bidaf_R = zip(*dev_pd.apply(getPandR, axis=1).values)
    P = 1.0*inter_num/bidaf_num
    R = 1.0*inter_num/ground_num
    
    print ('##############')
    print ('Micro Average:')
    print ('Tricks on choose multi answer: choose_multi_answer_threshold = {}, multi_answer_threshold = {}'.format(choose_multi_answer_threshold, multi_answer_threshold))
    print ('Bidaf P: ', P)
    print ('Bidaf R: ', R)
    print ('Bidaf F1: ', 2*P*R/(P+R))
    print ()

    inter_num = 0
    ground_num = 0
    bidaf_num = 0
    bidaf_P, bidaf_R = zip(*dev_pd.apply(getPandR_withoutTricks, axis=1).values)
    P = 1.0*inter_num/bidaf_num
    R = 1.0*inter_num/ground_num
    print ('Normal: choose first answer, multi_answer_threshold = {}'.format(multi_answer_threshold))
    print ('Bidaf P: ', P)
    print ('Bidaf R: ', R)
    print ('Bidaf F1: ', 2*P*R/(P+R))
    print ()

    inter_num = 0
    ground_num = 0
    bidaf_num = 0
    prod_P, prod_R = zip(*dev_pd.apply(getPandR_prod, axis=1).values)
    P = 1.0*inter_num/bidaf_num
    R = 1.0*inter_num/ground_num
    print ('Prod: choose first answer, multi_answer_threshold = {}'.format(prod_answer_threshold))
    print ('Bidaf P: ', P)
    print ('Bidaf R: ', R)
    print ('Bidaf F1: ', 2*P*R/(P+R))

# dev_file = '/home/t-honli/data/EQnA/dev-v1.1.json'
# bidaf_file = '/home/t-honli/bi-att-flow/out/EQnA/no_sent_token/01-08-2017/answer/test-020000.json'
# prod_file = '/home/t-honli/data/EQnA/EQnA_Highlighting_Test_ProdModel.tsv'

# # 是否选择多answer的阈值
# choose_multi_answer_threshold = 0.4
# # 第二个及之后的answer的阈值
# multi_answer_threshold = 0.5
# # 计算prod多answer的阈值
# prod_answer_threshold=0.6

import sys
if __name__ == '__main__':
    if len(sys.argv) == 1:
        dev_pd = get_dev()
    elif len(sys.argv) == 4:
        dev_file = argv[1]
        bidaf_file = argv[2]
        prod_file = argv[3]
        dev_pd = get_dev(dev_file, bidaf_file, prod_file)
    elif len(sys.argv) == 7:
        dev_file = argv[1]
        bidaf_file = argv[2]
        prod_file = argv[3]
        choose_multi_answer_threshold = float(argv[4])
        multi_answer_threshold = float(argv[5])
        prod_answer_threshold = float(argv[6])
    else:
        print ('Please input 3 parameters: dev_file, bidaf_file, prod_file \nor 6 parameters: dev_file, bidaf_file, prod_file, choose_multi_answer_threshold, multi_answer_threshold, prod_answer_threshold')
        exit(0)
    calPandR_Macro(dev_pd)
    calPandR_Micro(dev_pd)
