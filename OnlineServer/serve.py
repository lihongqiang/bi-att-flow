# trans to data json
import pandas as pd
import hashlib
import json
import os
import time
import requests
import threading
import sys
sys.path.append('/home/t-honli/bi-att-flow')

from squad.prepro_class import PreproClass
from basic.OlineServer import OlineTest

ISOTIMEFORMAT='%Y-%m-%d %X'

import gevent

class ServeClass():
    
    def __init__(self):
        #self.OlineTest = OlineTest(out_dir='/home/t-honli/bi-att-flow/out/EQnA/03-07-2017')
        self.prepro = PreproClass()
        self.AnswerByBiDAF = list()
        self.AnswerByRNet = list()

    def generateJson(self, context, query, answer=""):
        online_data = pd.DataFrame({"Query":query, "Context":context, "phrase":answer}, columns=["Query", "Context", "phrase"], index=[0])
        # 只处理lable为1的row
        target_dev_data = {}
        target_dev_data['version'] = '1.1'
        target_dev_data['data'] = list()

        item = {}
        item['paragraphs'] = list()
        item['title'] = 'Online'

        # 处理一个context对应一个quesiton，对应多个answer的情况
        paragraphs_dict = {}
        num = 0
        
        def GetHashCode(context):
            hash = hashlib.md5()
            hash.update(context.encode('utf-8'))
            return hash.hexdigest()
    
        def transEachRow(row):

            # 新生成的context
            paragraph = {}
            paragraph['qas'] = list()
            paragraph['context'] = row['Context'].strip()

            phrase = {}
            phrase['id'] = GetHashCode(row['Context'] + ' ' + row['Query'])
            phrase['question'] = row['Query'].strip()
            phrase['answers'] = list()

            phrase_answer = {}

            row['phrase'] = row['phrase'].strip()  # phrase 可能含有空格
            phrase_answer['text'] = row['phrase']

            # 过滤长度大于32的phrase
            global num
            if len(row['phrase']) >= 32:
                num += 1
                return 

            phrase_answer['answer_start'] = paragraph['context'].find(row['phrase'])
            if phrase_answer['answer_start'] == -1:
                print (paragraph['context'])
                print (row['phrase'])
                print (row)
                print (row.values)
                print 
            phrase['answers'].append(phrase_answer)

            paragraph['qas'].append(phrase)

            paragraph_key = GetHashCode(paragraph['context'])
            # 判断是否存在该context， 一个context只有一个quesiotn
            if paragraph_key not in paragraphs_dict:
                paragraphs_dict[paragraph_key] = paragraph
            else:
                paragraphs_dict[paragraph_key]['qas'][0]['answers'].append(phrase_answer)

        online_data.apply(transEachRow, axis=1)
        item['paragraphs'] = list(paragraphs_dict.values())

        # print (len(list(paragraphs_dict.values())))
        target_dev_data['data'].append(item)

        # print (target_dev_data)
        time_format='%Y-%m-%d_%X'
        name = time.strftime( time_format, time.localtime( time.time() ))  + '_' + GetHashCode(context + '_' + query) + '.json'
        target_dev_path = '/home/t-honli/data/online/' + name
        json.dump(target_dev_data, open(target_dev_path, 'w'))
        return target_dev_path

    # prepro json to test data  
    def preproJson(self):

        prepro_file = 'squad.prepro'
        source_dir = '/home/t-honli/data/online'
        target_dir = '/home/t-honli/bi-att-flow/data/online'
        file_name = 'online.json'

        os.system("python -m {} --online=True --data_file={} --source_dir={} --target_dir={}".format(prepro_file, file_name, source_dir, target_dir))

    def testData(self, num, file_path):
        # test data  python -m basic.cli --len_opt --cluster --data_dir=data/online --online=True
        file_name = file_path.split('/')[-1]
        data_dir = os.path.join('/home/t-honli/bi-att-flow/data/online', file_name.split('.')[0])
        answer_dir = data_dir
        ans_num = num
        out_dir = 'out/EQnA/no_sent_token/01-08-2017'
        os.system("python -m basic.cli --len_opt --cluster --dump_eval=False --data_dir={} --online=True --topk={} --out_dir={} --answer_dir={}".format(data_dir, ans_num, out_dir, answer_dir))
    
    def testDataOline(self, num, file_path):
        # test data  python -m basic.cli --len_opt --cluster --data_dir=data/online --online=True
        file_name = file_path.split('/')[-1]
        data_dir = os.path.join('/home/t-honli/bi-att-flow/data/online', file_name.split('.')[0])
        ans_num = num
        self.OlineTest.main(data_dir, ans_num)
        

    # show answer
    def showAnswer(self, file_name):

        answer_dir = '/home/t-honli/bi-att-flow/out/EQnA/no_sent_token/01-08-2017/answer'
        name = 'online-020000.json'
        answer = json.load(open(os.path.join(answer_dir, name), "r"))
        answer_list = []
        for key,val in list(answer.items()):
            if key != 'scores':
                phrases = val.split('|||')
                scores = answer['scores'][key].split('|||')
                cnt = 1
                for phrase, score in zip(phrases, scores):
                    # print (cnt, phrase, score)
                    answer_list.append([cnt, phrase, score])
                    cnt += 1
        return answer_list
    
    def getAnswerByRNet(self, context, question, num):
        payload = {
            "context":context,
            "question":question,
            "num":num
        }
        try:
            r = requests.post("http://10.172.126.49:5000/demo", data=payload, timeout=50)
        except:
            print ('rnet network error.')
            answer_list = []
            self.AnswerByRNet = answer_list
            return answer_list
        
        answer = json.loads(r.text)['answer']
        answer_list = []
        phrases = answer.split('|||')
        cnt = 1
        for phrase in phrases:
            ans, score = phrase.split(':::')
            answer_list.append([cnt, ans, score])
            cnt += 1

        self.AnswerByRNet = answer_list
        return answer_list    
        
    def getAnswerPhrase(self, context, query, num, answer=" "):
        
        # change workspace
        work_dir = '/home/t-honli/bi-att-flow'
        os.chdir(work_dir)
        
        # <1s
        # print ('build json', time.strftime( ISOTIMEFORMAT, time.localtime( time.time() )))
        file_path = self.generateJson(context, query, answer)

        # <1s search twice glove for word embedding
        # print ('build data', time.strftime( ISOTIMEFORMAT, time.localtime( time.time() )))
        # self.preproJson()
        self.prepro.prepro_online(file_path)

        # 12s run the model in GPU
        print ('tets data', time.strftime( ISOTIMEFORMAT, time.localtime( time.time() )))
        self.testData(num, file_path)
        #self.testDataOline(num, file_path)

        # <1s
        #print ('show data', time.strftime( ISOTIMEFORMAT, time.localtime( time.time() )))
        ans = self.showAnswer(file_path)
        print ('finish ', time.strftime( ISOTIMEFORMAT, time.localtime( time.time() )))
        
        self.AnswerByBiDAF = ans
        return ans
    
    def getAllAnswer(self, context, question, num, answer=" "):
        # self.getAnswerByRNet(context, question, num)
        # self.getAnswerPhrase(context, question, num, answer)
        # print (self.AnswerByBiDAF)
        # print (self.AnswerByRNet)
        print ('getAllAnswer start', time.strftime( ISOTIMEFORMAT, time.localtime( time.time() )))
        thread_list = []    #线程存放列表
        
        t1 = threading.Thread(target=self.getAnswerByRNet,args=(context, question, num))
        t1.setDaemon(True)
        thread_list.append(t1)
        
        t2 = threading.Thread(target=self.getAnswerPhrase,args=(context, question, num, answer))
        t2.setDaemon(True)
        thread_list.append(t2)
        
        for t in thread_list:
            t.start()
        for t in thread_list:
            t.join()
            
        #thread1 = gevent.spawn(self.getAnswerByRNet, context, question, num)
        #thread2 = gevent.spawn(self.getAnswerPhrase, context, question, num, answer)
        #gevent.joinall([thread1, thread2])
        print ('getAllAnswer end', time.strftime( ISOTIMEFORMAT, time.localtime( time.time() )))
        return self.AnswerByBiDAF, self.AnswerByRNet