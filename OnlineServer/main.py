import json
from flask import Flask
from flask import render_template
from flask_bootstrap import Bootstrap
from serve import ServeClass
app = Flask(__name__)
bootstrap = Bootstrap(app)

# 表单提交
from wtforms import Form, BooleanField, TextField, PasswordField, validators
from flask import request

servecls = ServeClass()

class ContentForm(Form):
    context = TextField('context')
    question = TextField('question')
    num = TextField('num')

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', name='lhq')

def submit_add(context, question, bidaf, num, rnet):
    # 创建session对象:
    session = DBSession()
    submit = Submit(context=context, question=question, bidaf=bidaf, rnet=rnet, num=num)
    session.add(submit)
    # 提交即保存到数据库:
    session.commit()
    # 关闭session:
    session.close()
    return True

@app.route("/demo", methods=['POST', 'GET'])
def demo():
    #servecls = ServeClass()
    if request.method == 'POST':
        form = ContentForm(request.form)
        context = form.context.data
        question = form.question.data
        answer = context.strip().split(' ')[0]
        num = form.num.data
        # print (context)
        # print (question)
        # print (num)
        # answer = servecls.getAnswerPhrase(context, question, num, answer)
        
        # RNet
        # answer_RNet = servecls.getAnswerByRNet(context, question, num)
        # print ("answer_Rnet = ", answer_RNet)
        
        AnswerByBiDAF, AnswerByRNet = servecls.getAllAnswer(context, question, num, answer)
        
        print ("bidaf: ", AnswerByBiDAF)
        print ("rnet: ",  AnswerByRNet)
        
        # add to sql
        cnt, bidaf_answers, bidaf_probability = zip(*AnswerByBiDAF)
        
        if not AnswerByRNet:
            AnswerByRNet = [['None', 'None', 'None'] for i in range(len(AnswerByBiDAF))]

        cnt, rnet_answers, rnet_probability = zip(*AnswerByRNet)
        
        submit_add(context, question, '|||'.join([ ans+':::'+score for ans, score in zip(bidaf_answers, bidaf_probability)]), num, '|||'.join([ans+':::'+score for ans, score in zip(rnet_answers, rnet_probability)]))
        
        
        # save the data
        data = {"context":context, "question":question, "num":num}
        answer = list()
        for bidaf, rnet in zip(AnswerByBiDAF, AnswerByRNet):
            answer.append({"cnt": bidaf[0], "bidaf":bidaf[1]+":::"+bidaf[2], "rnet":rnet[1]+":::"+rnet[2]})
        return render_template('demo.html', answer=answer, data=data)
    else:
        answer=[]
        data={}
        return render_template('demo.html', answer=answer, data=data)

# 查看某个特定的查询
@app.route("/example/<int:id>")
@app.route("/example")
def example(id=209):
    # 创建session对象:
    session = DBSession()
    submit = session.query(Submit).filter(Submit.id==id).one()
    data = {"context":submit.context, "question":submit.question, "num":submit.num}
    AnswerByBiDAF = submit.bidaf.split('|||')
    AnswerByRNet = submit.rnet.split('|||')
    AnswerByProd = None
    if submit.prod:
        AnswerByProd = submit.prod.split('|||')
    answer = list()
    cnt = 1
    if AnswerByProd:
        for bidaf, rnet, prod in zip(AnswerByBiDAF, AnswerByRNet, AnswerByProd):
            bidaf = bidaf.split(':::')
            rnet = rnet.split(':::')
            prod = prod.split(':::')
            answer.append({"cnt": cnt, "bidaf":bidaf[0]+":::"+bidaf[1], "rnet":rnet[0]+":::"+rnet[1], "prod":prod[0]+":::"+prod[1]})
            cnt += 1
    else:
        for bidaf, rnet in zip(AnswerByBiDAF, AnswerByRNet):
            bidaf = bidaf.split(':::')
            rnet = rnet.split(':::')
            answer.append({"cnt": cnt, "bidaf":bidaf[0]+":::"+bidaf[1], "rnet":rnet[0]+":::"+rnet[1]})
            cnt += 1
    # 关闭session:
    session.close()
    return render_template('example.html', answer=answer, data=data, prod=(AnswerByProd is not None))

@app.route("/history", methods=['GET'])
def history(name=None):
    return render_template('history.html', name=name)


## history 访问数据库
from model import DBSession, Submit

@app.route("/history_data", methods=['GET'])
def history_data(name=None):
    
    # 创建session对象:
    session = DBSession()   
    submits = session.query(Submit).order_by(Submit.date.desc()).all()
    history_data = []
    for submit in submits:
        history_data.append({"id":submit.id, "datetime":str(submit.date), "context":submit.context, "question":submit.question, "num":submit.num, "bidaf":submit.bidaf, "rnet":str(submit.rnet)})
    # 关闭session:
    session.close()
    return json.dumps({"data":history_data})

@app.route("/history_add", methods=['GET'])
def history_add():
    # 创建session对象:
    session = DBSession()
    
    for index in range(28):
        submit = Submit(context='context{}'.format(index), question='{}'.format(index), answer='{}'.format(index), probability=1.0, num=1)
        session.add(submit)
    # 提交即保存到数据库:
    session.commit()
    # 关闭session:
    session.close()
    return "ok"

    
if __name__ == '__main__':
    app.run(debug=True)
