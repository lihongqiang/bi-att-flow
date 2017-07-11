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

def submit_add(context, question, answer, num, probability):
    # 创建session对象:
    session = DBSession()
    submit = Submit(context=context, question=question, answer=answer, probability=probability, num=num)
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
        print (context)
        print (question)
        print (num)
        answer = servecls.getAnswerPhrase(context, question, num, answer)
        # add to sql
        cnt, answers, probability = zip(*answer)
        print (cnt)
        print (answers)
        print (probability)
        submit_add(context, question, '||'.join(answers), num, '||'.join([str(pro) for pro in probability]))
        # save the data
        data = {"context":context, "question":question, "num":num, "answer":answer}
        return render_template('demo.html', answer=answer, data=data)
    else:
        answer=[]
        data={}
        return render_template('demo.html', answer=answer, data=data)

@app.route("/example")
def example(name=None):
    return render_template('example.html', name=name)

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
        history_data.append({"datetime":str(submit.date), "context":submit.context, "question":submit.question, "num":submit.num, "answers":submit.answer, "probability":str(submit.probability)})
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
