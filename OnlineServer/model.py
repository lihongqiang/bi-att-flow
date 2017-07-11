# coding: utf8
from sqlalchemy import Column, Integer, Float, DateTime, String, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import datetime

# 创建对象的基类:
Base = declarative_base()

# 初始化数据库连接:
engine = create_engine('mysql+mysqlconnector://lhq:lhq@localhost:3306/online')

# 创建DBSession类型:
DBSession = sessionmaker(bind=engine)

####################
# 开始制作模型
####################
class Submit(Base):
    # 表的名字:
    __tablename__ = 'history'
    
    id = Column(Integer(), primary_key=True)
    context = Column(String(255))
    question = Column(String(255))
    answer = Column(String(255))
    probability = Column(String(255))
    num = Column(Integer())
    date = Column(DateTime())

    def __init__(self, context, question, answer, probability, num):
        self.context = context
        self.question = question
        self.answer = answer
        self.probability = probability
        self.num = num
        self.date = datetime.datetime.now()

    def __repr__(self):
        return "context:`{}`\n question:`{}`\n answer:{}\n".format(self.context, self.question, self.answer)