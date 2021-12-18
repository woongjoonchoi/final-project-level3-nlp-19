# from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
# from sqlalchemy.orm import relationship

# from .database import Base


# # 회원 : 아이디(PK) 비밀번호 이름 알람설정
# class User(Base):

#     # 테이블의 이름
#     __tablename__ = "users"

#     id = Column(Integer, primary_key=True, index=True) # index ?
#     hashed_password = Column(String)
#     name = Column(String, index=True)
#     alarm = Column(Boolean, default=True) # 일단 Boolean

#     scraps = relationship("Scrap", back_populates="owner_user")


# # 관리자 : 아이디(PK) 비밀번호
# class Admin(Base):

#     __tablename__ = "admins"

#     id = Column(Integer, primary_key=True, index=True)
#     hashed_password = Column(String)


# # 뉴스 : 아이디(PK) 제목 내용 스크랩(FK)
# class News(Base):

#     __tablename__ = "newss"

#     id = Column(Integer, primary_key=True, index=True)
#     title = Column(String, index=True)
#     article = Column(String, index=True)
#     owner_scrap_id = Column(Integer, ForeignKey("scraps.id"))
    
#     owner_scrap = relationship("Scrap", back_populates="newss")


# # 스크랩 : 아이디(PK) 질문뉴스_아이디(FK) 질문문장 답변뉴스_아이디(FK) 답변문장 작성자(FK)
# class Scrap(Base):

#     __tablename__ = "scraps"

#     id = Column(Integer, primary_key=True, index=True)
#     question_news_id = Column(Integer, index=True)
#     question_sentence = Column(String, index=True)
#     answer_news_id = Column(Integer, index=True)
#     answer_sentence = Column(String, index=True)
#     owner_user_id = Column(Integer, ForeignKey("users.id"))

#     owner_user = relationship("User", back_populates="scraps")
#     newss = relationship("News", back_populates="owner_scrap")

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .database import Base


# User : 유저아이디(PK) 비밀번호 이름 알람설정
class User(Base):

    # 테이블의 이름
    __tablename__ = "users"

    user_id = Column(String, primary_key=True, index=True) # index?
    hashed_password = Column(String)
    name = Column(String, index=True)
    alarm = Column(Boolean, default=False) # 일단 Boolean


# Admin : 관리자아이디(PK) 비밀번호
class Admin(Base):

    __tablename__ = "admins"

    admin_id = Column(String, primary_key=True, index=True)
    hashed_password = Column(String)


# UserNews : 유저와AI가보는뉴스아이디(PK) 제목 내용 관리자아이디(FK)
# 모든 유저가 볼 수 있는 뉴스
class UserNews(Base):

    __tablename__ = "user_newss"

    user_news_id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    article = Column(String, index=True)
    admin_id = Column(String, ForeignKey("admins.admin_id"))


# NewsScrap : news_scrap_id(PK) 유저아이디(FK) 유저가보는뉴스아이디(FK)
# 유저가 뉴스 스크랩한 뉴스
class NewsScrap(Base):

    __tablename__ = "news_scraps"

    news_scrap_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.user_id"))
    user_news_id = Column(Integer, ForeignKey("user_newss.user_news_id"))


# UserInput : user_input_id(PK) 유저아이디(FK) 유저가보는뉴스아이디(FK) 유저질문문장
# 유저가 입력한 question 관리
class UserInput(Base):

    __tablename__ = "user_input"

    user_input_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.user_id"))
    user_news_id = Column(Integer, ForeignKey("user_newss.user_news_id"))
    user_input = Column(String, index=True)


# DataFlow 상에서 삭제됨
# # AINewsScrap : ai_news_scraps_id(PK) 유저아이디(FK) AI가보는뉴스아이디(FK)
# # AI가 스크랩한 뉴스
# class AINewsScrap(Base):

#     __tablename__ = "ai_news_scraps"

#     ai_news_scraps_id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(String, ForeignKey("users.user_id"))
#     ai_news_id = Column(Integer, ForeignKey("user_newss.user_news_id"))


# AIInput : ai_input_id(PK) 유저아이디(FK) AI가보는뉴스아이디(FK) AI답변문장
# AI가 내놓은 답변
class AIInput(Base):

    __tablename__ = "ai_inputs"

    ai_input_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.user_id"))
    ai_news_id = Column(Integer, ForeignKey("user_newss.user_news_id"))
    ai_input = Column(String, index=True)
    
