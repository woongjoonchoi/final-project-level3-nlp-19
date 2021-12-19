from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from .database import Base


class Question(Base):
    __tablename__ = 'questions'

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)




# 회원 : 아이디(PK) 비밀번호 이름 알람설정
class User(Base):

    # 테이블의 이름
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True) # index ?
    hashed_password = Column(String)
    name = Column(String, index=True)
    alarm = Column(Boolean, default=True) # 일단 Boolean

    scraps = relationship("Scrap", back_populates="owner_user")


# 관리자 : 아이디(PK) 비밀번호
class Admin(Base):

    __tablename__ = "admins"

    id = Column(Integer, primary_key=True, index=True)
    hashed_password = Column(String)


# 뉴스 : 아이디(PK) 제목 내용 스크랩(FK)
class News(Base):

    __tablename__ = "newss"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    article = Column(String, index=True)
    owner_scrap_id = Column(Integer, ForeignKey("scraps.id"))
    
    owner_scrap = relationship("Scrap", back_populates="newss")


# 스크랩 : 아이디(PK) 질문뉴스_아이디(FK) 질문문장 답변뉴스_아이디(FK) 답변문장 작성자(FK)
class Scrap(Base):

    __tablename__ = "scraps"

    id = Column(Integer, primary_key=True, index=True)
    question_news_id = Column(Integer, index=True)
    question_sentence = Column(String, index=True)
    answer_news_id = Column(Integer, index=True)
    answer_sentence = Column(String, index=True)
    owner_user_id = Column(Integer, ForeignKey("users.id"))

    owner_user = relationship("User", back_populates="scraps")
    newss = relationship("News", back_populates="owner_scrap")