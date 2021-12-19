from typing import List, Optional
from pydantic import BaseModel
from uuid import UUID


class Question(BaseModel):

    id: UUID
    text: str


# Create에 id도 들어가야하나?
# 관리자 아이디가 뉴스에 들어가야하나?


# Admin : 관리자아이디(PK) 비밀번호
class AdminBase(BaseModel):
    admin_id: str


class AdminCreate(AdminBase):
    password: str


class Admin(AdminBase):
    pass

    class Config:
        orm_mode = True


# UserNews : 유저와AI가보는뉴스아이디(PK) 제목 내용 관리자아이디(FK)
# 모든 유저가 볼 수 있는 뉴스
class UserNewsBase(BaseModel):
    user_news_id: int
    title: str
    article: str


class UserNewsCreate(UserNewsBase):
    pass


class UserNews(UserNewsBase):
    admin_id: Optional[str]

    class Config:
        orm_mode = True


# NewsScrap : 유저아이디(FK) 유저가보는뉴스아이디(FK)
# 유저가 뉴스 스크랩한 뉴스
class NewsScrapBase(BaseModel):
    news_scrap_id: int
    user_news_id: int

class NewsScrapCreate(NewsScrapBase):
    pass


class NewsScrapDelete(NewsScrapBase):
    pass


class NewsScrap(NewsScrapBase):
    user_id: str
    
    class Config:
        orm_mode = True


# UserInput : 유저아이디(FK) 유저가보는뉴스아이디(FK) 유저질문문장
# 유저가 입력한 question 관리
class UserInputBase(BaseModel):
    user_id: str
    user_news_id: int
    user_input: str


class UserInputCreate(UserInputBase):
    pass


class UserInput(UserInputBase):
    pass

    class Config:
        orm_mode = True


# DataFlow 상에서 삭제됨
# # AINewsScrap : 유저아이디(FK) AI가보는뉴스아이디(FK)
# # AI가 스크랩한 뉴스
# class AINewsScrapBase(BaseModel):
#     user_id: str
#     ai_news_id: int


# class AINewsScrapCreate(AINewsScrapBase):
#     pass


# class AINewsScrap(AINewsScrapBase):
#     pass
    
#     class Config:
#         orm_mode = True


# AIInput : 유저아이디(FK) AI가보는뉴스아이디(FK) AI답변문장
# AI가 내놓은 답변
class AIInputBase(BaseModel):
    news_scrap_id: int
    user_id: str
    ai_news_id: int
    ai_input: str


class AIInputCreate(AIInputBase):
    pass


class AIInput(AIInputBase):
    pass

    class Config:
        orm_mode = True


# User : 유저아이디(PK) 비밀번호 이름 알람설정
class UserBase(BaseModel):
    user_id: str


class UserDelete(UserBase):
    password: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    name: Optional[str]
    alarm: Optional[bool] = False

    class Config:
        orm_mode = True


