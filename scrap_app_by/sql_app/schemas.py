# from typing import List, Optional
# from pydantic import BaseModel


# # Create에 id도 들어가야하나?
# # 관리자 아이디가 뉴스에 들어가야하나?


# # 관리자 : 아이디(PK) 비밀번호
# class AdminBase(BaseModel):
#     id: int


# class AdminCreate(AdminBase):
#     password: str


# class Admin(AdminBase):
#     pass

#     class Config:
#         orm_mode = True


# # 뉴스 : 아이디(PK) 제목 내용
# class NewsBase(BaseModel):
#     title: str
#     article: str


# class NewsCreate(NewsBase):
#     pass


# class News(NewsBase):
#     id: int
#     owner_scrap_id: int

#     class Config:
#         orm_mode = True


# # 스크랩 : 아이디(PK) 작성자(FK) 질문뉴스_아이디(FK) 질문문장 답변뉴스_아이디(FK) 답변문장
# class ScrapBase(BaseModel):
#     question_sentence: str
#     answer_sentence: Optional[str] = None


# class ScrapCreate(ScrapBase):
#     pass


# # ! 뉴스 id
# class Scrap(ScrapBase):
#     id: int
#     question_news: List[News] = []
#     answer_news: List[News] = []
#     owner_user_id: int

#     class Config:
#         orm_mode = True


# # 회원 : 아이디(PK) 비밀번호 이름 알람설정
# class UserBase(BaseModel):
#     id: int


# class UserCreate(UserBase):
#     password: str


# class User(UserBase):
#     name: Optional[str]
#     alarm: Optional[bool] = False
#     scraps: List[Scrap] = []

#     class Config:
#         orm_mode = True


from typing import List, Optional
from pydantic import BaseModel


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
    pass


class NewsScrapCreate(NewsScrapBase):
    pass


class NewsScrap(NewsScrapBase):
    pass
    
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


# AINewsScrap : 유저아이디(FK) AI가보는뉴스아이디(FK)
# AI가 스크랩한 뉴스
class AINewsScrapBase(BaseModel):
    user_id: str
    ai_news_id: int


class AINewsScrapCreate(AINewsScrapBase):
    pass


class AINewsScrap(AINewsScrapBase):
    pass
    
    class Config:
        orm_mode = True


# AIInput : 유저아이디(FK) AI가보는뉴스아이디(FK) AI답변문장
# AI가 내놓은 답변
class AIInputBase(BaseModel):
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


class UserCreate(UserBase):
    password: str


class User(UserBase):
    name: Optional[str]
    alarm: Optional[bool]
    news_scraps: List[NewsScrap] = [] # 유저 스크랩 페이지에서 볼 것이기 때문에
    ai_news_scraps: List[AINewsScrap] = [] # AI 스크랩 페이지에서 볼 것이기 때문에

    class Config:
        orm_mode = True
