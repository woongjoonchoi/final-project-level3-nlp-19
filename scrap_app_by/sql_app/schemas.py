from typing import List, Optional
from pydantic import BaseModel


# Create에 id도 들어가야하나?
# 관리자 아이디가 뉴스에 들어가야하나?


# 관리자 : 아이디(PK) 비밀번호
class AdminBase(BaseModel):
    id: int


class AdminCreate(AdminBase):
    password: str


class Admin(AdminBase):
    pass

    class Config:
        orm_mode = True


# 뉴스 : 아이디(PK) 제목 내용
class NewsBase(BaseModel):
    title: str
    article: str


class NewsCreate(NewsBase):
    pass


class News(NewsBase):
    id: int
    owner_scrap_id: int

    class Config:
        orm_mode = True


# 스크랩 : 아이디(PK) 작성자(FK) 질문뉴스_아이디(FK) 질문문장 답변뉴스_아이디(FK) 답변문장
class ScrapBase(BaseModel):
    question_sentence: str
    answer_sentence: Optional[str] = None


class ScrapCreate(ScrapBase):
    pass


# ! 뉴스 id
class Scrap(ScrapBase):
    id: int
    question_news: List[News] = []
    answer_news: List[News] = []
    owner_user_id: int

    class Config:
        orm_mode = True


# 회원 : 아이디(PK) 비밀번호 이름 알람설정
class UserBase(BaseModel):
    id: int


class UserCreate(UserBase):
    password: str


class User(UserBase):
    name: Optional[str]
    alarm: Optional[bool] = False
    scraps: List[Scrap] = []

    class Config:
        orm_mode = True