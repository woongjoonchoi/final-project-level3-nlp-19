from sqlalchemy.orm import Session
from . import models, schemas

# ! 뉴스 기사 수로 바꿔야 함

# User : 유저아이디(PK) 비밀번호 이름 알람설정
# 필터로 내가 선택한 유저만
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.user_id == user_id).first()


# 모든 유저
def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


# 유저 직접 삭제
def delete_user(db: Session, user_delete: schemas.UserDelete, user_id: str):
    db_user = models.User(**user_delete.dict(), user_id=user.user_id)
    db.delete(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


# 유저 직접 만듦
def create_user(db: Session, user: schemas.UserCreate):
    fake_hashed_password = user.password + "notreallyhashed"
    db_user = models.User(user_id=user.user_id, hashed_password=fake_hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


# Admin : 관리자아이디(PK) 비밀번호
# (해당 기능 필요 없음 -> 필터로 내가 선택한 관리자만 보기)
def get_admin(db: Session, admin_id: int):
    return db.query(models.Admin).filter(models.Admin.admin_id == admin_id).first()


# (해당 기능 필요 없음 -> 필터로 내가 선택한 관리자만 보기)
def get_admins(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Admin).offset(skip).limit(limit).all()


# 관리자 생성
def create_admin(db: Session, admin: schemas.AdminCreate):
    fake_hashed_password = admin.password + "notreallyhashed"
    db_admin = models.Admin(id=admin.admin_id, hashed_password=fake_hashed_password)
    db.add(db_admin)
    db.commit()
    db.refresh(db_admin)
    return db_admin


# 웅준님 파트
# UserNews : 유저와AI가보는뉴스아이디(PK) 제목 내용 관리자아이디(FK)
# 모든 유저가 볼 수 있는 뉴스
# def get_user_news(db: Session, user_news_id: int):
#     return db.query(models.UserNews).filter(models.UserNews.id == user_news_id).first()


# def get_user_newss(db: Session, skip: int = 0, limit: int = 100):
#     return db.query(models.UserNews).offset(skip).limit(limit).all()


# def create_user_news(db: Session, news: schemas.NewsScrapCreate, user_news_id: int):
#     db_news = models.UserNews(**news.dict(), user_news_id=scrap_id)
#     db.add(db_news)
#     db.commit()
#     db.refresh(db_news)
#     return db_news


# NewsScrap : 유저아이디(FK) 유저가보는뉴스아이디(FK)
# 유저가 뉴스 스크랩한 뉴스
# 유저가 스크랩한 뉴스로 필터
def get_news_scrap(db: Session, user_id: str, skip: int = 0, limit: int = 100):
    return db.query(models.NewsScrap).filter(models.User.user_id == user_id).offset(skip).limit(limit).all()


# 모든 유저가 스크랩한 뉴스에서 뉴스별로 필터
# (해당 기능 필요 없음)


# 모든 유저가 스크랩한 모든 뉴스
# (해당 기능 필요 없음)


# 유저가 뉴스 스크랩 생성
def create_news_scrap(db: Session, news_scrap: schemas.NewsScrapCreate, user_id: str):
    db_news_scrap = models.NewsScrap(**news_scrap.dict(), user_id=user_id)
    db.add(db_news_scrap)
    db.commit()
    db.refresh(db_news_scrap)
    return db_news_scrap


# 유저가 뉴스 스크랩 제거
def delete_news_scrap(db: Session, news: schemas.NewsScrapDelete, user_id: str, news_scrap_id: int):
    db_news_scrap = models.NewsScrap(user_id=user_id, news_scrap_id=news_scrap_id)
    db.delete(db_news_scrap)
    db.commit()
    db.refresh(db_news_scrap)
    return db_news_scrap


# 준수님 파트
# UserInput : 유저아이디(FK) 유저가보는뉴스아이디(FK) 유저질문문장
# 유저가 입력한 question 관리


# DataFlow 상에서 삭제됨 -> 참고
# AINewsScrap : 유저아이디(FK) AI가보는뉴스아이디(FK)
# AI가 스크랩한 뉴스


# 준수님 파트
# AIInput : 유저아이디(FK) AI가보는뉴스아이디(FK) AI답변문장
# AI가 내놓은 답변