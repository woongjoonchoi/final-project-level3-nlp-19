from sqlalchemy.orm import Session
from . import models, schemas


# ! 잘 이해 안 됨
# ! id str형으로 바꿔야 함
# ! 뉴스 기사 수로 바꿔야 함


# 회원 : 아이디(PK) 비밀번호 이름 알람설정
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


def create_user(db: Session, user: schemas.UserCreate):
    fake_hashed_password = user.password + "notreallyhashed"
    db_user = models.User(id=user.id, hashed_password=fake_hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


# 관리자 : 아이디(PK) 비밀번호
def get_admin(db: Session, admin_id: int):
    return db.query(models.Admin).filter(models.Admin.id == admin_id).first()

    
def get_admins(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Admin).offset(skip).limit(limit).all()


def create_admin(db: Session, admin: schemas.AdminCreate):
    fake_hashed_password = user.password + "notreallyhashed"
    db_admin = models.Admin(id=user.id, hashed_password=fake_hashed_password)
    db.add(db_admin)
    db.commit()
    db.refresh(db_admin)
    return db_admin


# 뉴스 : 아이디(PK) 제목 내용
def get_news(db: Session, news_id: int):
    return db.query(models.News).filter(models.News.id == news_id).first()


def get_newss(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.News).offset(skip).limit(limit).all()


def create_user_scrap_news(db: Session, news: schemas.NewsCreate, scrap_id: int):
    db_news = models.News(**news.dict(), owner_scrap_id=scrap_id)
    db.add(db_news)
    db.commit()
    db.refresh(db_news)
    return db_news


# 스크랩 : 아이디(PK) 작성자(FK) 질문뉴스_아이디(FK) 질문문장 답변뉴스_아이디(FK) 답변문장
def get_scrap(db: Session, scrap_id: int):
    return db.query(models.Scrap).filter(models.Scrap.id == scrap_id).first()


def get_scraps(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Scrap).offset(skip).limit(limit).all()


def create_user_scrap(db: Session, scrap: schemas.ScrapCreate, user_id: int):
    db_scrap = models.Scrap(**scrap.dict(), owner_user_id=user_id) 
    db.add(db_scrap)
    db.commit()
    db.refresh(db_scrap)
    return db_scrap

