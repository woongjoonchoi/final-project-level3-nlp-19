# from sqlalchemy.orm import Session
# from . import models, schemas


# # ! 잘 이해 안 됨
# # ! id str형으로 바꿔야 함
# # ! 뉴스 기사 수로 바꿔야 함


# # 회원 : 아이디(PK) 비밀번호 이름 알람설정
# def get_user(db: Session, user_id: int):
#     return db.query(models.User).filter(models.User.id == user_id).first()


# def get_users(db: Session, skip: int = 0, limit: int = 100):
#     return db.query(models.User).offset(skip).limit(limit).all()


# def create_user(db: Session, user: schemas.UserCreate):
#     fake_hashed_password = user.password + "notreallyhashed"
#     db_user = models.User(id=user.id, hashed_password=fake_hashed_password)
#     db.add(db_user)
#     db.commit()
#     db.refresh(db_user)
#     return db_user


# # 관리자 : 아이디(PK) 비밀번호
# def get_admin(db: Session, admin_id: int):
#     return db.query(models.Admin).filter(models.Admin.id == admin_id).first()

    
# def get_admins(db: Session, skip: int = 0, limit: int = 100):
#     return db.query(models.Admin).offset(skip).limit(limit).all()


# def create_admin(db: Session, admin: schemas.AdminCreate):
#     fake_hashed_password = user.password + "notreallyhashed"
#     db_admin = models.Admin(id=user.id, hashed_password=fake_hashed_password)
#     db.add(db_admin)
#     db.commit()
#     db.refresh(db_admin)
#     return db_admin


# # 뉴스 : 아이디(PK) 제목 내용
# def get_news(db: Session, news_id: int):
#     return db.query(models.News).filter(models.News.id == news_id).first()


# def get_newss(db: Session, skip: int = 0, limit: int = 100):
#     return db.query(models.News).offset(skip).limit(limit).all()


# def create_user_scrap_news(db: Session, news: schemas.NewsCreate, scrap_id: int):
#     db_news = models.News(**news.dict(), owner_scrap_id=scrap_id)
#     db.add(db_news)
#     db.commit()
#     db.refresh(db_news)
#     return db_news


# # 스크랩 : 아이디(PK) 작성자(FK) 질문뉴스_아이디(FK) 질문문장 답변뉴스_아이디(FK) 답변문장
# def get_scrap(db: Session, scrap_id: int):
#     return db.query(models.Scrap).filter(models.Scrap.id == scrap_id).first()


# def get_scraps(db: Session, skip: int = 0, limit: int = 100):
#     return db.query(models.Scrap).offset(skip).limit(limit).all()


# def create_user_scrap(db: Session, scrap: schemas.ScrapCreate, user_id: int):
#     db_scrap = models.Scrap(**scrap.dict(), owner_user_id=user_id) 
#     db.add(db_scrap)
#     db.commit()
#     db.refresh(db_scrap)
#     return db_scrap


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
def get_news_scrap(db: Session, user_id: str, skip: int = 0, limit: int = 10):
    return db.query(models.NewsScrap).filter(models.NewsScrap.user_id == user_id).first()


# 모든 유저가 스크랩한 뉴스에서 뉴스별로 필터
# (해당 기능 필요 없음)


# 모든 유저가 스크랩한 모든 뉴스
# (해당 기능 필요 없음)

    # return db.query(models.News).offset(skip).limit(limit).all()


# 유저가 뉴스 스크랩 생성
def create_scrap_news(db: Session, news: schemas.NewsScrapCreate, user_id: str, user_news_id: int):
    db_scrap_news = models.NewsScrap(user_id=user_id, user_news_id=user_news_id)
    db.add(db_scrap_news)
    db.commit()
    db.refresh(db_scrap_news)
    return db_scrap_news


# 유저가 뉴스 스크랩 제거
def delete_scrap_news(db: Session, news: schemas.NewsScrapCreate, user_id: str, user_news_id: int):
    db_scrap_news = models.NeNewsScrapws(user_id=user_id, user_news_id=user_news_id)
    db.delete(db_scrap_news)
    db.commit()
    db.refresh(db_scrap_news)
    return db_scrap_news


# 준수님 파트
# UserInput : 유저아이디(FK) 유저가보는뉴스아이디(FK) 유저질문문장
# 유저가 입력한 question 관리


# AINewsScrap : 유저아이디(FK) AI가보는뉴스아이디(FK)
# AI가 스크랩한 뉴스


# 준수님 파트
# AIInput : 유저아이디(FK) AI가보는뉴스아이디(FK) AI답변문장
# AI가 내놓은 답변
