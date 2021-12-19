from sqlalchemy.orm import Session
from schema.models import NewsScrap
from schema.schemas import NewsScrapCreate, Question

# 사용자가 스크랩 정보를 관리한다.
class Managenewsscrap():

    # 유저가 뉴스 스크랩 생성
    def create_scrap_news(db: Session, news: NewsScrapCreate, user_id: str, user_news_id: int):
        db_scrap_news = NewsScrap(user_id=user_id, user_news_id=user_news_id)
        db.add(db_scrap_news)
        db.commit()
        db.refresh(db_scrap_news)
        return db_scrap_news

    # 유저가 뉴스 스크랩 제거
    def delete_scrap_news(db: Session, news: NewsScrapCreate, user_id: str, user_news_id: int):
        db_scrap_news = NewsScrap(user_id=user_id, user_news_id=user_news_id)
        db.delete(db_scrap_news)
        db.commit()
        db.refresh(db_scrap_news)
        return db_scrap_news


    # NewsScrap : 유저아이디(FK) 유저가보는뉴스아이디(FK)
    # 유저가 뉴스 스크랩한 뉴스
    # 유저가 스크랩한 뉴스로 필터
    def get_news_scrap(db: Session, user_id: str, skip: int = 0, limit: int = 10):
        return db.query(NewsScrap).filter(NewsScrap.user_id == user_id).first()