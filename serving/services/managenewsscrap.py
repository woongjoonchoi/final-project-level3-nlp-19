from sqlalchemy.orm import Session
from ..schema import models, schemas

# 사용자가 스크랩 정보를 관리한다.
class Managenewsscrap():

    # 사용자가 스크랩한 정보를 DB에 저장하기
    def create_news_scrap(db: Session, news_scrap: schemas.NewsScrapCreate, user_id: str):
        db_news_scrap = models.NewsScrap(**news_scrap.dict(), user_id=user_id)
        db.add(db_news_scrap)
        db.commit()
        db.refresh(db_news_scrap)
        return db_news_scrap

    # 사용자가 스크랩한 정보를 DB에서 삭제하기
    def delete_news_scrap(db: Session, news: schemas.NewsScrapDelete, user_id: str, news_scrap_id: int):
        db_news_scrap = models.NewsScrap(user_id=user_id, news_scrap_id=news_scrap_id)
        db.delete(db_news_scrap)
        db.commit()
        db.refresh(db_news_scrap)
        return db_news_scrap


    pass