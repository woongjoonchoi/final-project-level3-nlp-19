from sqlalchemy.orm import Session
from ..schema import models, schemas

from random import randint

# 사용자가 스크랩 정보를 관리한다.
class Managenewsscrap():

    # 유저가 뉴스 스크랩 생성

    def create_news_scrap(db: Session, user_id: str, news_id: str):
        db_news_scrap = models.NewsScrap(user_id=user_id, user_news_id=news_id, news_scrap_id=randint(1, 100000000))
        db.add(db_news_scrap)
        db.commit()
        db.refresh(db_news_scrap)
        return db_news_scrap


    # 유저가 뉴스 스크랩 제거
    def delete_news_scrap(db: Session, user_id: str, user_news_id: str):
        # db에서 news_scrap_id 꺼내오기
        db_news_scrap = models.NewsScrap(user_id=user_id, user_news_id=user_news_id, news_scrap_id=randint(1, 100000000))
        db.delete(db_news_scrap)
        db.commit()
        db.refresh(db_news_scrap)
        return db_news_scrap

    pass
