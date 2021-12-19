from sqlalchemy.orm import Session
from ..schema import models, schemas

# 사용자가 스크랩 정보를 관리한다.
class Managenewsscrap():

    # 사용자가 스크랩한 정보를 DB에 저장하기
    def create_user_scrap(db: Session, scrap: schemas.ScrapCreate, user_id: int):
        db_scrap = models.Scrap(**scrap.dict(), owner_user_id=user_id) 
        db.add(db_scrap)
        db.commit()
        db.refresh(db_scrap)
        return db_scrap

    # 사용자가 스크랩한 정보를 DB에서 삭제하기
    def delete_news_input():
        pass

    pass