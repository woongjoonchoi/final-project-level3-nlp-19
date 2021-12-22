from sqlalchemy.orm import Session
from fastapi import Form

from ..schema import models, schemas

# 사용자가 입력한 정보를 관리한다.
class Manageuserinput():

    # 사용자가 입력한 정보를 DB에 저장하기
    # def insert_news_input(db: Session, user_info=schemas.UserInputBase, input: str = Form(...)):
    def insert_news_input(db: Session, user_id: str, news_id: str, user_input: str = Form(...)):
        # db_user_input = models.UserInput(**user_info.dict(), user_input=input)
        db_user_input = models.UserInput(user_id=user_id, user_news_id=news_id, user_input=user_input)
        
        db.add(db_user_input)
        db.commit()
        db.refresh(db_user_input)

        return db_user_input


    # 사용자가 입력한 정보를 DB에서 삭제하기
    def delete_news_input(db: Session, user_id: str, news_id: str):
        db_user_input = db.query(models.UserInput).filter(models.UserInput.user_id == user_id and models.UserInput.user_news_id == news_id).delete()
        db.commit()
        return db_user_input