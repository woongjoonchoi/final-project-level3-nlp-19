import random
from sqlalchemy.orm import Session
from schema.models import Question

# 사용자가 입력한 정보를 관리한다.
class Manageuserinput():

    # 사용자가 입력한 정보를 DB에 저장하기
    def insert_news_input(db: Session, text: str, user_id: str):
        new_question = Question(id=user_id, text=text)

        db.add(new_question)
        db.commit()
        db.refresh(new_question)
        
        return new_question

    # 사용자가 입력한 정보를 DB에서 삭제하기
    def delete_news_input():
        pass
