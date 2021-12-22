from sqlalchemy.orm import Session
from fastapi import Form

from schema import models, schemas
from prototype_predict import load_model, get_prediction

# 사용자가 입력한 정보를 관리한다.
class Manageuserinput():

    # 사용자가 입력한 정보를 DB에 저장하기
        # 다시 작성해야 함, 현재는 news_id가 7자리여서 가능하나 아이디 글자수, 소문자+영어조합만 가능 제한이 있음(int 길이 때문에)
    def insert_news_input(db: Session, user_id: str, news_id: str, user_input: str = Form(...)):
        # str_user_id = str(str2int(user_id))
        # user_input_id = news_id + str_user_id

        db_user_input = models.UserInput(user_id=user_id, user_news_id=news_id, user_input=user_input)
        db.add(db_user_input)
        db.commit()
        db.refresh(db_user_input)

        # db_user_input = models.UserInput(**user_info.dict(), user_input=input)

        # 모델 불러오기
        model, tokenizer = load_model()

        # 예측하기
        prediction, df = get_prediction(model, tokenizer, user_input)


        # 예측결과 AIInput 테이블에 저장하기        
        for idx, context in enumerate(df["context_list"][0]):
            if prediction[0]['prediction_text'] in context:
                db_user_input = models.AIInput(user_id=user_id, ai_news_id=df["context_id"][0][idx], ai_input=prediction[0]["prediction_text"])             
                db.add(db_user_input)
                db.commit()
                db.refresh(db_user_input)

        return db_user_input

    # 사용자가 입력한 정보를 DB에서 삭제하기
    def delete_news_input(db: Session, user_id: str, news_id: str):
        db_user_input = db.query(models.UserInput).filter(models.UserInput.user_id == user_id and models.UserInput.user_news_id == news_id).delete()
        db.commit()
        return db_user_input