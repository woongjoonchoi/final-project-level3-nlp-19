from typing import Optional, List
from sqlalchemy.orm.session import Session
from sqlalchemy.orm import Session
# from ...code import predict
from fastapi import UploadFile
import json
from upload_proto_prediction import load_model, get_prediction
from schema import models
from elasticsearch import Elasticsearch, helpers

es = Elasticsearch()



# Batch serving 하기 : 뉴스 기사를 업로드하기, 사용자 질문 정보 불러오기, 모델로 예측하기, 결과값 저장하기
class Batchserving():

    def serving(files: List[UploadFile], db: Session):
        if files is not None:
            string_data = files[0].file.read()
            latest_news = json.loads(string_data)

            # 최신 뉴스 elastic DB에 넣기            
            try:
                response = helpers.bulk(es, latest_news)
            except Exception as e:
                print("\nERROR:", e)

            # date_range 구하기
            date_range = [latest_news['data'][0]["date"], latest_news['data'][len(latest_news['data'])-1]["date"]]


            # User Input 불러오기 -> qeustions 만들기
            db_user_input = db.query(models.UserInput).all()

            questions = {}
            questions["data"] = []
            for row in db_user_input:
                sample = {
                    "id" : row.user_id,
                    "question" : row.user_input
                }
                questions["data"].append(sample)


            # 모델 불러오기
            model, tokenizer = load_model()

            prediction, df = get_prediction(model, tokenizer, questions, date_range)


            # prediction 결과를 AIInput table에 넣기, 업로드 창에 띄울 output 생성
            output = {}
            output["prediction"] = prediction
            output["context_list"] = df["context_list"]
            output["answers"] = []
            for list_idx, context_list in enumerate(df["context_list"]):
                for idx, context in enumerate(context_list):
                    if prediction[list_idx]['prediction_text'] in context:
                        output["answers"].append({"user_id" : questions["data"][list_idx]["id"],
                        "news_id" : df["context_id"][list_idx][idx],
                        "answer" : prediction[list_idx]["prediction_text"],
                        f'context{idx}' : context
                        })
                        db_user_input = models.AIInput(user_id=questions["data"][list_idx]["id"], ai_news_id=df["context_id"][list_idx][idx], ai_input=prediction[list_idx]["prediction_text"])             
                        db.add(db_user_input)
                        db.commit()
                        db.refresh(db_user_input)

        return output