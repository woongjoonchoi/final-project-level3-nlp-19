from typing import Optional, List
from sqlalchemy.orm.session import Session
from sqlalchemy.orm import Session
# from ...code import predict
from fastapi import UploadFile
import json
from ..code.upload_proto_prediction import load_model, get_prediction


# Batch serving 하기 : 뉴스 기사를 업로드하기, 사용자 질문 정보 불러오기, 모델로 예측하기, 결과값 저장하기
class Batchserving():

    def serving(files: List[UploadFile], db: Session):
        if files is not None:
            # To convert to a string based IO:
            # stringio = StringIO(files[0].file)
            # To read file as string:
            string_data = files[0].file.read()
            # print(string_data)
            questions = json.loads(string_data)
            date_range = [questions['data'][0]["date"], questions['data'][len(questions['data'])-1]["date"]]

            # 모델 불러오기
            model, tokenizer = load_model()

            prediction, df = get_prediction(model, tokenizer, questions, date_range)

        # Table_name = "user_input"
        # questions = db.query(Table_name).all()

        # model, tokenizer = predict.load_model()
        # prediction = predict.get_prediction(model, tokenizer, questions)

            # # 화면창에 띄우기
            # output = {}
            # output["prediction"] = prediction
            # output["context_list"] = df["context_list"]
            # for list_idx, context_list in enumerate(df["context_list"]):
            #     for idx, context in enumerate(context_list):
            #         if prediction[list_idx]['prediction_text'] in context:
            #             output["answers"] = {"user_id" : questions["data"][list_idx]["id"]},
            #             "news_id" : df["context_id"][list_idx][idx],
            #             "answer"}
                        
            #             st.write(f'news_id is {}')
            #             st.write(f' is {prediction[list_idx]["prediction_text"]}')
            #             st.write(f'context{idx} is {context}')


        return prediction ,df