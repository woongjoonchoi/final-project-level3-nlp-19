import streamlit as st
from upload_proto_prediction import load_model, get_prediction

import pandas as pd
from io import StringIO
import json

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

st.title("News Article Answer Model")


def main():

    # 모델 불러오기
    model, tokenizer = load_model()

    # 업로드로 json 데이터 받기
    uploaded_file = st.file_uploader("Choose a json file")
    # 기사 날짜의 범위 구하기

    if uploaded_file is not None:
        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # To read file as string:
        string_data = stringio.read()
        st.write(string_data)
        questions = json.loads(string_data)
        st.write(questions['data'][0]["date"])
        st.write(questions['data'][len(questions['data'])-1]["date"])
        date_range = [questions['data'][0]["date"], questions['data'][len(questions['data'])-1]["date"]]

        

    # # Elastic DB에 기사 저장하기


    # # Sqllite에서 user_id에 대한 question데이터 가져오기
    # questions = [1, "ask quenstion"]


        prediction, df = get_prediction(model, tokenizer, questions, date_range)
        
        # 화면창에 띄우기
        st.write(f'label is {prediction}')
        st.write(f'label is {df["context_list"]}')
        for list_idx, context_list in enumerate(df["context_list"]):
            for idx, context in enumerate(context_list):
                if prediction[list_idx]['prediction_text'] in context:
                    st.write(f'user_id is {questions["data"][list_idx]["id"]}')
                    st.write(f'news_id is {df["context_id"][list_idx][idx]}')
                    st.write(f'answer is {prediction[list_idx]["prediction_text"]}')
                    st.write(f'context{idx} is {context}')


    # prediction에 대응하는 user_id와 news_article_id 구하기  df["context_list"]

    # sqllite AIinput에 user_id, news_article_id, prediction 값을 저장하기


    

main()