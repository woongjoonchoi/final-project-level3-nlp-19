import streamlit as st

import os
import yaml


from predict import load_model , get_prediction

from confirm_button_hack import cache_on_button_press



st.set_page_config(layout = "wide")

st.write("password")

root_password = 'password'


def main():
    st.title("Question Answering Demo")


    text_input = st.text_input("'궁금해'를 입력해주세요")    

    st.write("잠시만 기다려주세요 🤗")
    model = load_model()


    

    context_input = st.text_input("context를 입력을 해주세요")
    answer_input = st.text_input("answer를 입력을 해주세요")

    print(type(context_input))
    if len(context_input) >0  and len(answer_input) >0 :


        st.write("Classifiying  don distraction me")

        answer = get_prediction(model, context_input , answer_input)


        st.write(f'answer is {answer}')


@cache_on_button_press('Authenticate')
def authenticate(password) ->bool :
    print(type(password))
    return password == root_password

password  = st.text_input('password' , type="password")


if authenticate(password) :
    st.success('you are authenticated')
    main()

else:
    st.error('The password is invalid')


# multiple widget
# main()