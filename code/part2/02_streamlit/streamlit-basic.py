import streamlit as st
import pandas as pd
import numpy as np
import time


st.title("Woongjon Title")
st.header("wjc HEader")
st.subheader("subheader_by_woj")

## unicode 출력됨
 
## b'\xec\x95\x88\xeb\x85\x95'
st.title('안녕'.encode())


## bytes 출력됨
## b'hello'
st.title('hello'.encode())


## Markdown 문법됨 
st.write("# Header Type 1")
st.write("## Header type 2")

if st.button("버튼이 클릭되면") :
    st.write("XXXX")

if st.button("버튼이 클릭되면 ver2") :
    ## 여러개 됨
    st.write("# XXX")
    st.write("## xXX_!")
    #잘못된 문법을 사용하면 빨강화면이 뜸
    st.write("### XXX_$")

checkbox_btn = st.checkbox('체크 버튼')

if checkbox_btn : 
    st.write('체트박스 버튼 클릭릭릭!')


## write: 보여 줄 수 있는것은 어느것이든 보여줌

## 
df = pd.DataFrame({
    'first_column' : [1,2,3,4] ,
    'second_column' : [10,20,30,40]
})

st.markdown("# -==--")

st.write(df)
st.dataframe(df)
st.table(df)
