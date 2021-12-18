from typing import Optional
from fastapi import FastAPI, APIRouter
from fastapi.templating import Jinja2Templates
import uvicorn

from pydantic import BaseModel

router = APIRouter(prefix="/news", tags=["News"])
templates = Jinja2Templates(directory='./templates')


# 뉴스 기사 불러오기(웅준)
@router.get("/{news_id}")
def get_news_page():
    # Board Service 객체로 news 기사 목록 데이터 가져오기
    pass


# 사용자가 뉴스 기사에 입력한 정보, 스크랩 정보를 DB에 저장하기(준수, 별이)
@router.post("/")
def post_news_input():

    # Userinput Service 객체로 사용자 입력 정보를 DB에 저장하기(준수)

    # Newsscrap Service 객체로 사용자 스크랩 정보를 DB에 저장하기(별이)

    pass


# 사용자가 뉴스 기사에 입력한 정보, 스크랩 정보를 DB에서 삭제하기(준수, 별이)
@router.delete("/")
def delete_news_input():
    # Userinput Service 객체로 사용자 입력 정보를 DB에서 삭제하기(준수)

    # 해당 기사에서 Userinput 정보가 없으면 Newsscrap Service 객체로 사용자 스크랩 정보를 DB에서 삭제하기(별이)

    pass


# 뉴스 기사 스크랩하기(별이)
@router.post("/")
def insert_scrap_input():
    # Newsscrap Service 객체로 사용자 스크랩 정보를 DB에 저장하기   
    pass


# 스크랩한 뉴스기사 취소하기(별이)
@router.delete("/")
def delete_scrap_input():
    # Newsscrap Service 객체로 사용자 스크랩 정보를 DB에서 삭제하기
    pass


if __name__ == '__main__':
    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app="AIPaperboy:app", host="0.0.0.0", port=8000, reload=True)