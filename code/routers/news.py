from typing import Optional
from fastapi import FastAPI, APIRouter, Depends, Form
from fastapi.templating import Jinja2Templates
import uvicorn

from services.newscontent import Newscontent
from services.manageuserinput import Manageuserinput
from services.managenewsscrap import Managenewsscrap

from schema.schemas import UserNewsBase, NewsScrap, NewsScrapCreate
from sqlalchemy.orm import Session
from routers.home import get_db

router = APIRouter(prefix="/news", tags=["News"])
templates = Jinja2Templates(directory='./templates')



# 뉴스 기사 불러오기(창한)
@router.get("/{news_id}",  description="사용자가 호출한 뉴스 본문을 볼 수 있도록 합니다.")
def get_news_page(news_id: str, user_id: str, db: Session = Depends(get_db)):
    # Newscontent Service 객체로 news 기사 본문 가져오기
    return Newscontent.get_news(db=db, news_id=news_id, user_id = user_id)


# 사용자가 뉴스 기사에 입력한 정보, 스크랩 정보를 DB에 저장하기(준수, 별이)
@router.post("/")
def question_to_db(text: str = Form(...), user_id: str = Form(...), db: Session = Depends(get_db)):

    # Manageuserinput Service 객체로 사용자 입력 정보를 DB에 저장하기(준수)
    Manageuserinput.insert_news_input(db=db, text=text, user_id=user_id)

    # Managenewsscrap Service 객체로 사용자 스크랩 정보를 DB에 저장하기(별이)

    pass


# 사용자가 뉴스 기사에 입력한 정보, 스크랩 정보를 DB에서 삭제하기(준수, 별이)
@router.delete("/{news_id}")
def delete_news_input():
    
    # Manageuserinput Service 객체로 사용자 입력 정보를 DB에서 삭제하기(준수)

    # 해당 기사에서 Userinput 정보가 없으면 Managenewsscrap Service 객체로 사용자 스크랩 정보를 DB에서 삭제하기(별이)

    pass


# 뉴스 기사 스크랩하기(별이)
@router.post("/{user_id}", response_model=NewsScrap, description="유저가 무엇을 스크랩할지 입력할 수 있음")
def create_scrap_for_user(user_id: str, news_scrap: NewsScrapCreate, db: Session = Depends(get_db)):
    # Managenewsscrap Service 객체로 사용자 스크랩 정보를 DB에 저장하기
    return Managenewsscrap.create_scrap_news(db=db, news_scrap=news_scrap, user_id=user_id)



# 스크랩한 뉴스기사 취소하기(별이)
@router.delete("/")
def delete_scrap_input():
    # Managenewsscrap Service 객체로 사용자 스크랩 정보를 DB에서 삭제하기
    pass


if __name__ == '__main__':
    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app="AIPaperboy:app", host="0.0.0.0", port=8000, reload=True)