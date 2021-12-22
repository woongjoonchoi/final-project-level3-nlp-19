from typing import Optional
from fastapi import FastAPI, APIRouter, Depends, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
import uvicorn

from ..services.newscontent import Newscontent
from ..services.manageuserinput import Manageuserinput
from ..services.managenewsscrap import Managenewsscrap

from pydantic import BaseModel

from .home import get_db
from ..schema import schemas
from sqlalchemy.orm import Session

router = APIRouter(prefix="/news", tags=["News"])
templates = Jinja2Templates(directory='serving/templates')


# 뉴스 기사 불러오기(창한)
@router.get("/{news_id}",  description="사용자가 호출한 뉴스 본문을 볼 수 있도록 합니다.")
def get_news_page(request: Request, news_id: str, user_id: str, db: Session = Depends(get_db)):
    # Newscontent Service 객체로 news 기사 본문 가져오기
<<<<<<< HEAD
    title, article  = Newscontent.get_news(db=db, news_id=news_id, user_id = user_id)
    article = article.replace('<br />', '\n')
    article = article.replace('<!------------ PHOTO_POS_0 ------------>', '')
    print(title, article, sep='\n')
    
    return templates.TemplateResponse('news_article.html', context={'request': request, 'news_id': news_id, 'user_id': user_id, 'title': title, 'article': article})
    
=======
    title, article = Newscontent.get_news(db=db, news_id=news_id, user_id=user_id)
    return templates.TemplateResponse('article_form.html', context={'request': request, 'title': title, 'article': article, 'news_id': news_id, 'user_id': user_id})
>>>>>>> 6b866e0104b640978b75a0a90cd474b77de70096


# # 사용자가 뉴스 기사에 입력한 정보, 스크랩 정보를 DB에 저장하기(준수, 별이)
# @router.post("/")
# def post_news_input(user_info: schemas.UserInputBase, news_scrap: schemas.NewsScrapCreate, db: Session = Depends(get_db), input: str = Form(...)):

<<<<<<< HEAD
# 사용자가 뉴스 기사에 입력한 정보, 스크랩 정보를 DB에 저장하기(준수, 별이)
@router.post("/question")
def post_news_input(user_info: schemas.UserInputBase, news_scrap: schemas.NewsScrapCreate, db: Session = Depends(get_db), input: str = Form(...)):
    print(input, user_info)
    # Manageuserinput Service 객체로 사용자 입력 정보를 DB에 저장하기(준수)
    Manageuserinput.insert_news_input(db=db, user_info=user_info, input=input)
    print("No Problem")
    # Managenewsscrap Service 객체로 사용자 스크랩 정보를 DB에 저장하기(별이)
    return Managenewsscrap.create_news_scrap(db=db, news_scrap=news_scrap)
=======
#     # Manageuserinput Service 객체로 사용자 입력 정보를 DB에 저장하기(준수)
#     Manageuserinput.insert_news_input(db=db, user_info=user_info, input=input)
#     # Managenewsscrap Service 객체로 사용자 스크랩 정보를 DB에 저장하기(별이)
#     Managenewsscrap.create_news_scrap(db=db, news_scrap=news_scrap)
#     pass

>>>>>>> 6b866e0104b640978b75a0a90cd474b77de70096

# # 사용자가 뉴스 기사에 입력한 정보, 스크랩 정보를 DB에서 삭제하기(준수, 별이)
# @router.delete("/")
# def delete_news_input(user_id: str, user_news_id: str, db: Session = Depends(get_db)):
    
#     # Manageuserinput Service 객체로 사용자 입력 정보를 DB에서 삭제하기(준수)

#     # 해당 기사에서 Userinput 정보가 없으면 Managenewsscrap Service 객체로 사용자 스크랩 정보를 DB에서 삭제하기(별이)
#     pass


# 뉴스 기사 스크랩하기(별이)
@router.get("/{news_id}/create", description="유저아이디, 스크랩할 뉴스아이디 가져와서 db에 저장")
def post_scrap_input(request: Request, news_id: str, user_id: str, db: Session = Depends(get_db)):
    # Managenewsscrap Service 객체로 사용자 스크랩 정보를 DB에 저장하기

    Managenewsscrap.create_news_scrap(db=db, user_id=user_id, news_id=news_id)
    print(111111111111111111111111111111111111111111111111111111)
    url = f'/login'
    return RedirectResponse(url=url, status_code=302)


# 스크랩한 뉴스기사 취소하기(별이)
@router.delete("/{news_id}/delete", description="유저아이디, 스크랩할 뉴스아이디 가져와서 db에서 제거")
def delete_scrap_input(request: Request, user_id: str, news_id: str, db: Session = Depends(get_db)):
    # Managenewsscrap Service 객체로 사용자 스크랩 정보를 DB에서 삭제하기
    Managenewsscrap.delete_news_scrap(db=db, user_id=user_id, user_news_id=news_id)
    return templates.TemplateResponse('alrtice_form.html', context={'request': request})


if __name__ == '__main__':
    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app="AIPaperboy:app", host="0.0.0.0", port=8000, reload=True)