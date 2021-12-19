from fastapi import FastAPI, APIRouter
from fastapi.templating import Jinja2Templates
import uvicorn

from ..services.aiscrappednewscontent import Aiscrappednewscontent
from ..services.manageuserinput import Manageuserinput
from ..services.managenewsscrap import Managenewsscrap

router = APIRouter(prefix="/ainews", tags=["AINews"])
templates = Jinja2Templates(directory='serving/templates')


# AI 스크랩 뉴스기사 화면이동(창한)
@router.get("/")
def get_ainews_page():
    # Aiscrappednewscontent Service 객체로 AI가 스크랩한 뉴스 기사 본문 가져오기
    pass

"""
@app.get("user/{user_id}/news_scrap/", description="유저가 스크랩하거나 해재할 뉴스 페이지 html")
def get_news_scraps_form(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse('article_form.html', context={'request': request})
"""


# 사용자가 AI가 스크랩해준 뉴스 기사에 입력한 정보, 스크랩 정보를 DB에 저장하기(준수, 별이)
@router.post("/")
def post_news_input(news_scrap: schemas.NewsScrapCreate, db: Session = Depends(get_db)):

    # Manageuserinput Service 객체로 사용자 입력 정보를 DB에 저장하기(준수)

    # Managenewsscrap Service 객체로 사용자 스크랩 정보를 DB에 저장하기(별이)
    return Managenewsscrap.create_news_scrap(db=db, news_scrap=news_scrap)


# 사용자가 AI가 스크랩해준 뉴스 기사에 입력한 정보, 스크랩 정보를 DB에서 삭제하기(준수, 별이)
@router.delete("/")
def delete_news_input(news_scrap: schemas.NewsScrapDelete, db: Session = Depends(get_db)):

    # Manageuserinput Service 객체로 사용자 입력 정보를 DB에서 삭제하기(준수)

    # 해당 기사에서 Userinput 정보가 없으면 Managenewsscrap Service 객체로 사용자 스크랩 정보를 DB에서 삭제하기(별이)
    pass




# AI가 스크랩해준 뉴스 기사를 사용자가 스크랩하기(별이)
@router.post("/", description="유저아이디, 스크랩할 뉴스아이디 가져와서 db에 저장")
def post_scrap_input(news_scrap: schemas.NewsScrapCreate, db: Session = Depends(get_db)):
    # Managenewsscrap Service 객체로 사용자 스크랩 정보를 DB에 저장하기
    return Managenewsscrap.create_news_scrap(db=db, news_scrap=news_scrap)


# 사용자가 스크랩한 AI가 스크랩해준 뉴스기사를 스크랩 취소하기(별이)
@router.delete("/", description="유저아이디, 스크랩할 뉴스아이디 가져와서 db에서 제거")
def delete_scrap_input(news_scrap: schemas.NewsScrapDelete, db: Session = Depends(get_db)):
    # Managenewsscrap Service 객체로 사용자 스크랩 정보를 DB에서 삭제하기
    return Managenewsscrap.delete_news_scrap(db=db, news_scrap=news_scrap)







if __name__ == '__main__':
    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app="AIPaperboy:app", host="0.0.0.0", port=8000, reload=True)