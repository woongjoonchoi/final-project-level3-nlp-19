from fastapi import FastAPI, APIRouter,Request,Form
from fastapi.templating import Jinja2Templates
import uvicorn

from ..services.homeboard import Homeboard


router = APIRouter(prefix="/home", tags=["Home"])
templates = Jinja2Templates(directory='serving/templates')


home_board= Homeboard()

# 뉴스 홈페이지 화면이동(웅준)
@router.get("/")
def get_home_page(request : Request):
    # Homeboard Service 객체로 뉴스 목록 가져오기
    res = home_board.get_news_title()
    return templates.TemplateResponse('home.html', context={'request': request , 'res_news' : res['hits']['hits']})


if __name__ == '__main__':
    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app="AIPaperboy:app", host="0.0.0.0", port=8000, reload=True)