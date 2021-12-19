from fastapi import FastAPI, APIRouter
from fastapi.templating import Jinja2Templates
import uvicorn

from ..services.scrappedboard import Scrappedboard
from ..services.manageuserinput import Manageuserinput
from ..services.managenewsscrap import Managenewsscrap

router = APIRouter(prefix="/scrap", tags=["Scrap"])
templates = Jinja2Templates(directory='serving/templates')


# 사용자 scrap 페이지로 이동(웅준)
@router.get("/")
def get_scrap_page():

    # 로그인이 되어있으면 Scrappedboard Service 객체로 사용자가 스크랩한 뉴스기사 목록 불러오기

    # 로그인이 안되어있으면 로그인 화면으로 이동(로그인 기능이 구현되어 있다면)

    pass



if __name__ == '__main__':
    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app="AIPaperboy:app", host="0.0.0.0", port=8000, reload=True)