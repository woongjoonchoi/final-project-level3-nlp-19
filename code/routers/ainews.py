from fastapi import FastAPI, APIRouter
from fastapi.templating import Jinja2Templates
import uvicorn

router = APIRouter(prefix="/ainews", tags=["AINews"])
templates = Jinja2Templates(directory='./templates')


# AI 스크랩 뉴스기사 화면이동(웅준)
@router.get("/")
def get_ainews_page():

    # 로그인이 되어있으면 해당 화면으로 이동

    # 로그인이 안되어있으면 로그인 화면으로 이동(로그인 기능이 구현되어 있다면)
    
    pass


# 사용자가 AI가 스크랩해준 뉴스 기사에 입력한 정보, 스크랩 정보를 DB에 저장하기(준수, 별이)
@router.post("/")
def post_news_input():

    # Userinput Service 객체로 사용자 입력 정보를 DB에 저장하기(준수)

    # Newsscrap Service 객체로 사용자 스크랩 정보를 DB에 저장하기(별이)

    pass


# 사용자가 AI가 스크랩해준 뉴스 기사에 입력한 정보, 스크랩 정보를 DB에서 삭제하기(준수, 별이)
@router.delete("/")
def delete_news_input():
    # Userinput Service 객체로 사용자 입력 정보를 DB에서 삭제하기(준수)

    # 해당 기사에서 Userinput 정보가 없으면 Newsscrap Service 객체로 사용자 스크랩 정보를 DB에서 삭제하기(별이)

    pass


# AI가 스크랩해준 뉴스 기사를 사용자가 스크랩하기(별이)
@router.post("/")
def insert_scrap_input():
    # Newsscrap Service 객체로 사용자 스크랩 정보를 DB에 저장하기   
    pass


# 사용자가 스크랩한 AI가 스크랩해준 뉴스기사를 스크랩 취소하기(별이)
@router.delete("/")
def delete_scrap_input():
    # Newsscrap Service 객체로 사용자 스크랩 정보를 DB에서 삭제하기
    pass


if __name__ == '__main__':
    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app="AIPaperboy:app", host="0.0.0.0", port=8000, reload=True)