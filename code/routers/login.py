from fastapi import FastAPI, APIRouter, Request, File, Form
from fastapi.templating import Jinja2Templates
import uvicorn

from services.checklogin import Checklogin

router = APIRouter(prefix="/login", tags=["login"])
templates = Jinja2Templates(directory='./templates')



# 로그인 페이지로 이동
@router.get("/")
def get_login_page():
    pass



# 로그인 하기
@router.post("/")
def login(username:str = Form(...), password:str = Form(...)):
    
    # Checklogin Service 객체로 로그인 정보 확인하기


    # 로그인에 성공하면 홈페이지로 이동

    # 로그인에 실패하면 실패이유 반환하기

    pass


if __name__ == '__main__':
    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app="AIPaperboy:app", host="0.0.0.0", port=8000, reload=True)