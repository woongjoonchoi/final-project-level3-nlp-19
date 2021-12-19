from fastapi import FastAPI, APIRouter, Request, File, Form
from fastapi.templating import Jinja2Templates
import uvicorn

from ..services.managelogin import Checklogin, Signup

router = APIRouter(prefix="/login", tags=["login"])
templates = Jinja2Templates(directory='serving/templates')



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


# 회원가입 페이지로 이동
@router.get("/signup")
def get_signup_page():
    pass


# 회원가입 하기
@router.post("/signup")
def create_user():
    # Signup Service 객체로 입력받은 회원정보를 db에 저장하기
    pass


if __name__ == '__main__':
    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app="AIPaperboy:app", host="0.0.0.0", port=8000, reload=True)