from fastapi import FastAPI, APIRouter
from fastapi.templating import Jinja2Templates
import uvicorn
from schema.database import SessionLocal

from services.homeboard import Homeborad



router = APIRouter(prefix="/home", tags=["Home"])
templates = Jinja2Templates(directory='./templates')

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 뉴스 홈페이지 화면이동(웅준)
@router.get("/")
def get_home_page():
    # Homeboard Service 객체로 뉴스 목록 가져오기
    return Homeborad.search()
    


if __name__ == '__main__':
    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app="AIPaperboy:app", host="0.0.0.0", port=8000, reload=True)