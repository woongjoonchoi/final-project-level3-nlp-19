from typing import Optional, List
from fastapi import FastAPI, APIRouter, File, UploadFile, Depends, Request
from fastapi.templating import Jinja2Templates
import uvicorn
from starlette.responses import HTMLResponse

from schema.schemas import UserNewsBase, NewsScrap, NewsScrapCreate
from sqlalchemy.orm import Session
from .home import get_db
from services.batchserving import Batchserving

from pydantic import BaseModel

from io import StringIO
import json

router = APIRouter(prefix="/upload", tags=["Upload"])
templates = Jinja2Templates(directory='templates')


# 업로드 페이지로 이동
@router.get("/")
def get_upload_page(request: Request):
    return templates.TemplateResponse('upload.html', context={'request':request})



# 파일 업로드하기
@router.post("/uploadfiles/")
def files_upload(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):

    prediction, df = Batchserving.serving(files, db)
    print(prediction)
    # 업로드로 json 데이터 받기
    # 기사 날짜의 범위 구하기

    if files is not None:
        # To convert to a string based IO:
        # stringio = StringIO(files[0].file)
        # To read file as string:
        string_data = files[0].file.read()
        # print(string_data)
        questions = json.loads(string_data)
        date_range = [questions['data'][0]["date"], questions['data'][len(questions['data'])-1]["date"]]

    return {"questions": questions, "date_range" : date_range}



if __name__ == '__main__':
    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app="AIPaperboy:app", host="0.0.0.0", port=8000, reload=True)
