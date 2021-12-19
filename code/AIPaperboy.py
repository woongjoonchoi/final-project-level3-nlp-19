import os

from fastapi import Depends, FastAPI, Form, Request, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
import uvicorn

from routers import (
    news, aiscrap, upload, 
    ainews, login, home, scrap, scrapnews)

from schema.models import Base
from schema.database import SessionLocal, engine



app = FastAPI()
templates = Jinja2Templates(directory='./templates')

# Base.metadata.create_all(engine)


# 뉴스 홈페이지로 연결하기
@app.get("/")
def get_home_page():
    return RedirectResponse("./home")
    
# router 리스트 목록 불러오기
path = './routers/'
file_list = os.listdir(path)
file_list_py = [file.replace('.py', '') for file in file_list if file.endswith('.py')]
file_list_py.remove('__init__')

# router 리스트 router로 추가하기
for name in file_list_py:
    app.include_router(locals()[name].router)


if __name__ == "__main__":
    uvicorn.run(app="AIPaperboy:app", host="0.0.0.0", port=8000, reload=True)