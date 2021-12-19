import os

from fastapi import FastAPI, Form, Request, Depends
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from . import tables
from .database import engine, get_db
from .routers import article

app = FastAPI()
templates = Jinja2Templates(directory='./app/script')

tables.Base.metadata.create_all(engine)

path = './app/routers/'
file_list = os.listdir(path)
file_list_py = [file.replace('.py', '') for file in file_list if file.endswith('.py')]
file_list_py.remove('__init__')

# router 리스트 router로 추가하기
for name in file_list_py:
    app.include_router(locals()[name].router)

@app.get('/main') #/main
def main_page(request: Request):
    return templates.TemplateResponse('main_form.html', context={'request': request})

