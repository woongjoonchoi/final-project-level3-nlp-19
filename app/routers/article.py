from fastapi import APIRouter, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..import predict
from .. import database
from .. import tables

import random

router = APIRouter()
templates = Jinja2Templates(directory='./app/script')

@router.get('/article')
def get_text_form(request: Request):
    return templates.TemplateResponse('article_form.html', context={'request': request})

@router.post('/question')
def question_to_db(text: str = Form(...), db: Session = Depends(database.get_db)):
    new_question = tables.Question(id=random.randint(0, 1000000), text=text)

    db.add(new_question)
    db.commit()
    db.refresh(new_question)

    model, tokenizer = predict.load_model()
    prediction = predict.get_prediction(model, tokenizer, text)

    return prediction