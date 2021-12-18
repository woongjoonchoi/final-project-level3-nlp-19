from fastapi import FastAPI, APIRouter
from fastapi.templating import Jinja2Templates
import uvicorn

router = APIRouter(prefix="/aiscrap", tags=["AIScap"])
templates = Jinja2Templates(directory='./templates')


# AI scrap 페이지로 이동(웅준)
@router.get("/")
def get_aiscrap_page():
    # return templates.TemplateResponse('aiscrap.html', context={'request':request})
    pass





if __name__ == '__main__':
    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app="AIPaperboy:app", host="0.0.0.0", port=8000, reload=True)