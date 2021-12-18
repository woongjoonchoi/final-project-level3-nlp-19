import uvicorn
from fastapi import FastAPI, APIRouter , Depends
from pydantic import BaseModel
from database import engine , SessionLocal
import schemas , models
from sqlalchemy.orm import Session
from router import *
from fastapi.responses import RedirectResponse

models.Base.metadata.create_all(engine)



app = FastAPI()

# @app.get("/")
# def get_home_page():
#     return RedirectResponse("./home")
app.include_router(aiscrap_router)
app.include_router(homescrap_router)

## if 안에서 지역변수 취급 받는듯
if __name__=='__main__' :
    print(dir())


    uvicorn.run(app = "main:app", host = "0.0.0.0" , port = 8000,reload=True )
    # uvicorn.run(app , host = "0.0.0.0" , port =8000)
