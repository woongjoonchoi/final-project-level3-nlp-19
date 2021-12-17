from typing import List

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from . import crud, models, schemas
from .database import SessionLocal, engine

# 데이터베이스 테이블 생성
models.Base.metadata.create_all(bind=engine)

app = FastAPI()


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/users/", response_model=List[schemas.User], description="모든 users 읽어 옮")
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users


@app.post("/users/", response_model=schemas.User, description="내가 user 생성함")
def create_user(user: schemas.UserCreate, db:Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user.id)
    if db_user:
        raise HTTPException(status_code=400, detail="이미 등록된 아이디 입니다.")
    return crud.create_user(db=db, user=user)


@app.get("/users/{users_id}", response_model=schemas.User, description="내가 생성한 user_id 입력하면 정보 볼 수 있음")
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="해당 유저를 찾을 수 없습니다.")
    return db_user


@app.post("/users/{user_id}/scraps", response_model=schemas.Scrap, description="유저가 무엇을 스크랩할지 입력할 수 있음")
def create_scrap_for_user(user_id: int, scrap: schemas.ScrapCreate, db: Session = Depends(get_db)):
    return crud.create_user_scrap(db=db, scrap=scrap, user_id=user_id)







@app.get("/scraps/", response_model=List[schemas.Scrap], description="모든 scraps 읽어 옮")
def read_scraps(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    scraps = crud.get_scraps(db, skip=skip, limit=limit)
    return scraps






# 내가 scrap 생성함 -> 유저가 만들어야하기 때문에 작성X


@app.get("/news/", response_model=List[schemas.News], description="모든 news 읽어 옮")
def read_newss(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    newss = crud.get_newss(db, skip=skip, limit=limit)
    return newss


@app.post("/news/", response_model=schemas.Scrap, description="내가 news 생성함")
def create_scrap(user: schemas.ScrapCreate, db:Session = Depends(get_db)):
    db_scrap = crud.get_scrapr(db, scrap_id=scrap.id)
    if db_scrap:
        raise HTTPException(status_code=400, detail="이미 등록된 스크랩 입니다.")
    return crud.create_scrap(db=db, scrap=scrap)




# 유저 여러명 -> 속에 스크랩
# 관리자 여러명? 유저id와 겹치면 안 됨
# 뉴스 여러개
# 스크랩 여러개 -> 속에 유저, 뉴스

# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)
