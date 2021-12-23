from sqlalchemy.orm import Session
from schema import models, schemas

from random import randint

import numpy as np

def str2int(in_upper_str, in_base=36):
    """
    입력한 base system으로 대문자문자열을 정수로 반환한다 
    10 base는 10진수 1-9
    36 base는 1-9, A-Z 총 36가지로 표현된다
    """
    assert isinstance(in_upper_str, str)
    assert 2 <= in_base and in_base <= 36
    return int(in_upper_str, base=in_base)


# 사용자가 스크랩 정보를 관리한다.
class Managenewsscrap():
    # news_scrap_id가 이미 존재한지 확인
    def get_news_scrap_id(db: Session, user_id: str, news_id: str):
        news_scraps_ids = db.query(models.NewsScrap).filter(models.NewsScrap.user_id == user_id).all()

        str_user_id = str(str2int(user_id))
        news_scrap_id = int(news_id + str_user_id)

        for i in range(len(news_scraps_ids)):
            if news_scraps_ids[i].news_scrap_id == news_scrap_id:
                return True
        return False


    # 유저가 뉴스 스크랩 생성
    # 다시 작성해야 함, 현재는 news_id가 7자리여서 가능하나 아이디 글자수, 소문자+영어조합만 가능 제한이 있음(int 길이 때문에)
    def create_news_scrap(db: Session, user_id: str, news_id: str):
        str_user_id = str(str2int(user_id))
        news_scrap_id = news_id + str_user_id
        
        db_news_scrap = models.NewsScrap(user_id=user_id, user_news_id=news_id, news_scrap_id=int(news_scrap_id))
        db.add(db_news_scrap)
        db.commit()
        db.refresh(db_news_scrap)
        return db_news_scrap


    # 유저가 뉴스 스크랩 제거
    def delete_news_scrap(db: Session, user_id: str, news_id: str):
        str_user_id = str(str2int(user_id))
        news_scrap_id = news_id + str_user_id
        print(3333333333333333333333333333333333333333333333333333333333333333333333)
        db_news_scrap = db.query(models.NewsScrap).filter(models.NewsScrap.news_scrap_id == int(news_scrap_id)).delete()
        db.commit()
        return db_news_scrap