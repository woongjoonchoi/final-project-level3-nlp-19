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
    print(in_upper_str)
    print(111111111111111111111111111111111111111111111111111111111111111111111)
    print(in_upper_str.upper())
    print(111111111111111111111111111111111111111111111111111111111111111111111)
    print(in_upper_str)
    # assert in_upper_str.upper() == in_upper_str    
    assert 2 <= in_base and in_base <= 36
    return int(in_upper_str, base=in_base)


# 사용자가 스크랩 정보를 관리한다.
class Managenewsscrap():

    # 유저가 뉴스 스크랩 생성

    def create_news_scrap(db: Session, user_id: str, news_id: str):
        str_user_id = str(str2int(user_id))
        news_scrap_id = news_id + str_user_id
        
        db_news_scrap = models.NewsScrap(user_id=user_id, user_news_id=news_id, news_scrap_id=int(news_scrap_id))
        db.add(db_news_scrap)
        db.commit()
        db.refresh(db_news_scrap)
        return db_news_scrap


    # 유저가 뉴스 스크랩 제거
    def delete_news_scrap(db: Session, user_id: str, user_news_id: str):
        # db에서 news_scrap_id 꺼내오기
        db_news_scrap = models.NewsScrap(user_id=user_id, user_news_id=user_news_id, news_scrap_id=randint(1, 100000000))
        db.delete(db_news_scrap)
        db.commit()
        db.refresh(db_news_scrap)
        return db_news_scrap

    pass
