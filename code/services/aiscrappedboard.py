
from pydantic import BaseModel
from schema.database import engine , SessionLocal

from schema import schemas , models
from sqlalchemy.orm import Session

from elasticsearch import Elasticsearch

es = Elasticsearch()
# AI가 스크랩한 뉴스기사 목록 불러오기
class Aiscrappedboard():

    # AI가 스크랩한 뉴스기사 제목 리스트 가져오기
    def get_user_news(self , db,owner_user_id ):
        news = db.query(models.AIInput).filter(models.AIInput.user_id==owner_user_id).all()
        
        return news
    def user_news_title(self,news_id) :
        news_list = es.get(index = "news_wiki_index_update", id=news_id)

        return news_list["_source"]["title"] , news_list["_source"]["context"]
    pass