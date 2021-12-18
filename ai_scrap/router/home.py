from fastapi import FastAPI, APIRouter
from fastapi.templating import Jinja2Templates
import uvicorn
from elasticsearch import Elasticsearch , helpers
# from services.homeboard import Homeborad


INDEX_NAME = "mbn_index2"


INDEX_SETTINGS = {
  "settings" : {
    "index":{
      "analysis":{
        "analyzer":{
          "korean":{
            "type":"custom",
            "tokenizer":"nori_tokenizer",
            "filter": [ "shingle" ],

          }
        }
      }
    }
  },
  "mappings": {

      "properties" : {
        "CONTEXT" : {
          "type" : "text",
          "analyzer": "korean",
          "search_analyzer": "korean"
        },
        "TITLE" : {
          "type" : "text",
          "analyzer": "korean",
          "search_analyzer": "korean"
        },
        "DATE" :{
            "type" : "text",
          "analyzer": "korean",
          "search_analyzer": "korean"
        },
        "CATEGORY" :{
            "type" : "text",
          "analyzer": "korean",
          "search_analyzer": "korean"
        }
      }

  }
}
# import .index_settings
homescrap_router = APIRouter(prefix="/home", tags=["Home"])
templates = Jinja2Templates(directory='./templates')

try:
    es.transport.close()
except:
    pass
es = Elasticsearch()
if es.indices.exists(INDEX_NAME):
    pass
# 뉴스 홈페이지 화면이동
@homescrap_router.get("/")
def get_home_page():
    # Homeboard Service 객체로 뉴스 목록 가져오기
    query = "사랑하지만 힘들어 죽겠네"
    res = es.search(index=INDEX_NAME, q=query, size=10)

    return res

if __name__ == '__main__':
    app = FastAPI()
    app.include_router(homescrap_router)
    uvicorn.run(app="AIPaperboy:app", host="0.0.0.0", port=8000, reload=True)