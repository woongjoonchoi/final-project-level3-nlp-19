from elasticsearch import Elasticsearch
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
INDEX_NAME = "news_wiki_index_update"
# 뉴스 기사 목록 불러오기
class Homeboard():
    def __init__(self) :
        self.es = Elasticsearch()
        if not self.es.indices.exists(INDEX_NAME) :
            raise Exception("INDEX {0} not exists. run inference_copy.py first".format(INDEX_NAME))
            exit()
        
    # self.query = "사랑하지만 힘들어 죽겠네"
    # 뉴스 기사 제목 리스트 불러오기
    def get_news_title(self , query = "사랑하지만 힘들어 죽겠네") :
        res = self.es.search(index=INDEX_NAME, q=query, size=5)
        return res
    