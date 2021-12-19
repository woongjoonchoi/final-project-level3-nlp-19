from elasticsearch import Elasticsearch , helpers
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
INDEX_NAME = "mbn_index2"
# 뉴스 기사 목록 불러오기
class Homeborad():
    def __init__(self) :
        try:
            es.transport.close()
        except:
            pass
        if not es.indices.exists(INDEX_NAME) :
            raise Exception("INDEX {0} not exists".format(INDEX_NAME))
            exit()
        self.es = Elasticsearch()
        # self.query = "사랑하지만 힘들어 죽겠네"
    # 뉴스 기사 제목 리스트 불러오기
    def get_news_title(self , query = "사랑하지만 힘들어 죽겠네") :
        res = self.es.search(index=INDEX_NAME, q=query, size=5)
        return res
    