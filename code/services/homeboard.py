from elasticsearch import Elasticsearch

es = Elasticsearch()

def search(index, data=None):
    if data is None:
        data = {"match_all" : {}}
    else:
        data = {"match" : data}
    body = {"query" : data}
    res = es.search(index=index, body=body)
    return res


# 뉴스 기사 목록 불러오기
class Homeborad():
    # 뉴스 기사 제목 리스트 불러오기
    def search():
        res = search(index = "news_wiki_index", data=None)
        return res
    