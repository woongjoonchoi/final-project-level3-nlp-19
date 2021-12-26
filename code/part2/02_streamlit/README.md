## QA 데모 in streamlit



Streamlit으로 QA데모를 진행 해보았다.





### HF(huggingface) 특징



tokenizier caching을 하는 방법이 꽤 까다롭다. streamlit의 hash func의 기능에 대해 자세히 알아야만 할 수 있다고 여겨진다.

따라서, 필자는 모델만 cache하고 나머지 , tokenizer등등의 부품들은 prediction 과정에서 loading하였다.



### 데모

![12-12-23-41](https://user-images.githubusercontent.com/50165842/145717023-6e692dfb-58b9-4920-af91-0e9994f72444.gif)



궁금해 입력칸은 그냥 넣어봤다. 



텍스트 가 기억이 안나서 일일이 복사해봤다. 

### 실행방법

```
streamlit run app.py --server.address=127.0.0.1
```

part2 디렉토리에 가서 실행하면 된다.



### Domain

이 QA는 news domain중 정치,사회 쪽에 fine-tuning했다. 



### Model

[wjc123/qa_finetuned · Hugging Face](https://huggingface.co/wjc123/qa_finetuned)

여기에서 받을 수 있다.