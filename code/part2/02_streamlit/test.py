from transformers import pipeline , AutoTokenizer , AutoModelForQuestionAnswering

import torch
import numpy as np
from preprocess import *
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


model_checkpoint = "wjc123/qa_finetuned"

model  =  AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)



context = """중국의 한 여성 경찰이 아파트에서 추락하던 3세 아이를 살리고 자신은 혼수상태에 빠졌습니다. 의인(義人)의 소식이 알려지자 각박한 중국 사회에 큰 반향을 일으키고 있습니다. 5일 귀주도시망 등 중국 현지 언론에 따르면 구이저우성 카일리시에 보조 교통 경찰로 일하는 천중핑(49)은 지난달 28일 한 아파트에서 비상 상황이 발생했다는 연락을 받고 현장으로 향했습니다. 도착했을 때 아파트 4층 창문에서 여자 아이가 매달려 있었습니다. 곧이어 아이는 손에 힘이 빠지면서 밑으로 추락했습니다. 천중핑과 다른 세명의 이웃들이 달려갔습니다. 그리고 아이는 바닥이 아니라 천중핑의 팔에 떨어졌습니다. 중간 비막이 천막 때문에 속도가 줄기는 했지만 추락의 충격은 천중핑이 고스란히 감당해야 했습니다. 아이는 즉시 병원으로 옮겨져 치료를 받았습니다. 다리 골절로 그리 심각한 상황은 아니라고 합니다. 하지만 생명의 은인이자 영웅은 커다란 댓가를 치러야 했다. 뇌출혈로 인한 의식불명 상태에 빠진 것이다. 다행히 이틀 간의 코마 상태 이후 의식을 회복해 지난 2일부터 중환자실에서 치료를 받고 있습니다. 아이는 열쇠공이 문을 따는 소리에 겁을 먹고 창문 밖으로 도망을 치려다 사고를 당한 것으로 전해졌습니다. 아이가 잠든 사이 돌보던 아이의 할머니가 쓰레기를 버리러 나갔다가 문이 잠기는 바람에 열쇠공을 불렀던 것입니다. 아이의 엄마는 “천중핑의 도움이 없었다면 아이는 죽었을 것”이라며 딸을 구해준 천중핑에게 감사의 뜻을 전했습다. 카일리시 정부 대표와 공안부 관계자들도 천중핑이 입원한 병원을 찾아 위로하고 회복될때까지 도움을 아끼지 않겠다고 밝혔습니다. 천중핑의 선행 사실을 접한 중국 기업 알리바바도 ‘중국의 좋은 이웃상’과 함께 상금 1만 위안(약 170만원)을 수여하기로 했습니다. [아직 살만한 세상]은 점점 각박해지는 세상에 희망과 믿음을 주는 이들의 이야기입니다. 힘들고 지칠 때 아직 살만한 세상을 만들어가는 ‘아살세’ 사람들의 목소리를 들어보세요. 따뜻한 세상을 꿈꾸는 독자 여러분의 제보를 기다립니다. 맹경환 기자 khmaeng@kmib.co.kr"""
question = "중국에서 아파트에서 추락하던 3세 아이를 살리고 자신은 혼수상태에 빠진 사람은 누구야?"

# question_answerer = pipeline("question-answering", model=model_checkpoint)

# question_answerer(context =context , question = question)
n_best = 20
max_answer_length = 30
predicted_answers = []


examples = {"id" : "0" , "question" : question , "context" : context}
feature_extractor = preprocess_extract_valid(tokenizer,384)
eval_example = feature_extractor(examples)
offset_mapping = eval_example.pop('offset_mapping')
example_id = eval_example.pop('example_id')
input_ids = torch.tensor(eval_example['input_ids'] , dtype=torch.long)
attention_mask = torch.tensor(eval_example['attention_mask'], dtype=torch.long)
outputs  = model(input_ids = input_ids  ,attention_mask= attention_mask)
start_logits = outputs.start_logits
end_logits=outputs.end_logits

start_logits = start_logits.detach().cpu().numpy()
end_logits = end_logits.detach().cpu().numpy()

answers = []


for feature_index in range(len(example_id)):

    offsets = offset_mapping[feature_index]
    start_logit = start_logits[feature_index]
    end_logit = end_logits[feature_index]
    start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
    end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()

    for start_index in start_indexes:
        for end_index in end_indexes:
            # Skip answers that are not fully in the context
            if offsets[start_index] is None or offsets[end_index] is None:
                continue
            # Skip answers with a length that is either < 0 or > max_answer_length.
            if (
                end_index < start_index
                or end_index - start_index + 1 > max_answer_length
            ):
                continue

            answers.append(
                {
                    "text": context[offsets[start_index][0] : offsets[end_index][1]],
                    "logit_score": start_logit[start_index] + end_logit[end_index],
                }
            )

best_answer = max(answers, key=lambda x: x["logit_score"])
predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})

print(best_answer)