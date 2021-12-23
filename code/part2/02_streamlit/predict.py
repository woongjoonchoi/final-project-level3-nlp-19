import torch
import streamlit as st
from model import MyEfficientNet
from utils import transform_image
import yaml
from typing import Tuple
import numpy as np
import warnings
from transformers import pipeline ,AutoTokenizer ,AutoModelForQuestionAnswering
from preprocess import *
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
model_checkpoint = "wjc123/qa_finetuned"
def my_hash_func():

    return AutoTokenizer.from_pretrained(model_checkpoint)

@st.cache()
def load_model() -> pipeline :
    model  =  AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    model.eval()
    return model

def get_prediction(model : AutoModelForQuestionAnswering , context:str , question : str) -> Tuple[torch.Tensor , torch.Tensor] :

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
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

    return best_answer

