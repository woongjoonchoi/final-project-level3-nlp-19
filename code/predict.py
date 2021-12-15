import torch
import streamlit as st
from typing import Tuple
import logging
import sys
import os
from typing import Callable, List, Dict, NoReturn, Tuple
import pandas as pd
import json

import numpy as np
from configure import *
from preprocess import *
from datasets import (
    load_metric,
    load_dataset,
    load_from_disk,
    Sequence,
    Value,
    Features,
    Dataset,
    DatasetDict,
)

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from utils_qa import postprocess_qa_predictions, check_no_error
from trainer_qa import QuestionAnsweringTrainer
from sparse_retrieval import SparseRetrieval
from dense_retrieval import DenseRetrieval
from retrieval_common_part import build_faiss, retrieve_faiss
import retrieval_common_part
from postprocessing import post_processing_function
from run_mrc import run_combine_mrc

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)
from inference_copy import run_retrieval


logger = logging.getLogger(__name__)

# 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
# --help flag 를 실행시켜서 확인할 수 도 있습니다.
parser = HfArgumentParser(
    (ModelArguments, DataTrainingArguments, TrainingArguments)
)
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
model_args.model_name_or_path = './models/train_dataset/'
data_args.output_dir = './outputs/test_dataset/'

print(training_args)

# logging 설정
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
logger.info("Training/evaluation parameters %s", training_args)

# 모델을 초기화하기 전에 난수를 고정합니다.
set_seed(training_args.seed)


def load_model():

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    model, tokenizer = configure_model(model_args, training_args, data_args)


    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    print("init trainer...")
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer( 
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function
    )

    print('load done!')

    return model, tokenizer


def run_retrieval_str(    
    tokenize_fn: Callable[[str], List[str]],
    sentence: str,
    data_args: DataTrainingArguments,
    data_path: str = "../data",
    context_path: str = "mbn_news_wiki.json",
) -> DatasetDict:

    
    retriever_sparse = SparseRetrieval(
        tokenize_fn=tokenize_fn.tokenize, data_path=data_path, context_path=context_path
        )

    
    context_list = retriever_sparse.elastic_retrieve(sentence, topk=data_args.top_k_retrieval)
    
    tmp = {
            # Query와 해당 id를 반환합니다.
            "question": sentence,
            "id": '1',
            # Retrieve한 Passage의 id, context를 반환합니다.
            "context": " ".join(context_list)
            }

    df = pd.DataFrame([tmp])

    f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    print(df['context'][0])
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})    
    return datasets



def get_prediction(model, tokenizer, sentence):
    print('run retreival')

    sample_test = {"data": [
		{
			"id": "1",
			"question": sentence
		}]}


    with open(os.path.join(data_args.dataset_name, 'sample_test.json'), "w", encoding='utf-8') as json_file:
        json.dump(sample_test, json_file, indent=4, sort_keys=True)


    datasets = load_dataset('json', data_files={'validation':os.path.join(data_args.dataset_name, 'sample_test.json')}, field='data')
    print(datasets)

    datasets = run_retrieval(
            tokenizer,
            datasets,
            training_args,
            data_args,
        )

    _ , max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    if training_args.do_eval or training_args.do_predict:
        predictions = run_combine_mrc(data_args, training_args, model_args, datasets, tokenizer, model)

    # column_names = datasets["validation"].column_names

    # eval_dataset = datasets["validation"]

    # print('start preprocessing')
    # prepare_valid_features = preprocess_extract_valid(tokenizer, data_args, column_names, max_seq_length)

    # # Validation Feature 생성
    # eval_dataset = eval_dataset.map(
    #     prepare_valid_features,
    #     batched=True,
    #     num_proc=data_args.preprocessing_num_workers,
    #     remove_columns = ["id", "context", "question"]
    # )
    # print('eval_dataset sample')
    # print(eval_dataset[0])
    
    # # print([eval_dataset[i]['example_id'] for i in range(len(eval_dataset))])

    # print('start prediction')
    # predictions = trainer.predict(test_dataset=eval_dataset, test_examples=datasets["validation"])
    # print('prediction done')
    # # data_output_dir를 설정하지 않으면 저장이되지 않는다.
    return predictions