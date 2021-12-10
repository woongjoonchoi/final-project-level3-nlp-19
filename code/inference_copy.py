"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.
대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""


import logging
import sys
from typing import Callable, List, Dict, NoReturn, Tuple

import numpy as np
from configure import *
from preprocess import *
from datasets import (
    load_metric,
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
from postprocessing import post_processing_function
from run_mrc import run_combine_mrc

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)


logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # training_args.do_train = True

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

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

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    model, tokenizer = configure_model(model_args, training_args, data_args)

    # rue일 경우 : run passage retrieval
    if data_args.eval_retrieval:
        datasets = run_retrieval(
            tokenizer,
            datasets,
            training_args,
            data_args,
        )


    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_combine_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:
    print(f"-----------------------------------------run_retrieval-----------------------------------------")

    # Query에 맞는 Passage들을 Retrieval 합니다.

    # Spase Passage Retrieval 부분 
    retriever_sparse = SparseRetrieval(
        tokenize_fn=tokenize_fn.tokenize, data_path=data_path, context_path=context_path
        )

    if data_args.sparse_name == "None":
        retriever_sparse.get_sparse_embedding()
        if data_args.use_faiss:
            # 수정
            retrieval_common_part.build_faiss(num_clusters=data_args.num_clusters)
            df_sparse = retrieval_common_part.retrieve_faiss(
                datasets["validation"], topk=data_args.top_k_retrieval
            )
        else:
            df_sparse = retriever_sparse.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)
        
    elif data_args.sparse_name == "elastic":
        df_sparse = retriever_sparse.elastic_retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # 테스트
    # Dense Passage Retrieval 부분
    retriever_dense = DenseRetrieval(
        tokenize_fn=tokenize_fn, datasets=datasets, data_path=data_path, context_path=context_path 
    )
    if data_args.dense_name == "None":
        retriever_dense.get_dense_embedding(inbatch=False)
        # 수정
        retrieval_common_part.build_faiss(num_clusters=data_args.num_clusters)
        df_dense = retrieval_common_part.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    elif data_args.dense_name == "in-batch":
        retriever_dense.get_dense_embedding(inbatch=True)
        # 수정
        retrieval_common_part.build_faiss(num_clusters=data_args.num_clusters)
        df_dense = retrieval_common_part.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )


    # sparse만 이용하는 경우
    df = df_sparse

    for idx in range(len(df_sparse)):
        df["context"][idx] = " ".join(df_sparse["context"][idx])

    # DPR 미사용으로 해당 코드 주석 처리
    # df = df_sparse
    # # Sparse Retrieval 결과와 Dense Retrieval 결과를 병합합니다. 
    # for idx in range(len(df_sparse)):
    #     if idx == 10:
    #         print(df_dense["context"][idx])
    #     temp = df_sparse["context"][idx] + df_dense["context"][idx]
    #     df["context"][idx] = " ".join(temp)

    # # Dense Retrieval 결과 일부 출력하기        
    # df = df_dense
    # for idx in range(len(df_dense)):
    #     if idx % 1000 == 0:
    #         print(df["context_id"][idx])
    #         print('-----------')
    #         print(df["question"][idx])
    #         print('-----------')
    #         print(df["context"][idx])
    #     df["context"][idx] = " ".join(df_dense["context"][idx])


    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    
    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    print(df)
    print(f)
    
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


if __name__ == "__main__":
    main()