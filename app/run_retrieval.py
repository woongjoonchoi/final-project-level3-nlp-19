from typing import Callable, List, Dict, NoReturn, Tuple
from datasets import Sequence, Value, Features, Dataset, DatasetDict
from transformers import TrainingArguments

from .arguments import DataTrainingArguments
from .sparse_retrieval import SparseRetrieval

def run_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "./app/data",
    context_path: str = "mbn_news_wiki.json",
) -> DatasetDict:
    print(datasets)
    print(f"-----------------------------------------run_retrieval-----------------------------------------")

    # Query에 맞는 Passage들을 Retrieval 합니다.
    # Spase Passage Retrieval 부분 
    retriever_sparse = SparseRetrieval(
        tokenize_fn=tokenize_fn.tokenize, data_path=data_path, context_path=context_path
    )

    df_sparse = retriever_sparse.elastic_retrieve(datasets["validation"], topk=data_args.top_k_retrieval)
    df = df_sparse

    print('======log1======')
    context_list = []
    print(len(df_sparse["context"][0]))
    for idx in range(len(df_sparse)):
        context_list.append(df_sparse["context"][idx])
    print('======log2======')

    for idx in range(len(df_sparse)):
        df["context"][idx] = " ".join(df_sparse["context"][idx])

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

    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    print("--------End Run_retrieval-----------")

    return datasets, context_list