import os
import json
import time
import faiss
import pickle
import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union


from dpr_train_generate import BertEncoder, run_dpr
from sklearn.feature_extraction.text import TfidfVectorizer

from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)

from elasticsearch import Elasticsearch, helpers
from retrieval_common_part import retrieve_faiss
from sparse_retrieval import SparseRetrieval

class DenseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        datasets,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        self.tokenize_fn = tokenize_fn
        self.datasets = datasets
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Transform by vectorizer
        self.p_encoder = None
        self.p_embedding = None  
        self.indexer = None  # build_faiss()로 생성합니다.

    def get_dense_embedding(self, inbatch) -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들며
            q_encoder, P_encoder 모델을 저장하고 P_Embedding을 pickle로 저장합니다.
            만약 미리 저장된 모델과 파일이 있으면 저장된 모델과, pickle을 불러옵니다.
        """

        # Pickle과 모델을 저장합니다.
        if inbatch == False:
            pickle_name = f"dense_embedding.bin"
            q_encoder_name = f"q_encoder.pt"
            emd_path = os.path.join(self.data_path, pickle_name)
            q_model_path = os.path.join("./models/train_dataset", q_encoder_name)
        else:
            pickle_name = f"dense_embedding_in.bin"
            q_encoder_name = f"q_encoder_in0.pt"
            emd_path = os.path.join(self.data_path, pickle_name)
            q_model_path = os.path.join("./models/train_dataset", q_encoder_name)


        if os.path.isfile(emd_path) and os.path.isfile(q_model_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
                # self.p_embedding = self.p_embedding.to('cpu').numpy()
            self.q_encoder = torch.load(q_model_path)
            print("Dense Embedding pickle load.")
        else:
            print("Build passage sparse_embedding")
            retriever = SparseRetrieval(tokenize_fn=self.tokenize_fn)
            self.q_encoder, self.p_embedding = run_dpr(self.contexts, self.tokenize_fn, retriever, inbatch=inbatch)
            # print(self.p_embedding.shape)

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        with torch.no_grad():
            self.q_encoder.eval()

        q_seqs_val = tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to('cuda')
        query_vec = self.q_encoder(**q_seqs_val).to('cpu')  #(num_query, emb_dim)

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        query_vecs = []
        with torch.no_grad():
            self.q_encoder.eval()    
            for q in queries:
                q = self.tokenize_fn(q, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                q_emb = self.q_encoder(**q).to('cpu').numpy()
                query_vecs.append(q_emb)
            query_vecs = torch.Tensor(query_vecs).squeeze()

        q_embs = query_vecs.to('cpu').numpy()
        print(q_embs.dtype)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()
