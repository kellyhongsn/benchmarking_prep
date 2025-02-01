from evals import evaluate
import pandas as pd
import chromadb
import os
from dotenv import load_dotenv
from openai import OpenAI
from chromadb.utils import embedding_functions
from tqdm import tqdm
import mteb
import json
from openai_model import OpenAIEmbedder
from jina_model import JinaAIEmbedder
import numpy as np

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
jina_api_key = os.getenv('JINA_API_KEY')

def custom_results(chroma_client, qrels_df, queries, query_ids, corpus, corpus_ids, ef, dataset_name, model_name):

    data_model_name = f"{dataset_name}_{model_name}"

    new_collection = chroma_client.get_or_create_collection(
        name=data_model_name, 
        embedding_function=ef,
        metadata={
            "hnsw:space": "cosine"
        }
    )

    print(f"{data_model_name} collection created")

    batch_size = 100

    for i in tqdm(range(0, len(corpus), batch_size), desc="Processing Batches"):
        batch_documents = corpus[i:i + batch_size]
        batch_ids = corpus_ids[i:i + batch_size]

        new_collection.add(
            documents=batch_documents,
            ids=batch_ids
        )

    print("embedding complete")

    results = dict()
    results["query-id"] = []
    results["corpus-id"] = []
    results["score"] = []

    for i in tqdm(range(0, len(queries), batch_size), desc="Processing Batches"):
        batch_documents = queries[i:i + batch_size]
        batch_ids = query_ids[i:i + batch_size]

        results["query-id"].extend(batch_ids)

        batch_result = new_collection.query(
            query_texts=batch_documents,
            n_results=10
        )

        scores = [[1 - item for item in sublist] for sublist in batch_result["distances"]]

        results["corpus-id"].extend(batch_result["ids"])
        results["score"].extend(scores)

    
    results_df = pd.DataFrame(results)

    results_df.to_parquet(f"df_results/{data_model_name}_results_df.parquet", engine="pyarrow", index=False)

    # results_df = pd.read_parquet(f"df_results/{data_model_name}_results_df.parquet", engine="pyarrow")

    # results = results_df.to_dict(orient="list")

    # results["score"] = np.array(results["score"]).reshape(-1, 10).tolist()

    print("evaluating")

    qrels_dict = qrels_df.groupby("query-id").apply(lambda g: dict(zip(g["corpus-id"], g["score"]))).to_dict()

    k_values = [1, 3, 5, 10]

    qrels_dict = {
        qid: {doc_id: int(score) for doc_id, score in doc_dict.items()}
        for qid, doc_dict in qrels_dict.items()
    }

    results_dict = dict()

    for query_id, doc_ids, scores in zip(
        results["query-id"],
        results["corpus-id"],
        results["score"],
    ):
        if query_id not in results_dict:
            results_dict[query_id] = {}

        for doc_id, score in zip(doc_ids, scores):
            results_dict[query_id][doc_id] = score 


    ndcg, _map, recall, precision, top_k_accuracy = evaluate(
        qrels=qrels_dict, 
        results=results_dict, 
        k_values=k_values)

    final_result = {
        "NDCG": ndcg,
        "MAP": _map,
        "Recall": recall,
        "Precision": precision,
        "Top-K Accuracy": top_k_accuracy
    }

    with open(f"custom_results/{dataset_name}/{model_name}.json", "w") as f:
        json.dump(final_result, f, indent=4)



def run_mteb_default(task_name):
    model = mteb.get_model("all-MiniLM-L6-v2")
    tasks = mteb.get_tasks(tasks=[task_name])
    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(model, output_folder=f"mteb_results/{task_name}/all-MiniLM-L6-v2")


def run_mteb_openai(task_name, size):
    model = OpenAIEmbedder(engine=f"text-embedding-3-{size}")
    tasks = mteb.get_tasks(tasks=[task_name])
    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(model, output_folder=f"mteb_results/{task_name}/text-embedding-3-{size}")


def run_mteb_jina(task_name):
    model = JinaAIEmbedder(engine="jina-embeddings-v3")
    tasks = mteb.get_tasks(tasks=[task_name])
    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(model, output_folder=f"mteb_results/{task_name}/jina-embeddings-v3")