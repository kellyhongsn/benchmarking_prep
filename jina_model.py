import argparse
import logging
import os
import pathlib
import pickle
import requests
import json

from mteb import MTEB

JINA_API_KEY = os.getenv("JINA_API_KEY")

class JinaAIEmbedder:
    """
    Benchmark JinaAI's embeddings endpoint.
    """
    def __init__(self, model="jina-embeddings-v3", task_name=None, batch_size=32, save_emb=False, **kwargs):
        self.model = model
        self.batch_size = batch_size
        self.save_emb = save_emb
        self.base_path = f"embeddings/{model}/"
        self.task_name = task_name
        self.url = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}"
        }

        if save_emb:
            assert self.task_name is not None

        pathlib.Path(self.base_path).mkdir(parents=True, exist_ok=True)

    def encode(self, sentences, decode=True, idx=None, **kwargs):
        fin_embeddings = []

        embedding_path = f"{self.base_path}/{self.task_name}_{sentences[0][:10]}_{sentences[-1][-10:]}.pickle"
        if sentences and os.path.exists(embedding_path):
            loaded = pickle.load(open(embedding_path, "rb"))
            fin_embeddings = loaded["fin_embeddings"]
        else:
            for i in range(0, len(sentences), self.batch_size):
                batch = sentences[i: i + self.batch_size]

                # Format input for JinaAI API
                data = {
                    "model": self.model,
                    "dimensions": 1024,  # Adjust based on the model
                    "normalized": True,
                    "embedding_type": "float",
                    "input": [{"text": sentence} for sentence in batch]
                }

                # API Request to JinaAI
                response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
                if response.status_code != 200:
                    raise Exception(f"JinaAI API error: {response.text}")

                response_json = response.json()
                out = [item["embedding"] for item in response_json["data"]]

                fin_embeddings.extend(out)

        # Save embeddings
        if fin_embeddings and self.save_emb:
            dump = {
                "fin_embeddings": fin_embeddings,
            }
            pickle.dump(dump, open(embedding_path, "wb"))

        assert len(sentences) == len(fin_embeddings)
        return fin_embeddings


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--startid", type=int)
    parser.add_argument("--endid", type=int)
    parser.add_argument("--model", type=str, default="jina-embeddings-v3")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--taskname", type=str, default=None)
    parser.add_argument("--batchsize", type=int, default=2048)
    args = parser.parse_args()
    return args


def main(args):
    for task in TASK_LIST[args.startid:args.endid]:
        print("Running task: ", task)
        model = JinaAIEmbedder(args.model, task_name=task, batch_size=args.batchsize, save_emb=True)
        eval_splits = ["validation"] if task == "MSMARCO" else ["test"]
        model_name = args.model.replace("/", "_")  # Ensure valid folder names
        evaluation = MTEB(tasks=[task], task_langs=[args.lang])
        evaluation.run(model, output_folder=f"results/{model_name}", batch_size=args.batchsize, eval_splits=eval_splits, corpus_chunk_size=10000)


if __name__ == "__main__":
    args = parse_args()
    main(args)