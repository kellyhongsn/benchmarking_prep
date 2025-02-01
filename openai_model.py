import argparse
import logging
import os
import pathlib
import pickle

from mteb import MTEB
import openai
import tiktoken


API_KEY = os.getenv('OPENAI_API_KEY')

class OpenAIEmbedder:
    """
    Benchmark OpenAIs embeddings endpoint.
    """
    def __init__(self, engine, task_name=None, batch_size=32, save_emb=False, **kwargs):
        self.engine = engine
        self.max_token_len = 8191
        self.batch_size = batch_size
        self.save_emb = save_emb
        self.base_path = f"embeddings/{engine.split('/')[-1]}/"
        self.tokenizer = tiktoken.get_encoding('cl100k_base')
        self.task_name = task_name
        self.client = openai.OpenAI(api_key=API_KEY)

        if save_emb:
            assert self.task_name is not None

        pathlib.Path(self.base_path).mkdir(parents=True, exist_ok=True)
        
    def encode(self, 
            sentences,
            decode=True,
            idx=None,
            **kwargs
        ):

        fin_embeddings = []

        embedding_path = f"{self.base_path}/{self.task_name}_{sentences[0][:10]}_{sentences[-1][-10:]}.pickle"
        if sentences and os.path.exists(embedding_path):
            loaded = pickle.load(open(embedding_path, "rb"))
            fin_embeddings = loaded["fin_embeddings"]
        else:
            for i in range(0, len(sentences), self.batch_size):
                batch = sentences[i : i + self.batch_size]

                batch = [self.tokenizer.decode(
                    self.tokenizer.encode(sentence)[:self.max_token_len]) 
                    for sentence 
                    in batch]
                
                out = [datum.embedding for datum in self.client.embeddings.create(input=batch, model=self.engine).data]

                fin_embeddings.extend(out)

        if fin_embeddings and self.save_emb:
            dump = {
                "fin_embeddings": fin_embeddings,
            }
            pickle.dump(dump, open(embedding_path, "wb"))

        assert len(sentences) == len(fin_embeddings)
        return fin_embeddings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--startid", type=int)
    parser.add_argument("--endid", type=int)
    parser.add_argument("--engine", type=str, default="text-embedding-3-small")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--taskname", type=str, default=None)
    parser.add_argument("--batchsize", type=int, default=2048)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()