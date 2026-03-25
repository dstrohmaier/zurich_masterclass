from pathlib import Path

# import numpy as np
from gensim import downloader
from gensim.models.keyedvectors import KeyedVectors

from embedding_processing.reduction import reduce_embeddings


def reduce_w2v(n_components: int) -> KeyedVectors:
    w2v_kv = downloader.load("word2vec-google-news-300")
    embeddings = w2v_kv.vectors
    reduced_embeddings = reduce_embeddings(embeddings, n_components=n_components)

    reduced_w2v_kv = KeyedVectors(vector_size=n_components)

    keys = sorted(w2v_kv.key_to_index.keys(), key=lambda k: w2v_kv.key_to_index[k])

    reduced_w2v_kv.add_vectors(keys, reduced_embeddings)

    return reduced_w2v_kv


def store_reduced_w2v(output_dir: Path, n_components: int = 2) -> None:
    reduced_w2v_kv = reduce_w2v(n_components=n_components)
    reduced_w2v_kv.save_word2vec_format(output_dir / f"word2vec_n{n_components}.w2v")
