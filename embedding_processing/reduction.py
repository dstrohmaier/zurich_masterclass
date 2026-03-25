import numpy as np
from sklearn.manifold import TSNE


def reduce_embeddings(
    embedding: np.ndarray, n_components: int, perplexity: int = 30
):
    reduced_embeddings = TSNE(
        n_components=n_components, learning_rate="auto", perplexity=perplexity
    ).fit_transform(embedding)

    return reduced_embeddings
