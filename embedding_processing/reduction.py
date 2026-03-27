import numpy as np
from sklearn.manifold import TSNE


def reduce_embeddings(embedding: np.ndarray, n_components: int, perplexity: int = 40):
    reduced_embeddings = TSNE(
        n_components=n_components,
        learning_rate="auto",
        perplexity=perplexity,
        random_state=1848,
    ).fit_transform(embedding)

    return reduced_embeddings
