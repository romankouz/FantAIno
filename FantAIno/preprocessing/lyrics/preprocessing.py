"""

Library of preprocessing modules for lyrics data in the FantAIno project.

Class Name Format: {task}_{data_input}_{method}

"""
import numpy as np
import pandas as pd
import warnings

from sklearn.decomposition import PCA

class Lyrics_Embeddings_PCA():

    def __init__(self, n_components: int = 100, **pca_kwargs):
        self.n_components = n_components
        self.pca_kwargs = pca_kwargs

    def __call__(self, lyrics_embeddings_df: pd.DataFrame) -> pd.DataFrame:
        warnings.warn(f"WARNING: Lyrics embeddings table has {(lyrics_embeddings_df.isnull().any(axis=1)).sum()} missing embeddings out of {lyrics_embeddings_df.shape[0]} entries.")
        lyrics_embeddings_df.fillna(0, inplace=True)
        pca = PCA(n_components=self.n_components, **self.pca_kwargs)
        pca_array = pca.fit_transform(lyrics_embeddings_df.select_dtypes(include=[np.number]))
        print(f"Your PCA model has explained {round(100 * pca.explained_variance_ratio_.cumsum()[-1], 2)}% of the total variance in the lyrics embeddings.")
        lyrics_embeddings_df_pca = pd.concat(
            [
                lyrics_embeddings_df[["album_name", "artist_name"]],
                pd.DataFrame(pca_array, columns=[f"embedding_pc_{i}" for i in range(self.n_components)])
            ],
            axis=1
        )
        return lyrics_embeddings_df_pca
