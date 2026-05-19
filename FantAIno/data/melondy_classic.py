import tensorflow as tf
import torch

from sklearn.model_selection import train_test_split

from FantAIno.data.melondy_base import MelondyBaseDataset
from FantAIno.utils.data_utils import extract_features

class MelondyClassicDataset(MelondyBaseDataset):
    """
    FantAIno dataset designed for classical machine learning algorithms.

    This dataset is designed to be entirely tabular, with no multi-modal capabilities.
    """

    def __init__(
        self,
        split: str = "train",
        data_type: str = "numpy",
        drop_cols: list[str] | None = None,
        openai_embedding_size: str = "small"
    ):

        super().__init__()
        self.album_data_df = self.retrieve_tabular_album_data()
        self.lyrics_embeddings_df = self.retrieve_lyrics_embeddings(openai_embedding_size=openai_embedding_size)

        dropped_cols = drop_cols or [
            # "artist", used as index
            # "album", used as index
            "genre_list",
            "featured_artists",
            "track_names",
        ]

        self.FantAIno_df = self.album_data_df.merge(self.lyrics_embeddings_df, on=["album_name", "artist_name"], how="left")
        print(f"NOTE: The number of rows with missing data: {(self.FantAIno_df.isnull().any(axis=1)).sum()} out of {self.FantAIno_df.shape[0]}.")
        self.FantAIno_df.fillna(0, inplace=True)

        self.FantAIno_df.set_index(["album_name", "artist_name"], inplace=True)
        self.FantAIno_df_X = extract_features(self.FantAIno_df, dropped_cols, omit_mode=True)
        self.FantAIno_df_y = self.FantAIno_df["rating"].astype(int)
        self.FantAIno_df_X_train, self.FantAIno_df_X_test, self.FantAIno_df_y_train, self.FantAIno_df_y_test = train_test_split(self.FantAIno_df_X, self.FantAIno_df_y, test_size=0.2, random_state=self.seed)

        match split:
            case "train":
                self.FantAIno_df_X = self.FantAIno_df_X_train
                self.FantAIno_df_y = self.FantAIno_df_y_train
            case "test":
                self.FantAIno_df_X = self.FantAIno_df_X_test
                self.FantAIno_df_y = self.FantAIno_df_y_test
            case _:
                raise ValueError(f"Invalid split: {split}")

        match data_type:
            case "numpy":
                self.FantAIno_df_X = self.FantAIno_df_X.to_numpy()
                self.FantAIno_df_y = self.FantAIno_df_y.to_numpy().reshape(-1, 1)
                self.FantAIno_index = self.FantAIno_df.index
            case "pytorch":
                self.FantAIno_df_X = torch.from_numpy(self.FantAIno_df_X.to_numpy())
                self.FantAIno_df_y = torch.from_numpy(self.FantAIno_df_y.to_numpy()).reshape(-1, 1)
                self.FantAIno_index = self.FantAIno_df.index
            case "tensorflow":
                self.FantAIno_df_X = tf.convert_to_tensor(self.FantAIno_df_X.to_numpy())
                self.FantAIno_df_y = tf.convert_to_tensor(self.FantAIno_df_y.to_numpy()).reshape(-1, 1)
                self.FantAIno_index = self.FantAIno_df.index
            case _:
                raise ValueError(f"The data type {data_type} is not supported.")
