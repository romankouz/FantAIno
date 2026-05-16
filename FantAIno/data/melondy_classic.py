from sklearn.model_selection import train_test_split

from FantAIno.data.melondy_base import MelondyBaseDataset

class MelondyClassicDataset(MelondyBaseDataset):
    """
    FantAIno dataset designed for classical machine learning algorithms.

    This dataset is designed to be entirely tabular, with no multi-modal capabilities.
    """

    def __init__(self, split: str = "train"):

        super().__init__()
        self.album_data_df = self.retrieve_tabular_album_data()
        self.lyrics_embeddings_df = self.retrieve_lyrics_embeddings()

        self.FantAIno_df = self.album_data_df.merge(self.lyrics_embeddings_df, on=["album_name", "artist_name"], how="left")
        print(f"NOTE: The number of rows with missing data: {(self.FantAIno_df.isnull().any(axis=1)).sum()} out of {self.FantAIno_df.shape[0]}.")
        self.FantAIno_df.fillna(0, inplace=True)

        self.FantAIno_df_X = self.FantAIno_df.drop(["rating"], axis=1)
        self.FantAIno_df_y = self.FantAIno_df["rating"]
        self.FantAIno_df_X_train, self.FantAIno_df_X_test, self.FantAIno_df_y_train, self.FantAIno_df_y_test  = train_test_split(self.FantAIno_df_X, self.FantAIno_df_y, test_size=0.2, random_state=self.seed)


        if split == "train":
            self.FantAIno_df_X = self.FantAIno_df_X_train
            self.FantAIno_df_y = self.FantAIno_df_y_train
        elif split == "test":
            self.FantAIno_df_X = self.FantAIno_df_X_test
            self.FantAIno_df_y = self.FantAIno_df_y_test
        else:
            raise ValueError(f"Invalid split: {split}")


