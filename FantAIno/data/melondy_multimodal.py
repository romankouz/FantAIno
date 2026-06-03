import datachain as dc 
from datachain import C

from datasets import Dataset, Features, Image, Value, Sequence, load_dataset

import numpy as np

from pydantic import BaseModel

import tensorflow as tf
import torch

from FantAIno.constants import (
    S3_GENERAL_PURPOSE_BUCKET_NAME,
    S3_ALBUM_DATA_TABLE_BUCKET_NAME,
    S3_EMBEDDINGS_TABLE_BUCKET_NAME,
    S3_EMBEDDINGS_DATABASE_NAME,
    S3_ALBUM_DATA_DATABASE_NAME,
    S3_SMALL_EMBEDDINGS_TABLE,
    S3_LARGE_EMBEDDINGS_TABLE,
    S3_ALBUM_DATA_TABLE
)
from FantAIno.data.melondy_base import MelondyBaseDataset
from FantAIno.data.melondy_classic import MelondyClassicDataset
from FantAIno.preprocessing.preprocessing import PreprocessingPipeline
from FantAIno.utils.data_utils import get_secret

class MelondyFragmentedMultimodalDataset(MelondyBaseDataset):
    """
    FantAIno dataset designed for classical machine learning algorithms.

    This dataset is designed for merging outputs from various models, some that can work
    with images and raw text with others that work on organized, tabular data.
    """

    def __init__(
        self,
        split: str = "train",
        construct_dataset: bool = False,
        data_type: str = "numpy",
        drop_cols: list[str] | None = None,
        album_data_preprocessing_pipeline: PreprocessingPipeline | None = None,
        lyrics_preprocessing_pipeline: PreprocessingPipeline | None = None,    
        album_art_preprocessing_pipeline: PreprocessingPipeline | None = None,
    ):
        super().__init__()

        if construct_dataset:
            self.construct_dataset(
                album_data_preprocessing_pipeline=album_data_preprocessing_pipeline,
                lyrics_preprocessing_pipeline=lyrics_preprocessing_pipeline,
                album_art_preprocessing_pipeline=album_art_preprocessing_pipeline,
            )
        else:
            self.load_dataset()

    def construct_dataset(
        self,
        album_data_preprocessing_pipeline: PreprocessingPipeline | None = None,
        lyrics_preprocessing_pipeline: PreprocessingPipeline | None = None,    
        album_art_preprocessing_pipeline: PreprocessingPipeline | None = None,
    ):

        album_data_pd = self.retrieve_tabular_album_data()
        if album_data_preprocessing_pipeline:
            self.album_data_df = album_data_preprocessing_pipeline(album_data_pd)
        self.album_data_df = dc.read_pandas(album_data_pd)

        self.lyrics_df = self.retrieve_lyrics_data()
        if lyrics_preprocessing_pipeline:
            self.lyrics_df = lyrics_preprocessing_pipeline(self.lyrics_df)

        self.image_df = self.retrieve_album_art_data()
        if album_art_preprocessing_pipeline:
            self.image_df = album_art_preprocessing_pipeline(self.image_df)

        image_df_with_join_key = self.image_df.map(id=lambda file: file.path.split("\\")[-1].rsplit(".", 1)[0], params=["file"], output=str)
        lyrics_df_with_join_key = self.lyrics_df.map(id=lambda artist_name, album_name: f"{artist_name}___{album_name}", params=["jsonl.artist", "jsonl.album"])
        album_data_df_with_join_key = self.album_data_df.map(id=lambda artist_name, album_name: f"{artist_name}___{album_name}", params=["artist_name", "album_name"])

        self.multi_modal_df = (
            image_df_with_join_key
            .merge(
                lyrics_df_with_join_key,
                on=dc.C.id,
                inner=False,
                full=False
            )
            .merge(
                album_data_df_with_join_key,
                on=dc.C.id,
                inner=False,
                full=False
            )
            .persist()
        )

        self.train_records = []
        self.test_records = []

        for row in self.multi_modal_df:
            # --- Pick out the pieces ---
            file_obj    = row[0]   # ImageFile object (the album cover)
            album_id    = row[2]
            lyrics_row  = row[3]   # This is the LyricsRow with the tracks dict
            artist      = row[5]
            album       = row[6]
            genre_list  = row[7]
            rating      = row[8]
            total_tracks = row[9]
            num_markets = row[10]
            release_year = row[11]
            release_month = row[12]
            release_day = row[13]
            duration    = row[14]
            explicit    = row[15]
            featured    = row[16]
            num_features = row[17]
            track_names = row[18]
            popularity  = row[19]

            # --- Read the image bytes from S3 ---
            try:
                img_bytes = file_obj.read()
            except AttributeError:
                # If .read() doesn't exist, try getting a local path first
                local_path = file_obj.get_local_path()
                with open(local_path, "rb") as f:
                    img_bytes = f.read()

            # --- Convert the tracks dict to a list (way better for ML) ---
            # lyrics_row.tracks looks like: {"My World (Intro)": "[Intro]...", "Doja": "..."}
            tracks_list = []
            if lyrics_row is not None and hasattr(lyrics_row, "tracks") and lyrics_row.tracks is not None:
                for track_name, lyrics in lyrics_row.tracks.items():
                    tracks_list.append({
                        "name": track_name,
                        "lyrics": lyrics
                    })
            # If lyrics_row is None or has no tracks, tracks_list remains empty.
        

            # --- Helper to fix NaN values ---
            def clean_num(val, default=0.0):
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return default
                return val

            corresponding_dataset = MelondyClassicDataset(split="train")
            training_albums = [x[0] for x in corresponding_dataset.FantAIno_index]

            # --- Build the record ---
            album_dict = {
                "image": img_bytes,
                "album_id": album_id,
                "artist": artist,
                "album": album,
                "tracks": tracks_list,
                "genre_list": list(genre_list) if genre_list else [],
                "rating": int(clean_num(rating, 0)),
                "total_tracks": clean_num(total_tracks),
                "num_available_markets": clean_num(num_markets),
                "release_year": clean_num(release_year),
                "release_month": clean_num(release_month),
                "release_day": clean_num(release_day),
                "album_duration_in_s": clean_num(duration),
                "explicit_proportion": clean_num(explicit),
                "featured_artists": list(featured) if featured else [],
                "num_features": clean_num(num_features),
                "track_names": list(track_names) if track_names else [],
                "artist_popularity": clean_num(popularity),
            }
            if album in training_albums:
                self.train_records.append(album_dict)
            else:
                self.test_records.append(album_dict)

        features = Features({
            "image": Image(),
            "album_id": Value("string"),
            "artist": Value("string"),
            "album": Value("string"),
            "tracks": [{
                "name": Value("string"),
                "lyrics": Value("string")
            }],
            "genre_list": Sequence(Value("string")),
            "rating": Value("int8"),
            "total_tracks": Value("float32"),
            "num_available_markets": Value("float32"),
            "release_year": Value("float32"),
            "release_month": Value("float32"),
            "release_day": Value("float32"),
            "album_duration_in_s": Value("float32"),
            "explicit_proportion": Value("float32"),
            "featured_artists": Sequence(Value("string")),
            "num_features": Value("float32"),
            "track_names": Sequence(Value("string")),
            "artist_popularity": Value("float32"),
        })

        print(f"✅ Processed {len(self.train_records)} training albums!")
        print(f"✅ Processed {len(self.test_records)} test albums!")

        # create huggingface dataset
        train_dataset = Dataset.from_list(self.train_records, features=features)
        train_dataset.push_to_hub("roromaniac/FantAIno", split="train")
        test_dataset = Dataset.from_list(self.test_records, features=features)
        test_dataset.push_to_hub("roromaniac/FantAIno", split="test")


    def load_dataset(self, split="train", data_type="numpy"):

        loaded_dataset = load_dataset("roromaniac/FantAIno", split=split) 
        self.FantAIno_df_X = loaded_dataset.remove_columns(["rating"])
        self.FantAIno_df_y = loaded_dataset["rating"]
        match data_type:
            case "numpy":
                self.FantAIno_df_X = self.FantAIno_df_X.to_numpy()
                self.FantAIno_df_y = self.FantAIno_df_y.to_numpy()
            case "pytorch":
                self.FantAIno_df_X = torch.from_numpy(self.FantAIno_df_X.to_numpy()).float()
                self.FantAIno_df_y = torch.from_numpy(self.FantAIno_df_y.to_numpy()).float().reshape(-1, 1)
            case "tensorflow":
                self.FantAIno_df_X = tf.convert_to_tensor(self.FantAIno_df_X.to_numpy())
                self.FantAIno_df_y = tf.convert_to_tensor(self.FantAIno_df_y.to_numpy()).reshape(-1, 1)
            case _:
                raise ValueError(f"The data type {data_type} is not supported.")
