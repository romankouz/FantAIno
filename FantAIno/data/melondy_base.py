import boto3
import datachain as dc

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
from FantAIno.utils.data_utils import convert_to_list, get_secret
from FantAIno.utils.s3_utils import retrieve_s3_table_catalog

class MelondyBaseDataset():
    """
    Base class for retrieving data for the FantAIno project.

    All other data classes will construct datasets leveraging these data loading methods.
    """

    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.seed = 888
        

    def retrieve_tabular_album_data(self):

        album_data_s3tablebucketname = S3_ALBUM_DATA_TABLE_BUCKET_NAME
        album_data_catalog_name = album_data_s3tablebucketname

        album_data_catalog = retrieve_s3_table_catalog(
            catalog_name=album_data_catalog_name,
            account_id=get_secret("AWS_ACCOUNT_ID"),
            s3tablebucketname=album_data_s3tablebucketname,
            region=get_secret("PYICEBERG_AWS_DEFAULT_REGION"),
        )

        album_data_table = album_data_catalog.load_table(f"{S3_ALBUM_DATA_DATABASE_NAME}.{S3_ALBUM_DATA_TABLE}")

        all_records_arrow = album_data_table.scan().to_arrow()
        album_data_pd = all_records_arrow.to_pandas()

        album_data_pd["genre_list"] = album_data_pd["genre_list"].apply(convert_to_list)
        album_data_pd["featured_artists"] = album_data_pd["featured_artists"].apply(convert_to_list)
        album_data_pd["track_names"] = album_data_pd["track_names"].apply(convert_to_list)

        return album_data_pd

    def retrieve_lyrics_data(self):

        lyrics_df = dc.read_json(
            rf"s3://{S3_GENERAL_PURPOSE_BUCKET_NAME}/lyrics\*",
            type="text",
            format="jsonl",
            spec=LyricsRow,
            client_config = {
                "key": get_secret("AWS_ACCESS_KEY_ID"),
                "secret": get_secret("AWS_SECRET_ACCESS_KEY")
            }
        ).persist()

        return lyrics_df

    def retrieve_album_art_data(self):
        
        image_df = dc.read_storage(
            rf"s3://{S3_GENERAL_PURPOSE_BUCKET_NAME}/album_art\*",
            type="image", 
            client_config = {
                "key": get_secret("AWS_ACCESS_KEY_ID"),
                "secret": get_secret("AWS_SECRET_ACCESS_KEY")
            }
        ).map(path=lambda file: file.path, output=str).persist()

        return image_df

    def retrieve_lyrics_embeddings(self, openai_embedding_size: str = "small"):

        embeddings_catalog = retrieve_s3_table_catalog(
            catalog_name=S3_EMBEDDINGS_TABLE_BUCKET_NAME,
            account_id=get_secret("AWS_ACCOUNT_ID"),
            s3tablebucketname=S3_EMBEDDINGS_TABLE_BUCKET_NAME,
            region=get_secret("PYICEBERG_AWS_DEFAULT_REGION"),
        )

        if openai_embedding_size == "small":
            embeddings_table = f"{S3_EMBEDDINGS_DATABASE_NAME}.{S3_SMALL_EMBEDDINGS_TABLE}"
        elif openai_embedding_size == "large":
            embeddings_table = f"{S3_EMBEDDINGS_DATABASE_NAME}.{S3_LARGE_EMBEDDINGS_TABLE}"
        else:
            raise ValueError(f"Invalid OpenAI embedding size: {openai_embedding_size}")

        embeddings_table = embeddings_catalog.load_table(embeddings_table)

        all_records_arrow = embeddings_table.scan().to_arrow()
        embeddings_pd = all_records_arrow.to_pandas()

        return embeddings_pd

class LyricsRow(BaseModel):
    artist: str
    album: str
    tracks: dict