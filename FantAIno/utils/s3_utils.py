import json
import os
import requests

# from io import BytesIO
# from PIL import Image
import pyarrow as pa
import pyiceberg
from pyiceberg.catalog import load_catalog
from pyiceberg.expressions import EqualTo, And

from FantAIno.constants import S3_GENERAL_PURPOSE_BUCKET_NAME
from FantAIno.utils.data_utils import sanitize_filename

def process_image_s3(s3_client, artist_name, album_name, original_image_path):
    """
        Processes an album image from melondy.com to upload [artist_name]___[album_name].jpg to AWS S3 bucket.

        Args:
            s3_client (boto3.client): The S3 client to use to upload the image to the bucket.
            artist_name (str): The name of the artist of the album.
            album_name (str): The name of the album.
            original_image_path (str): The URL to the stored image of the album. Usually a cloudfront URL.
    """

    try:
        if original_image_path is not None:
            _, extension = os.path.splitext(original_image_path)
            response = requests.get(original_image_path, timeout=10)
            album_image_filename = sanitize_filename(f"{artist_name}___{album_name}{extension}")
            
            s3_client.put_object(
                Body=response.content,
                Bucket=S3_GENERAL_PURPOSE_BUCKET_NAME,
                Key=os.path.join("album_art", album_image_filename)
            )
    except ConnectionError as e:
        print(f"{artist_name}'s {album_name} had an issue with retrieving album cover.")
        print(e)
    except Exception as e:
        print(f"{artist_name}'s {album_name} had an issue with uploading album cover to AWS S3 bucket.")
        print(e)

def process_lyrics_s3(s3_client, artist_name, album_name, lyrics):

    try:
        lyrics_filename = sanitize_filename(f"{artist_name}___{album_name}.jsonl")
        s3_client.put_object(
            Body=json.dumps(lyrics).encode("utf-8"),
            Bucket=S3_GENERAL_PURPOSE_BUCKET_NAME,
            Key=os.path.join("lyrics", lyrics_filename)
        )
    except Exception as e:
        print(f"{artist_name}'s {album_name} had an issue with uploading lyrics to AWS S3 bucket.")
        print(e)

def retrieve_s3_table_catalog(catalog_name: str, account_id: str, s3tablebucketname: str, region: str):
    """
    Initialize and return the REST catalog.
    """
    return load_catalog(
        catalog_name,
        **{
            "type": "rest",
            "warehouse": f"{account_id}:s3tablescatalog/{s3tablebucketname}",
            "uri": f"https://glue.{region}.amazonaws.com/iceberg",
            "rest.sigv4-enabled": "true",
            "rest.signing-name": "glue",
            "rest.signing-region": region,
            # Tuning commit retries to reduce chance of CommitFailedException
            "commit.retry.num-retries": "20",                           # more attempts
            "commit.retry.min-wait-ms": "500",                         # longer min wait between retries
            "commit.retry.max-wait-ms": "120000",                      # 2m max wait per retry
            "commit.retry.total-timeout-ms": "3600000",                # 1hr total retry window
            "commit.status-check.num-retries": "10",                   # more status check attempts
            "commit.status-check.min-wait-ms": "2000",                 # longer min wait for status
            "commit.status-check.max-wait-ms": "120000",               # 2m max wait per status check
            "commit.status-check.total-timeout-ms": "3600000",         # 1hr status check window
        }
    )

def create_s3_embeddings_schema(embeddings_dim: int = 1536) -> pa.Schema:
    
    embeddings_schema = pa.schema(
        [
            pa.field("artist_name", pa.string()),
            pa.field("album_name", pa.string()),
        ]
        + [pa.field(f"embedding_dim_{i}", pa.float64()) for i in range(embeddings_dim)]
    )
    return embeddings_schema

def create_s3_album_data_schema() -> pa.Schema:

    album_data_schema = pa.schema(
        [
            pa.field("artist_name", pa.string()),
            pa.field("album_name", pa.string()),
            pa.field("genre_list", pa.list_(pa.string())),
            pa.field("rating", pa.int64()),
            pa.field("total_tracks", pa.int64()),
            pa.field("num_available_markets", pa.int64()),
            pa.field("release_year", pa.int64()),
            pa.field("release_month", pa.int64()),
            pa.field("release_day", pa.int64()),    
            pa.field("album_duration_in_s", pa.float64()),
            pa.field("explicit_proportion", pa.float64()),
            pa.field("featured_artists", pa.list_(pa.string())),
            pa.field("num_features", pa.int64()),
            pa.field("track_names", pa.list_(pa.string())),
            pa.field("artist_popularity", pa.int64()),
        ]
    )

    return album_data_schema

def update_s3_album_data_genre_dummies():
    """Recomputes the genre dummies for all albums in the FantAIno dataset."""
    pass


def pyiceberg_record_exists(
    pyiceberg_table: pyiceberg.table.Table,
    artist_name: str,
    album_name: str
) -> bool:
    """
    Checks if a record exists in the pyiceberg table.
    """
    scan = pyiceberg_table.scan(
        row_filter=And(EqualTo("artist_name", artist_name), EqualTo("album_name", album_name)),
        selected_fields=["artist_name", "album_name"]
    )
    record_exists = scan.to_arrow().num_rows > 0
    return record_exists

def pyiceberg_insert_embeddings_record(
    pyiceberg_table: pyiceberg.table.Table,
    artist_name: str,
    album_name: str,
    embeddings: list[float],
    embeddings_schema: pa.Schema
) -> None:

    embeddings_entry = {k: v for k, v in zip(embeddings_schema.names, [artist_name, album_name] + embeddings)}
    album_embeddings_data = pa.Table.from_pylist([embeddings_entry], schema=embeddings_schema)
    pyiceberg_table.append(album_embeddings_data)

def pyiceberg_insert_album_data_record(
    pyiceberg_table: pyiceberg.table.Table,
    album_data: list,
    album_data_schema: pa.Schema
) -> None:

    album_data_entry = {k: v for k, v in zip(album_data_schema.names, album_data)}
    album_data_table = pa.Table.from_pylist([album_data_entry], schema=album_data_schema)
    pyiceberg_table.append(album_data_table)