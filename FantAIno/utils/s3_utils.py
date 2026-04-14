import json
import os
import requests

# from io import BytesIO
# from PIL import Image

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
