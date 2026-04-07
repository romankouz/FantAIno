from dotenv import load_dotenv
load_dotenv()
import json
import jsonlines
import logging
import os
import pandas as pd

from lyricsgenius import Genius

from utils.logging import create_logger

GENIUS_TOKEN = os.getenv("GENIUS_CLIENT_ACCESS_TOKEN")
genius = Genius(GENIUS_TOKEN)
logger = create_logger("genius_utils", "extract_lyrics.log", logging.DEBUG)

def get_album_lyrics(artist_name: str, album_name: str) -> dict | None:
    # get album
    try:
        album = genius.search_album(
            name=album_name,
            artist=artist_name
        )
        if album is None:
            logger.warning("Album %s by %s not found", album_name, artist_name)
            return
    except Exception as e:
        logger.error("Error finding album %s: %s.", album_name, repr(e))
        return

    # save lyrics
    album_obj = {
        "artist": artist_name,
        "album": album_name,
        "tracks": {}
    }
    for track in album.to_dict()["tracks"]:
        album_obj["tracks"][track["song"]["title"]] = track["song"]["lyrics"]

    return album_obj