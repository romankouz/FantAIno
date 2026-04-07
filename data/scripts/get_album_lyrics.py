from dotenv import load_dotenv
load_dotenv()
import json
import jsonlines
import logging
import os
import pandas as pd

from lyricsgenius import Genius

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

logging.getLogger("lyricsgenius").setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logging.basicConfig(filename='extract_lyrics.log', encoding='utf-8', level=logging.DEBUG)

GENIUS_TOKEN = os.getenv("GENIUS_CLIENT_ACCESS_TOKEN")
genius = Genius(GENIUS_TOKEN)

if not os.path.isfile("lyrics.jsonl"):
    with open("lyrics.jsonl", "w", encoding="utf-8") as f:
        pass  # create empty file

# don't use API credits on processed albums
cached_lyrics = set()
with open('lyrics.jsonl', mode='r', encoding='utf-8') as cached_lyrics_reader:
    for line in cached_lyrics_reader:
        if json.loads(line) == []:
            break
        else:
            cached_album = json.loads(line)
            cached_lyrics.add((cached_album['artist'], cached_album['album']))

for index, FantAIno_entry in FantAIno_df.iterrows():

    # don't use API credits on processed albums
    if (FantAIno_entry['artist'], FantAIno_entry['album']) in cached_lyrics:
        logger.info(f"Skipping {FantAIno_entry['artist']} - {FantAIno_entry['album']} is already cached.")
        continue

    else:

        # get album
        try:
            album = genius.search_album(
                name=FantAIno_entry['album'],
                artist=FantAIno_entry['artist']
            )
            if album is None:
                logger.warning(f"Album '{FantAIno_entry['album']}' by '{FantAIno_entry['artist']}' not found")
                continue
        except Exception as e:
            logger.error(f"Error finding album {FantAIno_entry['album']}: {repr(e)}.")
            continue
        
        # save lyrics
        album_obj = {
            "artist": FantAIno_entry["artist"],
            "album": FantAIno_entry["album"],
            "tracks": {}
        }
        for track in album.to_dict()["tracks"]:
            album_obj["tracks"][track["song"]["title"]] = track["song"]["lyrics"]
        with jsonlines.open('lyrics.jsonl', mode='a') as writer:
            writer.write(album_obj)
                
    logger.info(f"Finished processing artist & album: {FantAIno_entry['artist']}, {FantAIno_entry['album']}")