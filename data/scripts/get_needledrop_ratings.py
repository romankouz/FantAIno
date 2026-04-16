import boto3
from dotenv import load_dotenv
import json
import logging
import os

from playwright.sync_api import sync_playwright

from FantAIno.constants import MELONDY_URL, S3_GENERAL_PURPOSE_BUCKET_NAME
from FantAIno.utils.data_utils import clean_name, sanitize_filename, embeddings_from_lyrics_obj
from FantAIno.utils.genius_utils import get_album_lyrics
from FantAIno.utils.spotify_utils import get_spotify_album, process_spotify_album_data
from FantAIno.utils.s3_utils import (
    process_image_s3, 
    process_lyrics_s3,
    retrieve_s3_table_catalog,
    create_s3_embeddings_schema,
    create_s3_album_data_schema,
    pyiceberg_record_exists,
    pyiceberg_insert_embeddings_record,
    pyiceberg_insert_album_data_record
)
from FantAIno.utils.logging import create_logger

load_dotenv()

def get_numerical_rating(rating: str) -> int:
    """
        Convert's Fantano's English ratings to numerical ones.

        - Classic: 10
        - Not good: -1
        - Otherwise, return the numerical rating.
        - If the rating is not convertible, just return -2.
    """
    if rating.lower() == "classic":
        return 10
    elif rating.lower() == "not good":
        return -1
    else:
        try:
            return int(rating)
        except ValueError:
            return -2

def scraper(page):
    """
        Scrapes the Melondy website for album reviews and extracts the artist, album, genre, image, and rating.

        Args:
            page: Playwright page object.

        Returns:
            A set of tuples, each containing the artist, album, genre, image, and rating of a Melondy album entry.
    """

    try:

        reviews = page.locator("div.border-black").all()

        # create an empty set to store extracted album tuples
        snapshot_album_data = set()

        for review in reviews:

            data = {}

            # ------------------------------------------------------------
            # EXTRACT ALBUM IMAGE
            # ------------------------------------------------------------
            image_source = review.locator("img").first

            # get_attribute("src") returns the image URL
            data["image"] = image_source.get_attribute("src")

            # ------------------------------------------------------------
            # EXTRACT TITLE ROW TEXT
            # ------------------------------------------------------------
            album_row = review.locator("div.truncate.flex.gap-2.font-normal.text-base.tracking-tight")
            artist_album_text = album_row.inner_text().strip()

            if "–" in artist_album_text:
                artist_name, album_name = artist_album_text.split("–", 1)
                data["artist"] = artist_name.strip()
                data["album"] = album_name.strip()
            else:
                # fallback in case parsing fails
                data["artist"] = None
                data["album"] = artist_album_text.strip()

            # ------------------------------------------------------------
            # EXTRACT GENRES
            # ------------------------------------------------------------
            try:
                genre_text = review.locator("div.flex-1.truncate.text-gray-500")
                data["genre"] = genre_text.inner_text().split(', ')
            except Exception:
                logger.error("%s's %s had an issue with extracting genres.", data["artist"], data["album"])
                data["genre"] = []

            # ------------------------------------------------------------
            # EXTRACT RATING
            # ------------------------------------------------------------
            # the rating is the bold uppercase text like "3/10"
            rating_source = review.locator("div.uppercase.font-bold.text-gray-900").first
            raw_rating = rating_source.inner_text().strip()

            rating_value = raw_rating.split("/")[0].strip()

            data["rating"] = get_numerical_rating(rating_value)

            if data["rating"] == -2:
                continue

            snapshot_album_data.add(
                (
                    data["artist"],
                    data["album"],
                    str(data["genre"]),
                    data["image"],
                    data["rating"],
                )
            )

        # return the set of extracted review data
        return snapshot_album_data

    except Exception as e:
        return set()

with sync_playwright() as p:

    # handle embeddings catalog
    fantaino_lyrics_embeddings_catalog = retrieve_s3_table_catalog(
        catalog_name=os.getenv("S3_EMBEDDINGS_TABLE_BUCKET_NAME"),
        account_id=os.getenv("AWS_ACCOUNT_ID"),
        s3tablebucketname=os.getenv("S3_EMBEDDINGS_TABLE_BUCKET_NAME"),
        region=os.getenv("PYICEBERG_AWS_DEFAULT_REGION"),
    )

    # handle album data catalog
    fantaino_album_data_catalog = retrieve_s3_table_catalog(
        catalog_name=os.getenv("S3_ALBUM_DATA_TABLE_BUCKET_NAME"),
        account_id=os.getenv("AWS_ACCOUNT_ID"),
        s3tablebucketname=os.getenv("S3_ALBUM_DATA_TABLE_BUCKET_NAME"),
        region=os.getenv("PYICEBERG_AWS_DEFAULT_REGION"),
    )

    # create lyrics embeddings tables if they don't exist
    lyrics_embeddings_small_schema = create_s3_embeddings_schema(embeddings_dim=1536)
    lyrics_embeddings_large_schema = create_s3_embeddings_schema(embeddings_dim=3072)
    if not fantaino_lyrics_embeddings_catalog.table_exists(f"{os.getenv("S3_EMBEDDINGS_DATABASE_NAME")}.lyrics_embeddings_small"):
        fantaino_lyrics_embeddings_catalog.create_table(
            identifier=f"{os.getenv("S3_EMBEDDINGS_DATABASE_NAME")}.lyrics_embeddings_small",
            schema=lyrics_embeddings_small_schema
        )
    fantaino_lyrics_embeddings_small_table = fantaino_lyrics_embeddings_catalog.load_table(
        f"{os.getenv("S3_EMBEDDINGS_DATABASE_NAME")}.lyrics_embeddings_small"
    )
    if not fantaino_lyrics_embeddings_catalog.table_exists(f"{os.getenv("S3_EMBEDDINGS_DATABASE_NAME")}.lyrics_embeddings_large"):
        fantaino_lyrics_embeddings_catalog.create_table(
            identifier=f"{os.getenv("S3_EMBEDDINGS_DATABASE_NAME")}.lyrics_embeddings_large",
            schema=lyrics_embeddings_large_schema
        )
    fantaino_lyrics_embeddings_large_table = fantaino_lyrics_embeddings_catalog.load_table(
        f"{os.getenv("S3_EMBEDDINGS_DATABASE_NAME")}.lyrics_embeddings_large"
    )

    # create album data table if it does not exist
    if not fantaino_album_data_catalog.table_exists(f"{os.getenv("S3_ALBUM_DATA_DATABASE_NAME")}.album_data"):
        fantaino_album_data_catalog.create_table(
            identifier=f"{os.getenv("S3_ALBUM_DATA_DATABASE_NAME")}.album_data",
            schema=create_s3_album_data_schema()
        )
    fantaino_album_data_table = fantaino_album_data_catalog.load_table(
        f"{os.getenv("S3_ALBUM_DATA_DATABASE_NAME")}.album_data"
    )

    # load our s3 client to check if album is processed or not
    s3_client = boto3.client("s3")

    # start a browser instance (headless = True doesn't open the local window)
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # open the target page
    page.goto(MELONDY_URL)

    # select the LATEST REVIEWS option on Melondy
    page.get_by_text("LATEST REVIEWS").click()

    # scraper settings
    stale_rounds = 0
    MAX_STALE_ROUNDS = 10

    # Define how many pixels to scroll down each time
    SCROLL_STEP = 125

    # setup logging
    logger = create_logger("fantaino_data_pull", "get_needledrop_ratings.log", logging.DEBUG)

    recorded_albums = set()

    while True:

        # record failed album processes
        failed_albums = set()

        # Extract data from the currently visible content
        current_album_data = scraper(page)
        new_albums = current_album_data - recorded_albums

        # Scroll down by the defined step from the current position
        page.evaluate(f"window.scrollBy(0, {SCROLL_STEP});")

        # Wait for the page to load any new content after scrolling
        page.wait_for_timeout(500)

        if len(new_albums) == 0:
            stale_rounds += 1
            if stale_rounds >= MAX_STALE_ROUNDS:
                break
        else:
            stale_rounds = 0

        # check if album is processed or not
        for album_tuple in new_albums:

            artist, album, genre, image_url, rating = album_tuple

            # processing flags
            album_art_s3_exists, lyrics_s3_exists, album_data_s3_exists = False, False, False

            artist = clean_name(artist)
            album = clean_name(album)
            if (artist, album) in failed_albums:
                continue

            # check if album art is processed
            try:
                _, extension = os.path.splitext(image_url)
                album_image_filename = sanitize_filename(f"{artist}___{album}{extension}")
                s3_client.get_object(
                    Bucket=S3_GENERAL_PURPOSE_BUCKET_NAME, 
                    Key=os.path.join("album_art", album_image_filename)
                )
                album_art_s3_exists = True
            except s3_client.exceptions.NoSuchKey as e:
                try:
                    process_image_s3(s3_client, artist, album, image_url)
                    logger.info("%s's %s album cover art successfully processed!", artist, album)
                except Exception as e_inner:
                    logger.error("%s's %s had an issue with uploading album art.", artist, album)
                    logger.error("%s", repr(e_inner))
                    failed_albums.add((artist, album))

            # check if lyrics are processed
            lyrics_obj = {}
            try:
                lyrics_filename = sanitize_filename(f"{artist}___{album}.jsonl")
                lyrics_response = s3_client.get_object(
                    Bucket=S3_GENERAL_PURPOSE_BUCKET_NAME, 
                    Key=os.path.join("lyrics", lyrics_filename)
                )
                lyrics_obj = json.load(lyrics_response['Body'])
                lyrics_s3_exists = True
            except s3_client.exceptions.NoSuchKey as e:
                try:
                    lyrics_obj = get_album_lyrics(artist, album)
                    if lyrics_obj:
                        process_lyrics_s3(s3_client, artist, album, lyrics_obj)
                        logger.info("%s's %s lyrics successfully processed!", artist, album)
                except Exception as e_inner:
                    logger.error("%s's %s had an issue with retrieving lyrics.", artist, album)
                    logger.error("%s", repr(e_inner))
                    failed_albums.add((artist, album))

            # check if album lyrics embeddings are processed
            if lyrics_obj:
                try:

                    album_embeddings_small_exists = pyiceberg_record_exists(
                        fantaino_lyrics_embeddings_small_table,
                        artist_name=artist,
                        album_name=album
                    )

                    if not album_embeddings_small_exists:
                        album_lyrics_embeddings_small = embeddings_from_lyrics_obj(lyrics_obj, embeddings_model="text-embedding-3-small")
                        pyiceberg_insert_embeddings_record(
                            pyiceberg_table=fantaino_lyrics_embeddings_small_table,
                            artist_name=artist,
                            album_name=album,
                            embeddings=album_lyrics_embeddings_small,
                            embeddings_schema=lyrics_embeddings_small_schema
                        )
                    else:
                        logger.info("%s's %s album lyrics embeddings (small) already exists.", artist, album)
                    
                    album_embeddings_large_exists = pyiceberg_record_exists(
                        fantaino_lyrics_embeddings_large_table,
                        artist_name=artist,
                        album_name=album
                    )
                    if not album_embeddings_large_exists:
                        album_lyrics_embeddings_large = embeddings_from_lyrics_obj(lyrics_obj, embeddings_model="text-embedding-3-large")
                        pyiceberg_insert_embeddings_record(
                            pyiceberg_table=fantaino_lyrics_embeddings_large_table,
                            artist_name=artist,
                            album_name=album,
                            embeddings=album_lyrics_embeddings_large,
                            embeddings_schema=lyrics_embeddings_large_schema
                        )
                    else:
                        logger.info("%s's %s album lyrics embeddings (large) already exists.", artist, album)

                    if not (
                        album_embeddings_small_exists or 
                        album_embeddings_large_exists
                    ):
                        logger.info("%s's %s album lyrics embeddings (small and large) successfully processed!", artist, album)
                except Exception as e:
                    logger.error("%s's %s had an issue with uploading album lyrics embeddings.", artist, album)
                    logger.error("%s", repr(e))
                    failed_albums.add((artist, album))


            # check if album data is processed
            album_data_filename = sanitize_filename(f"{artist}___{album}.json")
            album_data_obj = {}
            try:
                album_data_response = s3_client.get_object(
                    Bucket=S3_GENERAL_PURPOSE_BUCKET_NAME, 
                    Key=os.path.join("album_data", album_data_filename)
                )
                album_data_obj = json.load(album_data_response['Body'])
                album_data_s3_exists = True
            except s3_client.exceptions.NoSuchKey as e:
                album_data_obj = {
                    "artist": artist,
                    "album": album,
                    "genre": genre,
                    "rating": rating,
                }
                try:
                    s3_client.put_object(
                        Body=json.dumps(album_data_obj).encode("utf-8"),
                        Bucket=S3_GENERAL_PURPOSE_BUCKET_NAME,
                        Key=os.path.join("album_data", album_data_filename)
                    )
                    logger.info("%s's %s album data successfully processed to general purpose bucket!", artist, album)
                except Exception as e_inner:
                    logger.error("%s's %s had an issue with uploading album data to general purpose bucket.", artist, album)
                    logger.error("%s", repr(e_inner))
                    failed_albums.add((artist, album))

            if album_data_obj:
                try:
                    album_data_exists = pyiceberg_record_exists(
                        fantaino_album_data_table,
                        artist_name=artist,
                        album_name=album
                    )
                    if not album_data_exists:
                        spotify_data = get_spotify_album(artist, album)
                        output = (
                            total_tracks,
                            num_available_markets,
                            release_year,
                            release_month,
                            release_day,
                            album_duration_in_s,
                            explicit_proportion,
                            featured_artists,
                            num_features,
                            track_names,
                            artist_popularity,
                        ) = process_spotify_album_data(spotify_data)

                        album_data = [
                            artist,
                            album,
                            album_data_obj["genre"],
                            album_data_obj["rating"],
                            total_tracks,
                            num_available_markets,
                            release_year,
                            release_month,
                            release_day,
                            album_duration_in_s,
                            explicit_proportion,
                            featured_artists,
                            num_features,
                            track_names,
                            artist_popularity,
                        ]
                    
                        pyiceberg_insert_album_data_record(
                            pyiceberg_table=fantaino_album_data_table,
                            album_data=album_data,
                            album_data_schema=create_s3_album_data_schema()
                        )
                    else:
                        logger.info("%s's %s album data already exists.", artist, album)
                    logger.info("%s's %s album data successfully processed to S3Table bucket!", artist, album)
                except Exception as e:
                    logger.error("%s's %s had an issue with uploading album data to S3Table bucket.", artist, album)
                    logger.error("%s", repr(e))
                    failed_albums.add((artist, album))
            
            if (album_art_s3_exists and lyrics_s3_exists and album_data_s3_exists):
                logger.info("%s's album: %s already exists in S3.", artist, album)
            recorded_albums.add(album_tuple)

        with open(os.path.join("logs", "failed_albums.json"), "a", encoding="utf-8") as f:
            for tup in failed_albums:
                f.write(f"{tup}\n")

    # close the browser
    browser.close()
