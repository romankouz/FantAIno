import boto3
from dotenv import load_dotenv
import json
import logging
import os

from playwright.sync_api import sync_playwright

from constants import MELONDY_URL, S3_BUCKET_NAME
from utils.data_utils import clean_name
from utils.genius_utils import get_album_lyrics
from utils.s3_utils import process_image_s3, process_lyrics_s3
from utils.logging import create_logger

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
                genre_text = review.locator("div.text-gray-500.dark:text-gray-400")
                data["genre"] = genre_text.inner_text().split(', ')
            except Exception:
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
    SCROLL_STEP = 250

    # setup logging
    logger = create_logger("fantaino_data_pull", "get_needledrop_ratings.log", logging.DEBUG)

    while True:

        # record album data
        current_album_data = set()

        # Extract data from the currently visible content
        current_album_data.update(scraper(page))

        # Scroll down by the defined step from the current position
        page.evaluate(f"window.scrollBy(0, {SCROLL_STEP});")

        # Wait for the page to load any new content after scrolling
        page.wait_for_timeout(500)

        if len(current_album_data) == 0:
            stale_rounds += 1
            if stale_rounds >= MAX_STALE_ROUNDS:
                break
        else:
            stale_rounds = 0

        # check if album is processed or not
        for artist, album, genre, image_url, rating in current_album_data:

            # processing flags
            album_art_S3_exists, lyrics_S3_exists, album_data_S3_exists = False, False, False

            artist = clean_name(artist)
            album = clean_name(album)
            # check if album art is processed
            try:
                s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=os.path.join("album_art", f"{artist}___{album}.jpg"))
                album_art_S3_exists = True
            except s3_client.exceptions.NoSuchKey as e:
                try:
                    process_image_s3(s3_client, artist, album, image_url)
                    logger.info("%s's %s album cover art successfully processed!", artist, album)
                except Exception as e_inner:
                    logger.error("%s's %s had an issue with uploading album art.", artist, album)
                    logger.error("%s", repr(e_inner))

            # check if lyrics are processed
            try:
                s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=os.path.join("lyrics", f"{artist}___{album}.jsonl"))
                lyrics_S3_exists = True
            except s3_client.exceptions.NoSuchKey as e:
                try:
                    lyrics = get_album_lyrics(artist, album)
                    if lyrics:
                        process_lyrics_s3(s3_client, artist, album, lyrics)
                        logger.info("%s's %s lyrics successfully processed!", artist, album)
                except Exception as e_inner:
                    logger.error("%s's %s had an issue with retrieving lyrics.", artist, album)
                    logger.error("%s", repr(e_inner))

            # check if album data is processed
            try:
                s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=os.path.join("album_data", f"{artist}___{album}.json"))
                album_data_S3_exists = True
            except s3_client.exceptions.NoSuchKey as e:
                album_dict = {
                    "artist": artist,
                    "album": album,
                    "genre": genre,
                    "rating": rating,
                }
                try:
                    s3_client.put_object(
                        Body=json.dumps(album_dict).encode("utf-8"),
                        Bucket=S3_BUCKET_NAME,
                        Key=os.path.join("album_data", f"{artist}___{album}.json")
                    )
                    logger.info("%s's %s album data successfully processed!", artist, album)
                except Exception as e_inner:
                    logger.error("%s's %s had an issue with uploading album data.", artist, album)
                    logger.error("%s", repr(e_inner))
            
            if (album_art_S3_exists and lyrics_S3_exists and album_data_S3_exists):
                logger.info("%s's album: %s already exists in S3.", artist, album)

    # close the browser
    browser.close()
