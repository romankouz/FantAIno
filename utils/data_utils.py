"""
Helper functions for processing any acquired data of Fantano's reviews
"""

import os
import requests

from PIL import Image
from io import BytesIO

def process_scraped_data(scraped_data: list) -> list:
    """
        An archived function that helped process album reviews straight from theneedledrop.com.
        Given a list of URLs, we only want to scrape the ones that indicated it was one that had the album review
        script and with a numerical score to scrape.
    """
    processed_data = []
    for i in range(len(scraped_data)):
        if scraped_data[i]['url'].endswith("album-review/"):
            processed_data.append(scraped_data[i])
    return processed_data

def sanitize_filename(filename: str) -> str:
    """
        Removes any invalid characters from a filename.
    """
    for banned_char in '<>:"/|?*\\':
        filename = filename.replace(banned_char, "_")
    return filename

def process_image(artist_name, album_name, original_image_path, rating, train=True):
    """
        Processes an album image from melondy.com to be of the torchvision.datasets.ImageFolder
        format, where ratings are directories and [artist_name]___[album_name].jpg is the filename.
    """
    try:
        if original_image_path is not None:
            train_folder = "train" if train else "test"
            _, extension = os.path.splitext(original_image_path)
            response = requests.get(original_image_path)
            img = Image.open(BytesIO(response.content))
            album_image_filename = sanitize_filename(f"{artist_name}___{album_name}{extension}")
            new_file = os.path.join(os.getcwd(), "album_ImageFolder", train_folder, f"{rating}", album_image_filename)
            os.makedirs(os.path.dirname(new_file), exist_ok=True)
            if img.format is 'PNG':
                # and is not RGBA
                if img.mode is not 'RGBA':
                    img = img.convert("RGBA")
            img.save(new_file)
    except ConnectionError as e:
        print(f"{artist_name}'s {album_name} had an issue with retrieving album cover.")
        print(e)

def process_image_series(s, train=True):
    """
        Process all the images in the melondy dataset.
    """
    return process_image(s["artist"], s["album"], s["image_url"], s["rating"], train=train)

def clean_name(name: str):
    """
        Removes any unwanted characters from album names scraped from melondy.com.
    """
    name = name.replace("’", "'") # remove right side smart apostrophes and replace with single quote
    name = name.replace('•', '') # remove bullet points as they don't show up on spotify album titles
    name = name.replace('“', '"') # replace left double quotation mark with regular double quote
    name = name.replace('”', '"') # replace right double quotation mark with regular double quote
    name = name.replace("'", "") # remove any single quotes because spotify uses fuzzy search and we don't want to URL encode unnecessarily
    return name