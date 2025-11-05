import numpy as np
import os
import spotipy
import time
from typing import Any

from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials

from constants import MELONDY_TO_SPOTIFY
from utils.data_utils import clean_name

# Load environment variables from .env file
load_dotenv()

_spotify = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(),
    requests_timeout=20,
    retries=5,
    status_retries=5
)

def get_album_features(track_items: dict) -> list[str]:
    artists = []
    artist_set = set()
    for track in track_items:
        for artist in track['artists']:
            if artist['name'] not in artist_set:
                artist_set.add(artist['name'])
                artists.append(artist['name'])
    return artists

def get_album_data_from_items(items: list[dict], target_album_name: str) -> dict[str, Any] | None:
    """
    Helper function to find a matching album in a list of Spotify album items.
    It first tries to find an exact or partial name match.
    If no specific match, it returns the first album in the list if available.
    """
    for album in items:
        # Clean the Spotify album name for comparison, converting to lowercase
        spotify_album_name_cleaned = clean_name(album["name"]).lower()
        
        # Check if the target album name matches exactly or is contained within the Spotify album name
        if target_album_name.lower() == spotify_album_name_cleaned or \
            target_album_name.lower() in spotify_album_name_cleaned:
            # If a match is found, retrieve all tracks for this album
            tracks = _spotify.album_tracks(album_id=album["id"])
            track_items = tracks['items']
            # Return the album details and its tracks
            return {"album": album, "tracks": track_items}
    
    # If no specific match was found after checking all items,
    # and if there are any albums in the list, return the very first one.
    if len(items) > 0:
        album = items[0]
        tracks = _spotify.album_tracks(album_id=album["id"])
        track_items = tracks['items']
        return {"album": album, "tracks": track_items}
    
    # If no albums were found at all in the list, return None
    return None

def get_spotify_artist_popularity(artist_name: str):

    cleaned_artist_name = clean_name(artist_name)
    # try getting the artist
    artist = get_spotify_artist(cleaned_artist_name)
    if not artist:
        # if you fail, try splitting the ampersan
        if "&" in artist_name:
            artists = artist_name.split("&")
            for artist in artists:
                solo_artist = get_spotify_artist(artist.strip())
                if solo_artist:
                    artist = solo_artist
                    break
    
    if artist:
        spotify_popularity = artist['popularity']
        return spotify_popularity

def get_spotify_artist(artist_name: str):

    cleaned_artist_name = clean_name(artist_name)
    results = _spotify.search(q=f'artist:{cleaned_artist_name}', type='artist', market=None)
    artist_items = results['artists']['items'] 
    for artist in artist_items:
        if artist['name'].lower() == cleaned_artist_name.lower():
            return artist
        renamed_artist = MELONDY_TO_SPOTIFY['artist_name'].get(cleaned_artist_name, cleaned_artist_name)
        if artist['name'].lower() == renamed_artist.lower():
            return artist
    return {}

def get_spotify_album(artist_name: str, album_name: str) -> dict[str, Any]:
    
    cleaned_album_name = clean_name(album_name)
    cleaned_artist_name = clean_name(artist_name)

    if "&" in cleaned_artist_name:
        artists = cleaned_artist_name.split("&")
        for artist in artists:
            solo_artist_attempt_results = get_spotify_album(artist.strip(), cleaned_album_name)
            if solo_artist_attempt_results:
                return solo_artist_attempt_results
            # don't overwhelm spotify API rate limit
            time.sleep(0.1)

    # When the artist and album name are the same or even share the same word,
    # Spotify search doesn't like it, so we need to extract the album matching the
    # name or matching word in the name.
    if cleaned_artist_name.lower() == cleaned_album_name.lower() or \
       (any(x in cleaned_album_name for x in cleaned_artist_name.split(" "))):
        albums_with_artist_name = _spotify.search(q=f'album:{cleaned_album_name}', type='album', market=None)
        album_items = albums_with_artist_name['albums']['items']
        for album in album_items:
            if (len(album['artists']) > 0 and
                album['name'].lower() == cleaned_album_name.lower() or
                cleaned_artist_name in album['name'].lower()
            ):
                tracks = _spotify.album_tracks(album_id=album["id"])
                track_items = tracks['items']
                artist_popularity = get_spotify_artist_popularity(cleaned_artist_name)
                return {"album": album, "tracks": track_items, "artist_popularity": artist_popularity}

    results = _spotify.search(q=f'artist:{cleaned_artist_name} album:{cleaned_album_name}', type='album', market=None)
    album_items = results['albums']['items']

    # Try to find the album using the initial search results.
    found_album_data = get_album_data_from_items(album_items, cleaned_album_name)
    if found_album_data:
        artist_popularity = get_spotify_artist_popularity(cleaned_artist_name)
        found_album_data["artist_popularity"] = artist_popularity
        return found_album_data

    # If the album wasn't found, it might be due to different naming conventions.
    # We check our manual translation dictionary (MELONDY_TO_SPOTIFY).
    cleaned_artist_name = MELONDY_TO_SPOTIFY['artist_name'].get(cleaned_artist_name, cleaned_artist_name)
    cleaned_album_name = MELONDY_TO_SPOTIFY['album_name'].get(cleaned_album_name, cleaned_album_name)
    results = _spotify.search(q=f'artist:{cleaned_artist_name} album:{cleaned_album_name}', type='album', market=None)
    album_items = results['albums']['items']
    found_album_data = get_album_data_from_items(album_items, cleaned_album_name)
    if found_album_data:
        artist_popularity = get_spotify_artist_popularity(cleaned_artist_name)
        found_album_data["artist_popularity"] = artist_popularity
        return found_album_data

    print(f'Album "{cleaned_album_name}" by {cleaned_artist_name} was not retrievable via Spotipy\'s API.')
    try:
        print(f'Here is what returned for the names of album_items: {[album["name"] for album in album_items]}')
    except KeyError:
        print(f'Here is what returned for the names of album_items: {album_items}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
    return {}

def process_spotify_album_data(album_dict: dict[str, dict]) -> list[list[Any]]:
    """
        Given a spotify response about an album, we process it further to create a tabular dataset.
        Returns the following features:

        "total_tracks" - the number of tracks on the album
        "num_available_markets" - the number of markets this album is available in
        "release_year" - the year the album released
        "release_month" - the month the album released
        "release_day" - the day the album released
        "album_duration_in_s" - album duration in seconds
        "explicit_proportion" - the proportion of songs on the album that have explicit lyrics
        "featured_artists" - list of featured artists on the album
        "num_features" - the number of featured artists on the album
        "track_names" - list of the track names
        "artist_popularity" - artist popularity as assumed by Spotify
    """
    if album_dict != {}:
        total_tracks = album_dict['album']['total_tracks']
        num_available_markets = len(album_dict['album']['available_markets'])
        match album_dict['album']['release_date_precision']:
            case "day":
                release_year, release_month, release_day = album_dict['album']['release_date'].split('-')
            case "month":
                release_day = None
                release_year, release_month = album_dict['album']['release_date'].split('-')
            case "year":
                release_day, release_month = None, None
                release_year = album_dict['album']['release_date']
        album_duration_in_s = sum([track.get('duration_ms', 0) for track in album_dict['tracks']]) / 1000
        explicit_proportion = np.mean([track.get('explicit') for track in album_dict['tracks']]).item()
        featured_artists = get_album_features(album_dict['tracks'])
        num_features = int(len(featured_artists))
        track_names = [track.get('name') for track in album_dict['tracks']]
        artist_popularity = album_dict["artist_popularity"]

        return (
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
            artist_popularity
        )
    else:
        return (None,) * 11