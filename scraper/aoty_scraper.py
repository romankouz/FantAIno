import FantAIno
import asyncio
import json
import jsonlines
import fnmatch
from scraper.crawler_config import Config
import sys
import os
import requests
from urllib.parse import urljoin
from collections import deque

from bs4 import BeautifulSoup

from constants import AOTY_URL_ROOT

def crawl(config: Config):
    visited_pages = set()
    results = []
    queue = deque([config.url])

    # Session is used for making several requests to the same host. The underlying TCP connection will be reused, 
    # which can result in a significant performance increase
    with requests.Session() as session, jsonlines.open(config.output_file_name, 'a') as writer:

        # ensures cookie name and value are set if login is required for scraping
        # USE ENV VARIABLES TO SET COOKIE NAME AND VALUE
        if config.cookie:
            session.cookies.set(config.cookie['name'], config.cookie['value'], domain=config.url)

        while queue and len(results) < config.max_pages_to_crawl:
            # try:
            url = queue.popleft()
            # if the URL doesn't start with http, it is an endpoint relative the root, so we need to prepend it
            if not url.startswith("http") and url.startswith("/"):
                url = AOTY_URL_ROOT + url
            if url in visited_pages or not url.startswith("https://www.albumoftheyear.org/publication/57-the-needle-drop"):
                continue
            print(f"Crawler: Crawling {url}")
            response = session.get(url)
            print(response)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            if url != AOTY_URL_ROOT and "57-the-needle-drop/reviews" in url:
                html = soup.select_one(config.selector).get_text() if soup.select_one(config.selector) else ""
                results.append({'url': url, 'html': html})

            # Extract and enqueue links
            links = soup.find_all("a")
            for link in links:
                href = link.get("href")
                # ensure we only enqueue links that match the pattern in config.match
                if href and fnmatch.fnmatch(href, config.match):
                    new_url = urljoin(response.url, href)
                    queue.append(href)

            # except Exception as e: # Catch any general exception and store it in 'e'
            #     print(f"Crawler: An error occurred: {e}") # Print the error message
            visited_pages.add(url)

    return results


def main(config: Config):

    output_dir = os.path.dirname(config.output_file_name)
    os.makedirs(output_dir, exist_ok=True)
    
    results = crawl(config)
    with open(config.output_file_name, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":

    root_dir = os.path.dirname(os.path.abspath(FantAIno.__path__[0]))
    current_crawler_config = Config(
        url=AOTY_URL_ROOT,
        match=f"*/57-the-needle-drop/reviews/*",
        selector=".albumBlock",
        max_pages_to_crawl=100_000,
        output_file_name=os.path.join(root_dir, "data", "raw", "aoty_reviews.json")
    )
    main(current_crawler_config)

