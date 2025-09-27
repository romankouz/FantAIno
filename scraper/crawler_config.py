from pydantic import BaseModel
from typing import Callable


class Config(BaseModel):
    """
        url: str, the starting url to crawl
        match: str, the pattern to match in the url
        selector: str
        max_pages_to_crawl: int
        output_file_name: str, name of the file to save output to
        cookie: dict[str, str] | None = None, the necessary cookies needed to access pages to crawl
        on_visit_page: Callable[[str], None] | None = None
    """
    url: str
    match: str
    selector: str
    max_pages_to_crawl: int
    output_file_name: str
    cookie: dict[str, str] | None = None
    on_visit_page: Callable[[str], None] | None = None
