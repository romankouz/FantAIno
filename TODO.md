# Data

[ ] Create tabular database in AWS for album lyrics embeddings.
[ ] When you collect an album, get ALL the artists that were main artists in the dataset. Ex: Silk Sonic will put Anderson Paak as a featuring artist instead of a lead author. Perhaps make a second column called, artist_2 or something.
[ ] Convert spotify_etl to functions.
[ ] We currently have a data input pipeline for S3 that scrapes Melondy based on the most recent reviews. We should also have a pipeline for updating individual albums. Use Case: Album isn't found due to a weird name mismatch and we want to correct for that.

# Models

[ ] Learning the weights of features in KNN: [https://medium.com/analytics-vidhya/feature-engineering-experiment-weighted-knn-3f28dfdf30e1](https://medium.com/analytics-vidhya/feature-engineering-experiment-weighted-knn-3f28dfdf30e1)
[ ] Explore sample weights with KNN to balance classes.
[ ] Weighted Feature Sampling for RF
[ ] Weighted KNN
[ ] LLM predictions
[ ] Finetuned LLM
[ ] Ordinal Logistic Regression

# Runtime Enhancements

[ ] Explore `joblib.parallel_backend` for n_jobs argument in sklearn models.

# Code Cleanliness

[ ] Ensure `process_spotify_album_data` has stricter type checking.
[ ] Edit doc strings to have args returns notation. (`spotify_utils.py`, `aoty_scraper.py`)
[ ] Ensure method strings have new lines.
[ ] Make underscores for the relevant helper methods.
[ ] Edit linter to be less picky about line length and snake case.
[ ] Ensure better logging than "print", especially in scratch. Save the print to log files to examine later.
[ ] Introduce bidict for MELONDY_TO_SPOTIFY usage.
[ ] Reorder imports.
    Standard library first
    fnmatch, json, os, collections, urllib.parse

    Third-party packages next
    requests, jsonlines, bs4

    Your project/local imports last
    constants, scraper.crawler_config, FantAIno

# Done!

[x] Fix the models to implement the abstract class.
[x] Handle the confusion matrix as a reporting function.
[x] Create an abstract class like Jeff did. 
[x] Rewrite KNN to have a pipe with standard scaler.
[x] If artists are separated by &, try getting the album ONLY with the first artist.
[X] CatBoost support.
[x] Try to get the lyrics of every song with Genius.
[x] S3 Integration. 
[x] Logging: UnicodeEncodeError: 'charmap' codec can't encode character '\u2010' in position 10: character maps to <undefined>
[x] Remove any sys.path.append and create a "package" of this project for imports. (pip install -e .)


