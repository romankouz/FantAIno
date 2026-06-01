# NN

[X] Why is loss NAN?
[ ] Incorporate Optuna (first via jupyter notebook)
[ ] Incorporate nested CV (save yaml like results to a verified or CV_result.yaml)
[ ] Add early stopping callback
[X] Implement evaluate correctly.
[X] ckpt doesn't get overwritten. FIX!

# Reading

[ ] SageMaker
[ ] Parquet (blog)
[ ] Spark
[ ] Tabular Foundation Models

# Data

[ ] When you collect an album, get ALL the artists that were main artists in the dataset. Ex: Silk Sonic will put Anderson Paak as a featuring artist instead of a lead author. Perhaps make a second column called, artist_2 or something.
[ ] We currently have a data input pipeline for S3 that scrapes Melondy based on the most recent reviews. We should also have a pipeline for updating individual albums. Use Case: Album isn't found due to a weird name mismatch and we want to correct for that.
[ ] Write a flag that stops the scraper if it runs to an album it has processed before. This in tandem with the individual album update should help us have near 100% coverage of Fantano's album reviews.
[ ] Vision Transformer Embedding Support

# Models

[ ] Learning the weights of features in KNN: [https://medium.com/analytics-vidhya/feature-engineering-experiment-weighted-knn-3f28dfdf30e1](https://medium.com/analytics-vidhya/feature-engineering-experiment-weighted-knn-3f28dfdf30e1)
[ ] Explore sample weights with KNN to balance classes.
[ ] Weighted Feature Sampling for RF
[ ] Weighted KNN
[ ] LLM predictions
[ ] Finetuned LLM
[ ] LightGBM
[ ] XGBoost
[ ] Ensure CatBoost doesn't preprocess the categorical variables away.
[ ] AIC/BIC on regression model inputs.
[ ] Bayesian Hyperparamter Tuning

# Runtime Enhancements

[ ] When running a hydra multirun, find a way to cache the data so you're not retrieving the data from AWS every time.

# Code Cleanliness

[ ] Change the run name to reflect the dataset parameters.
[ ] Train and test of various models likely doesn't take pd.DataFrames but the type hints don't reflect that. Fix it.
[ ] Ensure `process_spotify_album_data` has stricter type checking.
[ ] Edit doc strings to have args returns notation. (`spotify_utils.py`, `aoty_scraper.py`)
[ ] Ensure method strings have new lines.
[ ] Make underscores for the relevant helper methods.
[ ] Introduce bidict for MELONDY_TO_SPOTIFY usage. Use bidict for spotify validation in ingestion pipeline.
[ ] Reorder imports.
    Standard library first
    fnmatch, json, os, collections, urllib.parse

    Third-party packages next
    requests, jsonlines, bs4

    Your project/local imports last
    constants, scraper.crawler_config, FantAIno

# Documentation

[ ] Create a sample env file for github and README.

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
[x] Edit linter to be less picky about line length and snake case.
[x] Ordinal Logistic Regression
[x] Create tabular database in AWS for album lyrics embeddings.
[x] Convert spotify_etl to functions.
[x] Ensure better logging than "print", especially in scratch. Save the print to log files to examine later.
[x] Create a logger for spotify in it's utils. It currently prints out when something wasn't retrievable by spotify.
[x] Explore `joblib.parallel_backend` for n_jobs argument in sklearn models.
[x] Merge existing classes to handle classification or regression in the same file. Take "regression" or "classification" as the task and edit the appropriate parameters accordingly.



