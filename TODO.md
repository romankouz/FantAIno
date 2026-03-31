# Data

[ ] When you collect an album, get ALL the artists that were main artists in the dataset. Ex: Silk Sonic will put Anderson Paak as a featuring artist instead of a lead author. Perhaps make a second column called, artist_2 or something.
[ ] Convert spotify_etl to functions.

### 0) Using boto3, create an S3 bucket called fantaino-data.
### 1) Check Melondy for the latest review. If the album and artist has been recorded, break.
### 2) Otherwise, process the album, merge with spotify, and extract lyrics, and write to S3 bucket.

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

# Done!

[x] Fix the models to implement the abstract class.
[x] Handle the confusion matrix as a reporting function.
[x] Create an abstract class like Jeff did. 
[x] Rewrite KNN to have a pipe with standard scaler.
[x] If artists are separated by &, try getting the album ONLY with the first artist.
[X] CatBoost support.
[x] Try to get the lyrics of every song with Genius.

