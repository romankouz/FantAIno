# Data
- [ ] When you collect an album, get ALL the artists that were main artists in the dataset. Ex: Silk Sonic will put Anderson Paak as a featuring artist instead of a lead author. Perhaps make a second column called, artist_2 or something.
- [X] If artists are separated by &, try getting the album ONLY with the first artist.
- [ ] Convert spotify_etl to functions.
- [ ] Try to get the lyrics of every song with Genius.

# Models
- [ ] Learning the weights of features in KNN: https://medium.com/analytics-vidhya/feature-engineering-experiment-weighted-knn-3f28dfdf30e1
- [x] Rewrite KNN to have a pipe with standard scaler.
- [ ] Explore sample weights with KNN to balance classes.
- [ ] Weighted Feature Sampling for RF
- [ ] CatBoost
- [ ] Weighted KNN
- [ ] LLM predictions
- [ ] Ordinal Logistic Regression

# Code Cleanliness
- [ ] Edit doc strings to have args returns notation.
- [ ] Ensure method strings have new lines.
- [ ] Make underscores for the relevant helper methods.
- [x] Fix the models to implement the abstract class.
- [X] Handle the confusion matrix as a reporting function.
- [x] Create an abstract class like Jeff did. 