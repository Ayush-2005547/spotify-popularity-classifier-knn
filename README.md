# Spotify Popularity Classification using KNN

This project classifies Spotify tracks as **Popular** or **Unpopular** using the **K-Nearest Neighbors (KNN)** algorithm. It demonstrates data preprocessing, model training, evaluation, and interpretation using real-world audio feature data.

## Objective

Build a binary classification model to predict song popularity based on various audio features, helping to understand what makes a song popular.

## Dataset

- **Source**: [Spotify Tracks Dataset by Maharshi Pandya]
(www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- **Size**: Over 114,000 tracks
- **Target Variable**: `popularity` (binarized as Popular ≥ 70, else Unpopular)
- **Features Used**: 
  - danceability
  - energy
  - loudness
  - speechiness
  - acousticness
  - instrumentalness
  - liveness
  - valence
  - tempo
  - duration_ms

## Workflow Summary

1. **Data Preprocessing**
   - Handled missing values
   - Converted `popularity` into binary target variable
   - Normalized numerical features using MinMaxScaler

2. **Model Implementation**
   - Used `KNeighborsClassifier` from scikit-learn
   - Tried different `k` values (from 3 to 15)
   - Split data using train-test split

3. **Evaluation**
   - Measured performance using accuracy score
   - Analyzed confusion matrix
   - Identified top-performing value of `k`
   - Performed feature importance analysis via permutation method

4. **Visualization**
   - Confusion matrix heatmap
   - Accuracy vs K plot

## Results

- **Best Accuracy**: Achieved approximately `96%` accuracy at `k = 9`.
- **Key Observations**:
  - Features like `danceability`, `energy`, and `valence` had high influence.
  - Model performance was sensitive to normalization.
  - KNN showed decent results but would benefit from more advanced techniques or balanced class distributions.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Potential Enhancements

- Use multi-class genre classification instead of binary popularity.
- Integrate a Streamlit app for interactive predictions.
- Try other ML algorithms (Random Forest, Logistic Regression) for comparison.
- Handle class imbalance more effectively.
