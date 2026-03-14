# YouTube Trending Video Analysis and View Prediction

## Project Overview
This project analyzes YouTube trending video statistics using Python. The dataset was explored to understand patterns in views, likes, comments, and channel popularity. Data visualization techniques were used to identify relationships between engagement metrics.

A machine learning model was also implemented to predict video views based on likes, dislikes, and comment counts.

## Dataset
The project uses the **Trending YouTube Video Statistics** dataset.  
The India dataset (INvideos.csv) containing **37,352 records and 16 columns** was used for analysis.

## Technologies Used
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Visual Studio Code

## Analysis Performed
- Dataset exploration and preprocessing
- Top viewed and liked videos analysis
- Channel popularity analysis
- Data visualization (Views vs Likes)
- Engagement rate calculation

## Machine Learning
A **Linear Regression model** was used to predict video views based on:
- Likes
- Dislikes
- Comment count

Model Performance:
- R² Score: 0.768
- Mean Absolute Error: 624004

## Conclusion
The analysis shows that likes and comments strongly influence video views. Channels producing engaging content appear more frequently in the trending list. The machine learning model demonstrates that engagement metrics can be used to predict video popularity.

