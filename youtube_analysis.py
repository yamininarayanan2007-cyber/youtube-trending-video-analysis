import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("INvideos.csv")

# Show first 5 rows
print(df.head())

# Show dataset information
print(df.info())

# Top 10 most viewed videos
top_videos = df.sort_values(by="views", ascending=False).head(10)

print("\nTop 10 Most Viewed Videos:")
print(top_videos[["title", "channel_title", "views"]])

# Most popular channels by total views
popular_channels = df.groupby("channel_title")["views"].sum().sort_values(ascending=False).head(10)

print("\nTop Channels by Total Views:")
print(popular_channels)

# Views vs Likes scatter plot
plt.figure(figsize=(8,5))
sns.scatterplot(x="views", y="likes", data=df)

plt.title("Views vs Likes Relationship")
plt.xlabel("Views")
plt.ylabel("Likes")

plt.show()

# Top 10 most liked videos
top_liked = df.sort_values(by="likes", ascending=False).head(10)

print("\nTop 10 Most Liked Videos:")
print(top_liked[["title", "channel_title", "likes"]])

# Most popular categories
category_views = df.groupby("category_id")["views"].sum().sort_values(ascending=False)

print("\nViews by Category:")
print(category_views)

# Category distribution graph
plt.figure(figsize=(10,5))
sns.countplot(x="category_id", data=df)

plt.title("Distribution of YouTube Video Categories")
plt.xlabel("Category ID")
plt.ylabel("Number of Videos")

plt.show()

# Engagement rate
df["engagement_rate"] = (df["likes"] + df["comment_count"]) / df["views"]

top_engagement = df.sort_values(by="engagement_rate", ascending=False).head(10)

print("\nTop Videos by Engagement Rate:")
print(top_engagement[["title", "channel_title", "engagement_rate"]])

# Top 10 channels with most trending videos
top_channels = df["channel_title"].value_counts().head(10)

print("\nTop 10 Channels with Most Trending Videos:")
print(top_channels)

plt.figure(figsize=(10,5))
top_channels.plot(kind="bar")

plt.title("Top 10 Channels with Most Trending Videos")
plt.xlabel("Channel")
plt.ylabel("Number of Trending Videos")

plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))

corr = df[["views","likes","dislikes","comment_count"]].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")

plt.title("Correlation Between Views, Likes, Dislikes and Comments")

plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Features and target
X = df[["likes", "dislikes", "comment_count"]]
y = df["views"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

print("\nSample Predictions for Views:")
print(predictions[:10])

from sklearn.metrics import r2_score

score = model.score(X_test, y_test)
print("\nModel Accuracy (R² Score):", score)

import matplotlib.pyplot as plt

# Plot Actual vs Predicted
plt.figure(figsize=(8,5))
plt.scatter(y_test, predictions)

plt.xlabel("Actual Views")
plt.ylabel("Predicted Views")
plt.title("Actual vs Predicted Views")

plt.show()

from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(y_test, predictions)
print("\nMean Absolute Error:", error)

coeff = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
print("\nInfluence of Features on Views:")
print(coeff)

# Engagement distribution
plt.figure(figsize=(8,5))

sns.histplot(df["engagement_rate"], bins=50)

plt.title("Distribution of Video Engagement Rate")
plt.xlabel("Engagement Rate")
plt.ylabel("Number of Videos")

plt.show()

# Save processed dataset
df.to_csv("processed_youtube_data.csv", index=False)

print("Processed dataset saved successfully!")