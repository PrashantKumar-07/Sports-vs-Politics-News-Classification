import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset from a CSV file, handling potential issues with separators and bad lines
news_df = pd.read_csv(
    "bbc-news-data.csv",
    sep="\t",
    engine="python",
    on_bad_lines="skip"
)

# Clean up column names by stripping whitespace and converting to lowercase
news_df.columns = [col.strip().lower() for col in news_df.columns]
print("Available columns in the dataset:", news_df.columns.tolist())
print("Number of rows loaded:", len(news_df))

# Filter the data to include only 'sport' and 'politics' categories
filtered_news = news_df[news_df["category"].isin(["sport", "politics"])]

# Display a few examples from each category for verification
print("\nExamples of sport-related titles:")
print(filtered_news[filtered_news["category"] == "sport"][["category", "title"]].head(5))

print("\nExamples of politics-related titles:")
print(filtered_news[filtered_news["category"] == "politics"][["category", "title"]].head(5))

# Extract the title texts, filling any missing values with empty strings
title_texts = filtered_news["title"].fillna("").values

# Create binary labels: 0 for sport, 1 for politics
category_labels = np.array([0 if cat == "sport" else 1 for cat in filtered_news["category"]])

# Split the dataset into training and testing sets, ensuring stratified sampling
train_texts, test_texts, train_labels, test_labels = train_test_split(
    title_texts,
    category_labels,
    test_size=0.2,
    random_state=42,
    stratify=category_labels
)

# Initialize TF-IDF vectorizer with parameters to preprocess text
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 1),
    min_df=2,
    max_df=0.9
)

# Transform the training and testing texts into TF-IDF feature vectors
train_features = vectorizer.fit_transform(train_texts)
test_features = vectorizer.transform(test_texts)

# Define a list of classifiers to evaluate
classifiers = [
    ("Multinomial Naive Bayes", MultinomialNB()),
    ("Logistic Regression", LogisticRegression(max_iter=1000)),
    ("Linear Support Vector Machine", LinearSVC(dual=False)),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
]

# Store results for each model
evaluation_results = []

# Train and evaluate each classifier
for model_name, classifier in classifiers:
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    
    # Calculate performance metrics
    acc = accuracy_score(test_labels, predictions)
    prec = precision_score(test_labels, predictions)
    rec = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    
    evaluation_results.append((model_name, acc, prec, rec, f1))

# Create a DataFrame to display the results, sorted by F1-score
results_df = pd.DataFrame(
    evaluation_results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
)

print("\nModel Performance Comparison:")
print(results_df.sort_values(by="F1-Score", ascending=False))
