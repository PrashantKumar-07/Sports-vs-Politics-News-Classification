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


# load csv safely
data = pd.read_csv(
    "bbc-news-data.csv",
    sep="\t",
    engine="python",
    on_bad_lines="skip"
)

# clean column names 
data.columns = [c.strip().lower() for c in data.columns]
print("Columns found:", data.columns.tolist())
print("Total rows after loading:", len(data))


# keep only sport and politics
data = data[data["category"].isin(["sport", "politics"])]


# small preview for clarity
print("\nSample sport articles:")
print(data[data["category"] == "sport"][["category", "title"]].head(5))

print("\nSample politics articles:")
print(data[data["category"] == "politics"][["category", "title"]].head(5))


# use title text
texts = (data["title"].fillna("")).values


# labels: sport -> 0, politics -> 1
labels = np.array([0 if c == "sport" else 1 for c in data["category"]])


# split data
x_train, x_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


# tf-idf
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 1),
    min_df=2,
    max_df=0.9
)

x_train_vec = tfidf.fit_transform(x_train)
x_test_vec = tfidf.transform(x_test)


# models
model_list = [
    ("NaiveBayes", MultinomialNB()),
    ("LogReg", LogisticRegression(max_iter=1000)),
    ("LinearSVM", LinearSVC(dual=False)),
    ("DecisionTree", DecisionTreeClassifier(random_state=42)),
    ("RandomForest", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
]


results = []

for name, clf in model_list:
    clf.fit(x_train_vec, y_train)
    preds = clf.predict(x_test_vec)

    results.append((
        name,
        accuracy_score(y_test, preds),
        precision_score(y_test, preds),
        recall_score(y_test, preds),
        f1_score(y_test, preds)
    ))


res_df = pd.DataFrame(
    results,
    columns=["model", "accuracy", "precision", "recall", "f1"]
)

print("\nResults:\n")
print(res_df.sort_values(by="f1", ascending=False))
