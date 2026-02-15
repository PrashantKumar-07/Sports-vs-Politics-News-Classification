# Sports vs Politics News Classification

## Overview
This project focuses on classifying news articles into **Sports** and **Politics** categories using classical machine learning techniques. The objective is to analyze how traditional text classification models perform on real-world news data when combined with simple and interpretable feature representations.

This work is part of the course **CSL 7640: Natural Language Understanding**.

---

## Dataset
The dataset used in this project is the **BBC News Archive**, which is publicly available on Kaggle.

**Dataset link:**  
https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive

The original dataset contains five categories:
- business  
- entertainment  
- politics  
- sport  
- tech  

For this project, only the **Sport** and **Politics** categories were used.  
Classification was performed using **news article titles only**.

---

## Preprocessing
The following preprocessing steps were applied:

- Converted all text to lowercase  
- Removed English stopwords  
- Used TF-IDF for feature representation  
- Considered unigram features only  

No stemming or lemmatization was applied.

---

## Machine Learning Models
The following machine learning models were implemented and compared:

- Multinomial Naive Bayes  
- Logistic Regression  
- Linear Support Vector Machine  
- Decision Tree  
- Random Forest  

All models were trained and evaluated using the same feature representation and data split.

---

## Evaluation
The dataset was split into training and testing sets using an **80:20 ratio** with stratified sampling.  
The models were evaluated using the following metrics:

- Accuracy  
- Precision  
- Recall  
- F1 Score  

Among all models, **Linear Support Vector Machine** achieved the best overall performance.

---

## Requirements
This project was developed and tested using:

- Python **3.11.5**
- pandas
- numpy
- scikit-learn

---

## How to Run
1. Download the BBC News dataset from the Kaggle link above  
2. Place the dataset file `bbc-news-data.csv` in the same directory as the script  
3. Run the experiment using:

```bash
python run_experiment.py
