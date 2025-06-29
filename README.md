# Twitter Sentiment Analysis using Logistic Regression

This project performs **sentiment analysis on tweets** using **Logistic Regression**, trained on the **Sentiment140** dataset. It aims to classify tweets as either positive or negative based on their text content.

## 📌 Project Overview

* **Goal**: Build a machine learning model that can predict the sentiment (positive/negative) of a tweet.
* **Model Used**: Logistic Regression
* **Dataset**: Sentiment140
* **Accuracy Achieved**: **91.6%**

## 🔧 Features & Workflow

### 1. Data Preprocessing

To improve the quality of input text, the following steps were performed:

* **Cleaning**: Removed user mentions, hashtags, URLs, special characters, and numbers.
* **Lowercasing**: Converted all text to lowercase for consistency.
* **Stemming**: Applied stemming to reduce words to their root forms.
* **Tokenization**: Broke sentences into individual tokens.
* **Encoding**: Transformed text into numerical vectors using `CountVectorizer`.

### 2. Model Training

* **Model**: Trained a Logistic Regression model on the cleaned and vectorized tweet data.
* **Train-Test Split**: Used an 80-20 split for training and evaluation.
* **Evaluation Metric**: Accuracy — achieved **91.6%** on test data.

### 3. Model Saving

* The trained model and vectorizer were saved using `pickle` for future use in predicting sentiments on new, unseen tweets.

## 📈 Results

* **Model**: Logistic Regression
* **Test Accuracy**: **91.6%**
