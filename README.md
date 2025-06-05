# Twitter Sentiment Analysis using Logistic Regression

This project performs **sentiment analysis on tweets** using **Logistic Regression**, trained on the **Sentiment140** dataset. It aims to classify tweets as either positive or negative based on their text content.

## 📌 Project Overview

* **Goal**: Build a machine learning model that can predict the sentiment (positive/negative) of a tweet.
* **Model Used**: Logistic Regression
* **Dataset**: Sentiment140
* **Accuracy Achieved**: **77.6%**

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
* **Evaluation Metric**: Accuracy — achieved **77.6%** on test data.

### 3. Model Saving

* The trained model and vectorizer were saved using `pickle` for future use in predicting sentiments on new, unseen tweets.

## 🔍 How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook or use the saved model to predict new tweet sentiments:

   ```python
   # Load the saved model and vectorizer
   with open('logistic_model.pkl', 'rb') as model_file:
       model = pickle.load(model_file)

   with open('vectorizer.pkl', 'rb') as vec_file:
       vectorizer = pickle.load(vec_file)

   # Predict sentiment
   new_tweet = ["I love using Python for data science!"]
   cleaned = preprocess(new_tweet)  # apply same preprocessing
   transformed = vectorizer.transform(cleaned)
   prediction = model.predict(transformed)
   ```

## 📂 Files in the Repository

* `Twitter_Sentitment_analysis_using_ML.ipynb`: Jupyter notebook containing full code.
* `logistic_model.pkl`: Saved trained Logistic Regression model.
* `vectorizer.pkl`: Saved CountVectorizer instance.
* `README.md`: Project description and usage instructions.

## 🚀 Future Enhancements

* Deploy as a web app using Flask or Streamlit.
* Include support for neutral sentiment classification.
* Experiment with advanced models like Naive Bayes, SVM, or deep learning models (LSTM, BERT).

## 📈 Results

* **Model**: Logistic Regression
* **Test Accuracy**: **77.6%**
