import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import nltk
from collections import Counter
from nltk.corpus import stopwords
import re
from nltk.tokenize import WhitespaceTokenizer
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
import string
import os

stopwords_list = nltk.corpus.stopwords.words('english')

dataset = pd.read_csv('./Datasets/Spam-Classification.csv')

dataset['SMS'] = dataset['SMS'].apply(lambda x: x.lower())
tk = WhitespaceTokenizer()
dataset['tokenized_sms'] = dataset['SMS'].apply(lambda x: tk.tokenize(x))

def remove_punc(text):
    punc_free = "".join([i for i in text if i not in string.punctuation])
    return punc_free

dataset['SMS'] = dataset['SMS'].apply(lambda x: remove_punc(x))

def tokenization(text):
    tk = WhitespaceTokenizer()
    return tk.tokenize(text)

dataset['tokenized_sms'] = dataset['SMS'].apply(lambda x: tokenization(x))

def remove_stopwords(text):
    output = [word for word in text if word not in stopwords_list]
    return output

dataset['cleaned_sms'] = dataset['tokenized_sms'].apply(lambda x: remove_stopwords(x))

dataset['cleaned_sms'] = dataset['cleaned_sms'].apply(lambda x: ' '.join(x))

print(dataset['cleaned_sms'].head())

vector = CountVectorizer()
message_vector = vector.fit_transform(dataset['cleaned_sms'])

dataset['CLASS'] = dataset['CLASS'].map({'ham': 0, 'spam': 1})

message = message_vector
label = dataset['CLASS']

train_message, test_message, train_label, test_label = train_test_split(message, label, test_size=0.2, random_state=30)

logistic_regression_classifier = LogisticRegression(random_state=30)
logistic_regression_classifier.fit(train_message, train_label)

label_predictions = logistic_regression_classifier.predict(test_message)

rmse = np.sqrt(mean_squared_error(test_label, label_predictions))
mse = mean_squared_error(test_label, label_predictions)
mae = mean_absolute_error(test_label, label_predictions)
r2 = r2_score(test_label, label_predictions)

cv_scores = cross_val_score(logistic_regression_classifier, train_message, train_label, cv=10, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)

accuracy = np.mean(cross_val_score(logistic_regression_classifier, train_message, train_label, cv=10, scoring='accuracy'))

precision = np.mean(cross_val_score(logistic_regression_classifier, train_message, train_label, cv=10, scoring='precision'))

recall = np.mean(cross_val_score(logistic_regression_classifier, train_message, train_label, cv=10, scoring='recall'))

f1 = np.mean(cross_val_score(logistic_regression_classifier, train_message, train_label, cv=10, scoring='f1'))

excel_sheet = 'all_metrics.xlsx'
sheet_name = 'Perf Metrics'


metrics_df = pd.DataFrame({
    'ML Model': 'Logistic Regression',
    'RMSE': [rmse],
    'MSE': [mse],
    'MAE': [mae],
    'R squared': [r2],
    'Cross-validation RMSE': [cv_rmse.mean()],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1 Score': [f1],
    'Dataset': 'spam.csv'
})

try:
    existing_metrics = pd.read_excel(excel_sheet, sheet_name=sheet_name)
except FileNotFoundError:
    existing_metrics = pd.DataFrame(columns=metrics_df.columns)

existing_metrics = pd.concat([existing_metrics, metrics_df], ignore_index=True)

with pd.ExcelWriter(excel_sheet, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    existing_metrics.to_excel(writer, sheet_name=sheet_name, index=False)

