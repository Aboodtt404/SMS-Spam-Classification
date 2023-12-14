import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
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

# label_counts = dataset['CLASS'].value_counts()
#                                        
# print("Class Counts:")
#                                 #testing purposes only
# print(label_counts)

#text preprocessing
#lower casing
dataset['SMS'] = dataset['SMS'].apply(lambda x: x.lower())
tk = WhitespaceTokenizer()
dataset['tokenized_sms'] = dataset['SMS'].apply(lambda x: tk.tokenize(x))

# Remove Punctuation
def remove_punc(text):
    punc_free = "".join([i for i in text if i not in string.punctuation])
    return punc_free

dataset['SMS'] = dataset['SMS'].apply(lambda x: remove_punc(x))

#Stopwords
", ".join(stopwords_list)

def tokenization(text):
    tk = WhitespaceTokenizer()
    return tk.tokenize(text)

dataset['tokenized_sms'] = dataset['SMS'].apply(lambda x: tokenization(x))

#removing stopwords

def remove_stopwords(text):
    output= [word for word in text if word not in stopwords_list]
    return output

dataset['cleaned_sms'] = dataset['tokenized_sms'].apply(lambda x: remove_stopwords(x))

#Remove frequent words

counter = Counter()
for text in dataset['cleaned_sms'].values:
    for word in text:
        counter[word] += 1

frequent_words = set([w for (w, wc) in counter.most_common(10)])

def remove_frequent_words(text):
    return " ".join([word for word in text if word not in frequent_words])

dataset['filtered_sms'] = dataset['cleaned_sms'].apply(lambda x: remove_frequent_words(x))

#reduce vocab size (Stemming) for the model

porter_stememer = PorterStemmer()

def stemming(text):
    stem_text = [porter_stememer.stem(text) for text in text]
    return stem_text

# dataset['filtered_sms'] = dataset['filtered_sms'].apply(lambda x: stemming(x))


#remove urls and htmls

def remove_urls(text):
    if isinstance(text, str):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    else:
        return text


def remove_html(text):
    if isinstance(text, str):
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub(r'', text)
    else:
        return text


dataset['filtered_sms'] = dataset['filtered_sms'].apply(lambda x: remove_urls(x))
dataset['filtered_sms'] = dataset['filtered_sms'].apply(lambda x: remove_html(x))

#rare words removal
cnt = Counter()
for text in dataset['filtered_sms'].values:
    for word in text:
        cnt[word] += 1
 
n_rare_words = 10
 
Rare_words = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
def remove_rarewords(text):
    return " ".join([word for word in str(text).split() if word not in Rare_words])

# dataset['filtered_sms'] = dataset['filtered_sms'].apply(lambda x: remove_rarewords(x))


message = dataset['filtered_sms']
label = dataset['CLASS']

# print(dataset.head())  #Exploring the dataset just in case

train_message, test_message, train_label, test_label = train_test_split(message, label, test_size=0.1, random_state=30)

vector = CountVectorizer()
train_message_vector = vector.fit_transform(train_message)
test_message_vector = vector.transform(test_message)

random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=30)

random_forest_classifier.fit(train_message_vector, train_label)

label_predictions = random_forest_classifier.predict(test_message_vector)

cv_scores = cross_val_score(random_forest_classifier, train_message_vector, train_label, cv=10, scoring='accuracy')

excel_sheet = 'all_metrics.xlsx'
sheet_name = 'Perf Metrics'

metrics_df = pd.DataFrame({
    'ML Model': 'Random Forest',
    'Accuracy': [cv_scores.mean()],
    'Precision': [precision_score(test_label, label_predictions, pos_label='spam')],
    'Recall': [recall_score(test_label, label_predictions, pos_label='spam')],
    'F1 Score': [f1_score(test_label, label_predictions, pos_label='spam')],
    'Dataset' : 'Spam-Classification.csv'
})

try:
    existing_metrics = pd.read_excel(excel_sheet, sheet_name=sheet_name)
except FileNotFoundError:
    existing_metrics = pd.DataFrame(columns=metrics_df.columns)

existing_metrics = pd.concat([existing_metrics, metrics_df], ignore_index=True)

with pd.ExcelWriter(excel_sheet, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    existing_metrics.to_excel(writer, sheet_name=sheet_name, index=False)