import pandas as pd
import nltk
from nltk.corpus import stopwords
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle

df = pd.read_csv(r'C:/Binus\Semester 5/NLP/Coba Pycharm/Datasets/clickbait.csv')
# print(df)

nltk.download("stopwords")
stop = stopwords.words('english')
# print(stop)

df['title_without_stopwords'] = df['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in(stop)]))
# print(df['title_without_stopwords'])

le = LabelEncoder()
le.fit(df['label'])
df['label_encoded'] = le.transform(df['label'])

title = df.title_without_stopwords
y = df.label_encoded

    # TFIDF
tfidf = TfidfVectorizer()
x_tfidf = tfidf.fit_transform(title)
x_tfidf.toarray()

x_tfidf_train, x_tfidf_test, y_tfidf_train, y_tfidf_test = train_test_split(x_tfidf, y, test_size = 0.2, random_state = 42)
x_tfidf_test, x_tfidf_val, y_tfidf_test, y_tfidf_val = train_test_split(x_tfidf_test, y_tfidf_test, test_size = 0.5, random_state = 42)

svm_clf_kernel = SVC(kernel = 'linear', probability = True)
svm_clf_kernel.fit(x_tfidf_train, y_tfidf_train)
    
pickle.dump(svm_clf_kernel, open("model.pkl", "wb"))
model = pickle.load(open("model.pkl", "rb"))

    
    



