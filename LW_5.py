import math
import numpy as np
import re

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('movie_reviews')
from nltk.corpus import movie_reviews
nltk.download('stopwords')
from nltk.corpus import stopwords

def stem_list_of_text(list_of_text):
    result = []
    ps = PorterStemmer()
    for text in list_of_text:
        word_list = text.lower().split()
        stop = set(stopwords.words('english'))
        stem_word = [ps.stem(word) for word in word_list if not word in stop]
        result.append(' '.join(stem_word))
    return result

def f_i_and_c():
    neg_file_ids = movie_reviews.fileids('neg')
    pos_file_ids = movie_reviews.fileids('pos')
    neg_reviews = [movie_reviews.raw(fileids=ids) for ids in neg_file_ids]
    pos_reviews = [movie_reviews.raw(fileids=ids) for ids in pos_file_ids]
    list_of_text = stem_list_of_text(neg_reviews + pos_reviews)
    cv = CountVectorizer()
    f_i = cv.fit_transform(list_of_text).toarray()
    c = np.array([0] * len(neg_reviews) + [1] * len(pos_file_ids))
    return (f_i, c)

class NBClassifier:
    def __init__(self):
        self.length = 0 # Feature matrix dimension
        self.stat = dict()
        self.d = 0 # Documensts count
        self.v = 0 # Number of unique words (dictionary size)
        """
        [Document class] -> (d_i, l_c, w_i_c)
        d_i   - [Number of documents of this class]
        l_c   - [Number of words in all documents of this class]
        w_i_c - [Number of occurrences i-th word in all documents of class]
        """
    def fit(self, f_i_matrix, c_vector):
        """
        f_i_matrix : Feature matrix - Cell (i, j) contains number of occurrences of j-th word in i-th document
        c_vector : Vector of classes assignment - contains result of the classification for i-th class
        """
        unique, counts = np.unique(c_vector, return_counts=True)
        length = len(f_i_matrix[0])
        self.length = length
        self.stat = { unique[i] : (counts[i], 0, np.array([0] * length)) for i in range(len(unique)) }
        self.d = counts.sum()
        vocabulary = np.array([0] * length)
        for i in range(len(c_vector)):
            temp = self.stat[c_vector[i]]
            self.stat[c_vector[i]] = (temp[0],
                temp[1] + f_i_matrix[i].sum(), temp[2] + f_i_matrix[i])
            vocabulary += f_i_matrix[i]
        self.v = np.count_nonzero(vocabulary)

    def predict(self, f_i_matrix):
        classes = []
        for j in range(len(f_i_matrix)):
            max_value = -math.inf
            c_value = None
            for c in self.stat.keys():
                result = math.log(self.stat[c][0] / self.d)
                for i in range(self.length):
                    if f_i_matrix[j][i]:
                        term = math.log(
                            (self.stat[c][2][i] + 1) /
                            (self.v + self.stat[c][1]))
                        result += f_i_matrix[j][i] * term
                if result > max_value:
                    max_value = result
                    c_value = c
            classes.append(c_value)
        return classes

f_i, c = f_i_and_c()
x_train, x_test, y_train, y_test = train_test_split(f_i, c, test_size = 0.3, random_state = 0, shuffle = True)

nb_class = NBClassifier()
nb_class.fit(x_train, y_train)
y_pred = nb_class.predict(x_test)
print('accuracy_score = {0}\n'.format(accuracy_score(y_test, y_pred)))

m_nb_class = MultinomialNB()
m_nb_class.fit(x_train, y_train)
y_pred = m_nb_class.predict(x_test)
print('accuracy_score = {0}\n'.format(accuracy_score(y_test, y_pred)))
