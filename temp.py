# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import nltk

nltk.download()

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

string1 = "hi singa the self driving car will be late Best Linda"
string2 = "This is cool stuff to learn. I love it."
email_list = [string1, string2]

bag_of_words = vectorizer.fit(email_list)
ag_of_words = vectorizer.transform(email_list)



print(vectorizer.vocabulary_.get('singa'))