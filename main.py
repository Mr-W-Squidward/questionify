import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ORGANIZE DATA
# Will use logistic regression model
# Learn gridsearch and scikitlearn and linear regression thx

vectorizer = TfidfVectorizer()