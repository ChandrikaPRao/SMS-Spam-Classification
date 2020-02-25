# -*- coding: utf-8 -*-
"""
This project classifies SMS messages as SPAM or NOT SPAM(HAM). Please find 
more details below

Context:
    The SMS Spam Collection is a set of SMS tagged messages that have been 
    collected for SMS Spam research. It contains one set of SMS messages in
    English of 5,574 messages, tagged acording being ham (legitimate) or spam.

Content:
    The files contain one message per line. Each line is composed by two 
    columns: v1 contains the label (ham or spam) and v2 contains the raw text.

Dataset:
    https://www.kaggle.com/uciml/sms-spam-collection-dataset

Inspiration:
    Can you use this dataset to build a prediction model that will accurately 
    classify which texts are spam?

Created on Thu Jan 30 11:29:43 2020
Author: Chandrika Rao
Version: 01/30/2020
Email: chandrika.1209@gmail.com
"""

import numpy as np
import pandas as pd


from sklearn.naive_bayes import *
from xgboost import XGBClassifier
import xgboost
from sklearn import svm

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction import DictVectorizer

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix

from collections import defaultdict
import re 

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

###############################################################################   

def preprocess_string(str_arg):
    """
    Preprocess the string argument - str_arg - such that :
        1. punctuations are removed
        2. multiple spaces are replaced by single space
        3. str_arg is converted to lower case  
    """
    
    processed_str=re.sub('[^a-z\s]+',' ',str_arg,flags=re.IGNORECASE)
    processed_str=re.sub('(\s+)',' ',processed_str)
    processed_str=processed_str.lower()
    
    return processed_str

###############################################################################   

def display_scores(actual_label, predicted_label):
    """
    This function will compare the actual and predicted data and output scores:
    f1, precision, accuracy and ROC_AUC along with confusion matrix.
    """
    accuracy = round(accuracy_score(actual_label, predicted_label),5)
    f1 = round(f1_score(actual_label, predicted_label),5)
    precision = round(precision_score(actual_label, predicted_label),5)
    ROC_AUC = round(roc_auc_score(actual_label, predicted_label),5)
    c_matrix = confusion_matrix(actual_label, predicted_label)  
    
    print("Confusion Matrix:")
    print(pd.DataFrame(data = c_matrix, columns = ['Predicted HAM(0)', \
                                                   'Predicted SPAM(1)'],
            index = ['Actual HAM(0)', 'Actual SPAM(1)']))
    print("Scores for the Test data >>> Accuracy:",accuracy, ", f1 score:",\
          f1, ", Precision:",precision,", ROC_AUC:", ROC_AUC,"\n")
#    print("------------------------------------------------------------------")

###############################################################################   
    
def classify(classifiers, vectorizers, parameters, train_data, test_data):
    """
    This function will run all the classifiers with the specified
    vectorizers on the training data first. It chooses the best hyperparameters
    usign the grid search. Using the best estimator, the chosen model will be
    run on the final test data.
    
    Input:
        1. classifiers : The classifiers we are interested to use
        2. vectorizers : These vectorizers will be used in combination with the
            classifier
        3. parameters : These are the parameters that will be used by 
            GridSearchCV functionality to identify the best parameters.
        4. train_data : Training data
        5. test_data : Test data
    """
    
    print("Running each classifier...")
    
    for classifier in classifiers:
        params_dict = {}
        for vectorizer in vectorizers:
            print("------------------------------------------------------------------")
            pipe_steps = [vectorizer, classifier]

            if (vectorizer[0] == "hVec" and classifier[0] == "XGB"):
                break
            else:
                try:
                    for params in parameters:
                        for key, value in params.items ():
                            if (key.split('__')[0]) == (classifier[0]):
                                params_dict[key]=value
                
                    pipeline = Pipeline(pipe_steps)
                    
                    grid_search = GridSearchCV(pipeline, param_grid = params_dict, cv=10)
                    grid_search.fit(train_data.text, train_data.label)   
                    final_model = grid_search.best_estimator_
                    predicted_val = final_model.predict(test_data.text)

                    print("For classifier: ", classifier[0], ", vectorizer: ",\
                          vectorizer[0], " and parameters: ",params_dict)    
                    print("Best Parameters:", grid_search.best_params_)  
                    display_scores(test_data.label, predicted_val)
                except:
                    print("Classifier: ", classifier[0], ", with vectorizer:",\
                          vectorizer[0],", cannot be used for classification.")

###############################################################################    
def EDA():
    global data
    
#    print(data.groupby("label").describe())
    
    #Histograms of message length by label type
    data['msg_length'] = data['text'].apply(len)
    
    _,ax = plt.subplots(figsize=(10,5))
    sns.kdeplot(data.loc[data.label == 0, "msg_length"], shade=True, label="Ham", clip=(0, 250)) # removing observations with message length above 250 because there is an outlier
    sns.kdeplot(data.loc[data.label == 1, "msg_length"], shade=True, label="Spam")
    ax.set(xlabel = "Message length", ylabel = "Density",title = "Spam messages are longer than ham messages, concentrated around 150 characters")
    plt.show()
    
    #data = data.drop(['msg_length'], axis=1)

###############################################################################
    
def data_prep():    
    """
    This function will import the data from spam.csv file into pandas
    dataframe followed by few data preprocessing steps.
    """
    global data, train, test, train_data, train_labels,test_data, test_labels
    
    data = pd.read_csv("Dataset/spam.csv", encoding='latin-1', na_filter=False)
    print("data.describe() :","\n" ,data.describe(),"\n")
    print("data.info() :","\n" ,data.info(),"\n")
    print("data.head(5) :","\n" ,data.head(5),"\n")
    
    #Convert the label column values, ham > 1 and spam > 0
    data["v1"]=data["v1"].map({'spam':1,'ham':0})
    
    # Rename the columns
    data = data.rename(columns={"v1":"label", "v2":"text", "Unnamed: 2":"UN2", "Unnamed: 3":"UN3", "Unnamed: 4":"UN4"})
    # Merge columns 3 to 5 that hold message information with column 2 and drop them
    data['text'] = data['text'].map(str) + ' ' + data['UN2'].map(str) + ' ' + \
                data['UN3'].map(str) + ' ' + data['UN4'].map(str)
    data = data.drop(["UN2", "UN3", "UN4"], axis=1)

    # Stemming for clubbing similar words together
    #stemmer = SnowballStemmer("english")
    #data["text"] = data["text"].apply(cleanText)
        
    #split the data into train and test sets
    train, test =  train_test_split(data, test_size = 0.2)
    train_data = train['text']
    train_labels = train['label']
    test_data=test['text']
    test_labels=test['label']

###############################################################################

if __name__ == "__main__":
    
    # Specify the classifers you want to use on the data
    classifiers = [('bNB', BernoulliNB()),
                   ('mNB', MultinomialNB()),
                   ('XGB', XGBClassifier()),
                   ('SVC', svm.SVC()),
                   ('LinearSVC', svm.LinearSVC())]      
    
    # Specify the vectorizers you want to use on the data. Vectorizers
    vectorizers = [
                ('cVec', CountVectorizer()),
               ('tVec', TfidfVectorizer()),
               ('hVec', HashingVectorizer())] 
    
    # Specify the parameters that should be used by the GridSearchCV 
    # functionality to identify the best parameters. 
    # NOTE : The first characters in the key should match with the keys of 
    # 'classifiers' dictionary. 
    # Ex: 'bNB__alpha' for ('bNB', BernoulliNB())
    check_params = [
            {'bNB__alpha':[0.05, 0.10,0.5,1.0, 1.2]},
            {'mNB__alpha':[0.05, 0.10,0.5,1.0, 1.2]},
            {'coNB__alpha':[0.05, 0.10,0.5,1.0, 1.2]},
            {'SVC__kernel':['sigmoid','linear']},
            {'SVC__C':[500, 1000,1500]}
            ]
    
    # Import the data into pandas dataframe followed by data preprocessing
    data_prep()
    # Exploratory data analysis
    EDA()
    # Run the classifiers on the input data
    classify(classifiers, vectorizers, check_params, train, test)

