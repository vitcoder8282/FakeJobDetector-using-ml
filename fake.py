#T LIBRARIES NEEDED

#pandas is used for data manipulation and data analysis
#seaborn is used for data visualization and for creating plots,heatmaps
#skitlearn is used for ml algorithms

import pandas as pd
import numpy as np
import matplotlib as mp
import seaborn as sb
import sklearn as skl

#LOAD THE DATASET

#Here we create a dataframe and read the csv file and we print the dimensions of the dataframe
df = pd.read_csv("fake_job_postings.csv")
print(df.shape)

#BASIC EDA(EXPLORATORY DATA ANALYSIS)

#EDA is used to analyze datasets,visualizing patterns and trends

#prints rows,columns and datatypes of our dataset and also detects columns with null values
print(df.info())
# "df.isnull()" returns all null values and ".sum()" smums up the null values
print(df.isnull().sum())
#"df["fraudlent"]" selects a column in datset and ".value_counts" counts how many times does nul values occured and "normalize  = True" converts it into percentages
# 0=real and 1 = fake
print(df["fraudulent"].value_counts(normalize = True))

#COMBINING TEXT FIELDS

# ".fillna()" replaces the null values in the specific column with empty strings
df["text"] = df["title"].fillna('')+df["location"].fillna('')+df["description"].fillna('')+df["requirements"].fillna('')
df  = df.dropna(subset = ["fraudulent"])
df["fraudlent"] = df["fraudulent"].astype(int)
#NLP VECTORIZATION

'''Nlp vectorization is the process of converting text to numbers so the ml model can understand it 
and TF-IDF(Term frequency-inverse documnent frequency) removes the common english words as is,as,the etc'''

#This will import Tfidfvectorizer class from sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
#creates a tfidf a object for TfidfVectorizer class  and "max_features" only considers top 5000 words in post and "stop_words" removes common english words
tfidf = TfidfVectorizer(max_features=5000,stop_words='english')
#"fit" learns vocabulary from top 5000 words and "transform" converts text to tfidf vector of size 5000
X = tfidf.fit_transform(df["text"])
#assigns a new column in dataframe so ml model learns patterns from X and predicts y
y = df["fraudulent"]

#TRAIN/TEST SPLIT

'''This is important to split our training data into 2 parts training data where we will train our model
on that data and another part is testing data where our model is tested'''

#imports train_test split function from module model_selection from sklearn library to test/train our data
from sklearn.model_selection import train_test_split
''' train_test splits the dataset into training and testing data and "test_size=0.2" defines 20% data
goes for testing  and "startify = y" ensures that test/train preserves the same orginal distribution
of fake and real jobs data and "random state = 42" ensures that every time same train/test split is considered 
for model training and testing'''
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

#MODEL TRAINING (LOGISTIC REGRESION)


''' Model training is method for training our model  on training data using algorithms which detect patterns
.logistic regression is a algorithm used for binary classification '''

#Importing LogisticRegression class from sklearn to train our model
from sklearn.linear_model import LogisticRegression
#classification_report prints how efficient our model performed using prescison,reacll and confusion matrix  returnns count of tn/fp/fn/tn
'''TN → Model said No, and it was No ✅
FP → Model said Yes, but it was No ❌
FN → Model said No, but it was Yes ❌
TP → Model said Yes, and it was Yes ✅'''
from sklearn.metrics import classification_report,confusion_matrix
#so we creating an instance of Logistic regression and "Max_iter" gives gives max no of iterations to solver to use with correct weights
model  =LogisticRegression(max_iter = 1000)
# .fit" trains the logistic regression model where X-train is table of inputs and Y_train is table of outputs
model.fit(X_train,y_train)
#Predict
#It uses trained model to predict class labels
y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))


#EVALUAING THE MATRIX

''' Confusion matrix is a table used to evaluate how well our model is performing.Binary classification
is a way of categorizing the data into 2 classes: real and fake'''

import seaborn as sns
import matplotlib.pyplot  as plt
#confusion matrix compares labels y_test with predictes y_pred
from sklearn.metrics import confusion_matrix
#we created a confusion matrixand output is 2*2 matrix of [[tp,fp],[fn,tp]]
cm = confusion_matrix(y_test,y_pred)
''' Creating a heatmap to visualize the confusion matrix and annot =  True will show numbers in heatmap and
fmt = "d' tells to format no to integers and cmap ="blues" fixes the colour'''
sns.heatmap(cm,annot = True,fmt = "d",cmap = "Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print(df.columns)


#SAVE THE MODEL AND VECTORIZER

'''We will save the model to reuse it and vectorizer to convert  text to int.Add "pickle library saves and loads our
trained and saved model'''

import pickle
#Save model
#creates  a file with model is saved
with open("fake_job_model.pkl","wb") as f:
    #it serializes model and writes it to file f
    pickle.dump(model,f)
with open ("tfidf_vectorizer.pkl","wb") as f:
    pickle.dump(tfidf,f)

#OUTPUT/INPUT

while True:
    print("\nEnter job posting text or type 'exit' or quit")
    job_text = input()
    if job_text.lower() == 'exit':
        break
    #Preprocess input
    job_features = tfidf.transform([job_text])

    #Predict
    pred = model.predict(job_features)[0]
    #Result
    print(f"Prediction:{'Fake job Posting' if pred == 1 else 'Real Job Posting'}")
