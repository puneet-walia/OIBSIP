#<u>Oasis Infobyte Project</u>
##Email Spam detection<br>
#"task 4"
###Arshpreet Singh


# importing packages
import pandas as pd
import numpy as np

# importing data
df = pd.read_csv("spam.csv", encoding='latin1') # had to change the encoder as our dataset has english alphabets
df = df.iloc[:,0:2]

df.head()

# inspect dataset
df.groupby("v1").describe()

# adding a new column in dataframe
df['spam'] = df['v1'].apply( lambda x : 1 if x=='spam' else 0)

df.head()

# train - test split
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(df.v2,df.spam,test_size=0.25)

#find word count
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values);

x_train_count.toarray()

# training model
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train_count,y_train)

# Manual validation
sample = ["REWARD click the link"]
sample = cv.transform(sample)
x = model.predict(sample)

print("Spam") if x == [1] else print("Not Spam")

# Accuracy score
y_pred = model.predict(cv.transform(x_test))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred)*100)
