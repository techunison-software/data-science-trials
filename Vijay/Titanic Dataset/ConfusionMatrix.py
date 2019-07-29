import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt  
import warnings
import matplotlib.cbook

# Load Dataset 
# Dataset can be found at: https://www.kaggle.com/uciml/sms-spam-collection-dataset

df = pd.read_csv('E:/LINEARREGRESSION/Vijay/Titanic Dataset/INPUT/spam.csv', encoding = 'latin-1' )

# Keep only necessary columns
df = df[['v2', 'v1']]

# Rename columns
df.columns = ['SMS', 'Type']
df.head()

# Let's process the text data 
# Instantiate count vectorizer 
countvec = CountVectorizer(ngram_range=(1,4), 
                           stop_words='english',  
                           strip_accents='unicode', 
                           max_features=1000)

X = df.SMS.values
y = df.Type.values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state = 0)

# Instantiate classifier
mnb = MultinomialNB()

# Create bag of words
X_train = countvec.fit_transform(X_train)
X_test = countvec.transform(X_test)

# Train the classifier/Fit the model
mnb.fit(X_train, y_train)

# Make predictions
y_pred = mnb.predict(X_test)

# Build confusion metrics
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

print ('Confusion Matrix :')
print(cm) 
print ('Accuracy Score :',accuracy_score(y_true=y_test, y_pred=y_pred)) 
print ('Classification Report : ')
print (classification_report(y_true=y_test, y_pred=y_pred))


# cm = array([[1414,   20],[  17,  221]], dtype=int64)
# Plot confusion matrix in a beautiful manner
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g') #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('top') 
ax.xaxis.set_ticklabels(['ham', 'spam'], fontsize = 15)
ax.xaxis.tick_top()

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(['spam', 'ham'], fontsize = 15)
plt.show()





















