# Language-detection
using the [Language Detection dataset](https://www.kaggle.com/basilb2s/language-detection), which contains text details for 17 different languages.
About the Dataset
It's a small language detection dataset. This dataset consists of text details for 17 different languages, ie, you will be able to create an NLP model for predicting 17 different language..

Languages
1) English
2) Malayalam
3) Hindi
4) Tamil
5) Kannada
6) French
7) Spanish
8) Portuguese
9) Italian
10) Russian
11) Sweedish
12) Dutch
13) Arabic
14) Turkish
15) German
16) Danish
17) Greek

Using the text we have to create a model which will be able to predict the given language. This is a solution for many artificial intelligence applications and computational linguists. These kinds of prediction systems are widely used in electronic devices such as mobiles, laptops, etc for machine translation, and also on robots. It helps in tracking and identifying multilingual documents too. The domain of NLP is still a lively area of researchers.

### 22.Import all the required libraries, then import the language detection dataset
this dataset contains text details for 17 different languages. You can run the the value count for each language:
data["Language"].value_counts()
Output :
English       1385
French        1014
Spanish        819
Portugeese     739
Italian        698
Russian        692
Sweedish       676
Malayalam      594
Dutch          546
Arabic         536
Turkish        474
German         470
Tamil          469
Danish         428
Kannada        369
Greek          365
Hindi           63
Name: Language, dtype: int64

### Separating Independent and Dependent features
Now we can separate the dependent and independent variables, here text data is the independent variable and the language name is the dependent variable.
X = data["Text"]
y = data["Language"]

Introduction
Every Machine Learning enthusiast has a dream of building/working on a cool project, isn’t it? Mere understandings of the theory aren’t enough, you need to work on projects, try to deploy them, and learn from them. Moreover, working on specific domains like NLP gives you wide opportunities and problem statements to explore. Through this article, I wish to introduce you to an amazing project, the Language Detection model using Natural Language Processing. This will take you through a real-world example of ML(application to say). So, let’s not wait anymore.

 

About the dataset
We are using the Language Detection dataset, which contains text details for 17 different languages.

Languages are:

* English

* Portuguese

Loading Image
Recession Proof Your Career
Become a Data Science with 100% Job Guarantee
* French

* Greek

* Dutch

* Spanish

* Japanese

* Russian

* Danish

* Italian

* Turkish

* Swedish

* Arabic

* Malayalam

* Hindi

* Tamil

* Telugu

Using the text we have to create a model which will be able to predict the given language. This is a solution for many artificial intelligence applications and computational linguists. These kinds of prediction systems are widely used in electronic devices such as mobiles, laptops, etc for machine translation, and also on robots. It helps in tracking and identifying multilingual documents too. The domain of NLP is still a lively area of researchers.

 

Implementation
Importing libraries and dataset
So let’s get started. First of all, we will import all the required libraries.

import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
Now let’s import the language detection dataset


As I told you earlier this dataset contains text details for 17 different languages. So let’s count the value count for each language.

data["Language"].value_counts()
Output :

English       1385
French        1014
Spanish        819
Portugeese     739
Italian        698
Russian        692
Sweedish       676
Malayalam      594
Dutch          546
Arabic         536
Turkish        474
German         470
Tamil          469
Danish         428
Kannada        369
Greek          365
Hindi           63
Name: Language, dtype: int64
Separating Independent and Dependent features
Now we can separate the dependent and independent variables, here text data is the independent variable and the language name is the dependent variable.

X = data["Text"]
y = data["Language"]


### Label Encoding
Our output variable, the name of languages is a categorical variable. For training the model we should have to convert it into a numerical form, so we are performing label encoding on that output variable. For this process, we are importing LabelEncoder from sklearn.

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

### Text Preprocessing
This is a dataset created using scraping the Wikipedia, so it contains many unwanted symbols, numbers which will affect the quality of our model. So we should perform text preprocessing techniques.

#creating a list for appending the preprocessed text
data_list = []
#iterating through all the text
for text in X:
       #removing the symbols and numbers
        text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
        text = re.sub(r'[[]]', ' ', text)
        # converting the text to lower case
        text = text.lower()
        # appending to data_list
        data_list.append(text)

### Bag of Words
As we all know that, not only the output feature but also the input feature should be of the numerical form. So we are converting text into numerical form by creating a Bag of Words model using CountVectorizer.

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(data_list).toarray()
X.shape # (10337, 39419)


### Train Test Splitting
We preprocessed our input and output variable. The next step is to create the training set, for training the model and test set, for evaluating the test set. For this process, we are using a train test split.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

### Model Training and Prediction
And we almost there, the model creation part. We are using the naive_bayes algorithm for our model creation. Later we are training the model using the training set.

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)

So we’ve trained our model using the training set. Now let’s predict the output for the test set.

y_pred = model.predict(x_test)


### Model Evaluation
Now we can evaluate our model

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy is :",ac)
#Accuracy is : 0.9772727272727273

The accuracy of the model is 0.97 which is very good and our model is performing well. Now let’s plot the confusion matrix using the seaborn heatmap.

plt.figure(figsize=(15,10))
sns.heatmap(cm, annot = True)
plt.show()


### Predicting with some more data
Now let’s test the model prediction using text in different languages.

def predict(text):
     x = cv.transform([text]).toarray() # converting text to bag of words model (Vector)
     lang = model.predict(x) # predicting the language
     lang = le.inverse_transform(lang) # finding the language corresponding the the predicted value
     print("The langauge is in",lang[0]) # printing the language
     
     
