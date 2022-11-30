We preprocessed our input and output variable. The next step is to create the training set, for training the model and test set, for evaluating the test set. For this process, we are using a train test split.
````
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
````
# Model Training and Prediction
And we almost there, the model creation part. We are using the naive_bayes algorithm for our model creation. Later we are training the model using the training set.
`````
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)
``````
So we’ve trained our model using the training set. Now let’s predict the output for the test set.
````
y_pred = model.predict(x_test)

