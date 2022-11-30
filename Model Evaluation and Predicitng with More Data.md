````
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy is :",ac)
# Accuracy is : 0.9772727272727273
`````

The accuracy of the model is 0.97 which is very good and our model is performing well. Now let’s plot the confusion matrix using the seaborn heatmap.

````
plt.figure(figsize=(15,10))
sns.heatmap(cm, annot = True)
plt.show()
`````
# Predicting with some more data
Now let’s test the model prediction using text in different languages.
````
def predict(text):
     x = cv.transform([text]).toarray() # converting text to bag of words model (Vector)
     lang = model.predict(x) # predicting the language
     lang = le.inverse_transform(lang) # finding the language corresponding the the predicted value
     print("The langauge is in",lang[0]) # printing the language
