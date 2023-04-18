'''
MLP classification is based on CAP_NN_new.py?
normalized attendance is used as data output, Y

'''
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
#import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from math import sqrt
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'..\Data\output_lecture_seminar_processed3.csv', parse_dates=['date'])

X = df.drop(['Unnamed: 0', 'date', 'year','attendance', 'attendance_by_class','normalized_attendance', 'class_type_new'], axis=1)  #

y = df['attendance_by_class'] # same that is set as y = data_output in CAP_NN_new.py .

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70)

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

#Importing MLPClassifier

#MODEL
#Initializing the MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(50,100,50), max_iter=1500,activation = 'relu',solver='adam',random_state=1)
classifier.fit(X_train, y_train)

#PREDICT
y_pred = classifier.predict(X_test)

#Comparing the predictions against the actual observations in y_val
#SCORE
accuracy = metrics.accuracy_score (y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)

#Printing the accuracy
print("Accuracy of MLPClassifier : ", accuracy)
print("confusion matrix\n", cm)
print(classification_report(y_test, y_pred, target_names=['Not attended 0', 'Attended 1', ]))


cm_df = pd.DataFrame(cm,
                     index = ['Class 1', 'Class 2', ],
                     columns = ['Class 1', 'Class 2', ])

plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()