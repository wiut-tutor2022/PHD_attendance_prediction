# MLP classification and label encoding for room sorting based
# on popular room on demand closer to the date,
'''
It will help to monitor popularity of the rooms and determine more efficient use of resources. It means instead of using
large lecture rooms smaller seminar rooms can be used for delivering seminars and other events.


Attendance sorted by Class type from 1 - 5, not present in UNSW but suggested by observed attendance

and predicted
https://michael-fuchs-python.netlify.app/2019/10/31/introduction-to-logistic-regression/#model-evaluation
https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
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


#df2 = pd.read_csv('2017_sem2_attendance_data_12_weeks_23_march.csv')
#df2 = pd.read_csv('attendance_pattern_fig5.csv')
#df2 = pd.read_csv('2017_sem2_attendance_data.csv')
#df = pd.read_csv('2017_sem2_attendance_data.csv')

df = pd.read_csv(r'..\Data\output_lecture_seminar_processed.csv', parse_dates=['date'])


# df2.fillna(0, inplace=True)
# print("Dataframe")
# print(df2.shape)

#figure 4 from paper, attendance pattern of three courses across weeks is represented below


#attendace and enrollment vs time
X = df.drop(['date', "class_type",'class_type_new'], axis=1)
#y = df['attendance']
y= df['class_type_new'] # attendance sorted by possible class size and thus suggest this kind of class

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70)

# X_real = df2['time']#timeslot, week
# y_real = df2['attendance', 'enrollment']
# y_real = df2['attendance']
# model = MLPClassifier(max_iter=150)
# model.fit(X_train, y_train)


#Importing MLPClassifier

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

#MODEL
#Initializing the MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(50,100,50), max_iter=1500,activation = 'relu',solver='adam',random_state=1)
#classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=1500,activation = 'relu',solver='adam',random_state=1)

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
print(classification_report(y_test, y_pred, target_names=['Class 1', 'Class 2', 'Class 3','Class 4', 'Class 5']))

cm_df = pd.DataFrame(cm,
                     index = ['Class 1', 'Class 2', 'Class 3','Class 4', 'Class 5'],
                     columns = ['Class 1', 'Class 2', 'Class 3','Class 4', 'Class 5'])

plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()