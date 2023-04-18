import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


df = pd.read_csv(r'..\Data\output_lecture_seminar_processed.csv', parse_dates=['date'])
df.head()
#split dataset in features and target variable
'''
UPDATES
 feature_cols = attributes affecting the attendance, 11 attributes NOT including date parameters - day, month, year of date
'''

#
feature_cols = ['week', 'day', 'time_of_day', 'faculty','school','joint', 'status', 'degree', 'enrollment', 'class_duration']
X = df[feature_cols]
y = df['room_name'] # target, determines the attendance recorded as at is by university, seminar, lecture, other facilities

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)


# Polynomial Kernel

svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print('Polynomial Kernel:')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Gaussian Kernel

svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print('Gaussian Kernel:')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Sigmoid Kernel

from sklearn.svm import SVC
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print('Sigmoid Kernel:')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Grid Search

param_grid = {'C':[0.1,1,10,100], 'gamma':[0.1,1,0.01,0.001]}
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose=3)
grid.fit(X_train, y_train)

pred_grid = grid.predict(X_test)

print('Grid Search:')
print(confusion_matrix(y_test, pred_grid))
print(classification_report(y_test, pred_grid))