import numpy as np
import pandas as pd
import sns as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# load dataset
#pima = pd.read_csv('results_output_lecture_seminar_processed.csv')
#pima.head()
#df = pd.read_csv('results_output_lecture_seminar_processed_old.csv', parse_dates=['date'])
df = pd.read_csv(r'..\Data\output_lecture_seminar_processed.csv', parse_dates=['date'])

#df = pd.read_csv(r'\Data\output_lecture_seminar_processed.csv', parse_dates=['date'])
df.head()
#split dataset in features and target variable
'''
UPDATES
 feature_cols = attributes affecting the attendance, 11 attributes NOT including date parameters - day, month, year of date
'''

#
feature_cols = ['week', 'day', 'time_of_day', 'faculty','school','joint', 'status', 'degree', 'enrollment', 'class_duration']
X = df[feature_cols]
y = df['class_type'] # target, determines the attendance recorded as at is by university, seminar, lecture, other facilities

#X = pima[feature_cols] # Features
#y = pima.Outcome # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
# 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print(y_pred)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=7)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
# y_pred = clf.predict(X_test)
# print(y_pred)

# Model Accuracy, how often is the classifier correct?

print("Accuracy after Optimisation:",metrics.accuracy_score(y_test, y_pred))

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf,
                   feature_names=feature_cols,
                   class_names=['0','1','2'],
                   filled=True)
plt.savefig('CAttendance_ex1_tree.png', dpi=1050)

print("Confusion matrix\n", confusion_matrix(y_test, y_pred, labels = [0, 1, 2]))

df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print (df)


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))