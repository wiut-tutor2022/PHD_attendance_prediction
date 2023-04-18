# Attendance and enrollment against timeslots (time)
'''
representation of stats, not real
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
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



#df2 = pd.read_csv('2017_sem2_attendance_data_12_weeks_23_march.csv')
# attendance, enrollment vs time

df2 = pd.read_csv(r'..\Data\attendance_pattern_fig5.csv')
#df2 = pd.read_csv('2017_sem2_attendance_data.csv')
#df = pd.read_csv('results_output.csv')
#df = pd.read_csv('2017_sem2_attendance_data.csv')


df2.fillna(0, inplace=True)
print("Dataframe")
print(df2.shape)

#figure 4 from paper, attendance and enrollment against timeslots (time)

X = df2['time'] #timeslot, week
y = df2['attendance'] #attendance for course-1
y2 = df2['enrollment']
X2 = df2['time']

# X2 = df2['week']
# y2 = df2['course-2']
# print("test")
#
'''
Case 1 - course 1 - several modules today year long 9-11 - 
Case 2 - 

100 students attend 4 modules per day
several modules 
'''
# X3 = df2['week']
# y3 = df2['course-3']

#attendace and enrollment vs time
# X = df.drop(['date', 'attendance', ], axis=1)
# y = df['attendance']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70)

# X_real = df2['time']#timeslot, week
# y_real = df2['attendance', 'enrollment']
# y_real = df2['attendance']

# model = MLPRegressor()
# model.fit(X_train, y_train)

#mlp = MLPClassifier(hidden_layer_sizes=(20,), activation='logistic', solver='adam', max_iter=1500)
# X_train1 = np.squeeze(X_train)
# y_train1 = np.squeeze(y_train)
# X_test1 = np.squeeze(X_test)
# y_test1 = np.squeeze(y_test)
#mlp.fit(X_train1, y_train1)


# print(model)
# expected_y  = y_test
# print("printing expectations")
# print(expected_y)
# predicted_y = model.predict(X_test)


# predict_train = model.predict(X_train)
# predict_test = model.predict(X_test)
#score = np.sqrt(mean_absolute_error(y_train1, y_test))
#print("The Mean Absolute Error of our Model is {}".format(round(score, 2)))
# print("R2", metrics.r2_score(expected_y, predicted_y))
#############NEW

################################################################################################################
#figure 4 from paper, attendance pattern of three courses across weeks is represented below (imitated below)

plt.plot(X, y,color='blue') #The X  - Features vs. The predicted label
plt.plot(X, y,color='blue') #The X- Features vs. The predicted label
plt.plot(X, y,color='red')
plt.scatter(x=X, y=y,color='black')  #The X-Features vs. The Real Label

#plt.plot(X2, y2,color='blue') #The X- Features vs. The predicted label
#plt.plot(X2, y2,color='blue') #The X- Features vs. The predicted label
# plt.plot(y2,color='orange')
# plt.scatter(x=X2, y=y2,color='black')
#
# plt.plot(X3, y3,color='blue')
# plt.scatter(x=X3, y=y3,color='black')

#plt.plot(X2, y2,color='blue') #The X- Features vs. The predicted label
plt.plot(X2, y2,color='purple') #The X- Features vs. The predicted label

plt.title('Attendance and enrollment against timeslots (time)')
plt.xlabel('Timeslots (time)')
plt.ylabel('Attendance and enrollment')

plt.show()#To show your figures code here
plt.figure(figsize=(10, 10))
plt.ion()
###################################################################################################################

# plt.scatter(x=X_features_main, y=y_label_main,color='black')  #The X-Features vs. The Real Label
# plt.plot(X_features_main, y_predicted_from_X_features_main,color='blue') #The X- Features vs. The predicted label
# plt.show()#To show your figures code here

#from https://stackoverflow.com/questions/58410187/how-to-plot-predicted-values-vs-the-true-value
# #######
# sns.regplot(expected_y, predicted_y, fit_reg=True, scatter_kws={"s": 5})

#sns.regplot(expected_y, fit_reg=True, scatter_kws={"s": 5})
#sns.regplot(expected_y, fit_reg=True)

#######3

#source: https://www.projectpro.io/recipes/use-mlp-classifier-and-regressor-in-python
