'''
MLP classification is based on CAP_NN_paper.py?
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
#########
import pandas
from sklearn import linear_model


df = pd.read_csv(r'..\Data\output_lecture_seminar_processed3.csv', parse_dates=['date'])

X = df.drop(['Unnamed: 0', 'year', 'date','room_name', 'year','attendance', 'attendance_by_class', 'date-year', 'date-month', 'date-day','normalized_attendance','class_type_new',], axis=1)  #

y = df['normalized_attendance'] # same that is set as y = data_output in CAP_NN_paper.py .

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70)

regr = linear_model.LinearRegression()
regr.fit(X, y)
##predict the attendance where the following parameters are set:
predictedAttendance = regr.predict([[8,2,2,0,2,16,0,4,2,452,1]])
print(predictedAttendance)