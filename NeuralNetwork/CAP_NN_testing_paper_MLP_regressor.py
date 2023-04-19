'''
MLP classification is based on CAP_NN_paper.py?
normalized attendance is used as data output, Y

'''
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

#Load data
from NeuralNetwork.CAP_NN_testing_MLP import classifier

df = pd.read_csv(r'..\Data\output_lecture_seminar_processed3.csv', parse_dates=['date'])
X = df.drop(['Unnamed: 0', 'year', 'date','room_name', 'status', 'year','attendance', 'attendance_by_class', 'date-year', 'date-month', 'date-day','normalized_attendance','class_type_new',], axis=1)  #
y = df['normalized_attendance'] # same that is set as y = data_output in CAP_NN_paper.py .

#Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)


#MODEL
#Initializing the MLPRegressor

nn = MLPRegressor(
    activation='relu',
    hidden_layer_sizes=(10, 100),
    alpha=0.001,
    random_state=20,
    early_stopping=False)

#Train the model
nn.fit(X_train, y_train)

#PREDICT

#Comparing the predictions against the actual observations in y_val
#SCORE

# Make prediction
y_pred = nn.predict(X_test)
#
# Calculate accuracy and error metrics
#
test_set_rsquared = nn.score(X_test, y_test)
test_set_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#
# Print R_squared and RMSE value
#

print('R_squared value: ', test_set_rsquared)
print('RMSE: ', test_set_rmse)



