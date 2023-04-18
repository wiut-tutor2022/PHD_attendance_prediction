#updates in 5.0 version
import csv
from datetime import datetime
import datetime
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import category_encoders
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

le = LabelEncoder()

# # written by Olga
# def season(month):
#     # season function
#     if month in [0, 1, 2]:
#         return 0  # winter
#     elif month in [3, 4, 5]:
#         return 1  # spring
#     elif month in [6, 7, 8]:
#         return 2  # summer
#     elif month in [9, 10, 11]:
#         return 3  # winter
#
#
# #
# df['date:season'] = df['date:month'].apply(season)
# #to encode column season
# df['date:season'] = le.fit_transform(df['date:season'])

# print(df)
# print(df.head())
# df.to_csv("results3.csv")  # output to csv

##########################################
# to predict attendance neet the csv with attendance
df = pd.read_csv(r'..\Data\output_lecture_seminar_processed3.csv', parse_dates=['date'])

#################################
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
scaler = MinMaxScaler()
time_components = ['year', 'month', 'day']
df[time_components] = scaler.fit_transform(df[time_components])

###################################
data_input = df.drop(['Unnamed: 0', 'date', 'month', 'year','attendance', 'attendance_by_class','normalized_attendance'], axis=1)  #

print("after drop shape and head")
print(data_input.head)

############

print("Data input")
print(data_input.shape)
print("data_output")


data_output = df[['attendance_by_class']].T
print(data_output.shape)

#f#######################################

# for data 2 RealAttendance data is removed and predicted data needs to be added
#data_2 = df.drop(['date','attendance', 'year', 'month','day'], axis=1)

########### for playing with date############
# df['date'] = pd.to_datetime(df['date'], errors='coerce') #important for assigning data
# df['date:month'] = df['date'].dt.month
# df['date:day_of_week'] = df['date'].dt.day_of_week # correct DAY of the week are displayed
# df['date:weekofyear'] = df['date'].dt.weekofyear
# df['year_encoded'] = le.fit_transform(df['year'])
# df['month_encoded'] = le.fit_transform(df['month'])
# df['day_encoded'] = le.fit_transform(df['date:day'])
# df["month_encoded"] = df["date"].dt.month

# #important for assigning time-of-day
# df['time_of_day'] = pd.to_datetime(df['time_of_day'], errors='coerce') #important for assigning data
# df['time_of_day']=df['time_of_day'].apply(time_of_day)
# df['time_of_day']=le.fit_transform(df['time_of_day'])
# df['time_of_day'] = df['time_of_day'].dt.time_of_day
#

#
# df['day:numbered'] = df['day'].apply(day_of_week)
# df['day:numbered'] = le.fit_transform(df['day'])#not correctly calculated

################
##########################################

class NeuralNetwork():
    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)

        # converting weights to a 16 by 1 matrix with values [-1;1) (we need [-1;1])
        self.synaptic_weights = 2 * np.random.random((16, 1)) - 1

    def sigmoid(self, x):
        # the sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # derivative to the Sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        # training the model to make accurate predictions while adjusting weights continually
        print("we got inside train", training_iterations)
        for iteration in range(training_iterations):
            # siphon the training data via  the neuron
            output = self.think(training_inputs)

            print("output", output)
            # computing error rate for back-propagation
            error = training_outputs - output
            print("error", output)


            # performing weight adjustments

            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            print("adjustments", adjustments)

            self.synaptic_weights += adjustments

    def think(self, inputs):

        inputs = inputs.astype(int or str)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output



if __name__ == "__main__":
    # initializing the neuron class
    neural_network = NeuralNetwork()

    print("Beginning Randomly Generated Weights: (synaptic weights) ")
    print(neural_network.synaptic_weights, len(neural_network.synaptic_weights), neural_network.synaptic_weights.shape)

    print(data_input.shape)
    training_inputs = np.array(data_input)
    training_outputs = np.array(data_output).T
    print("training output")
    # training taking place
    neural_network.train(training_inputs, training_outputs, 1500)
    # user_input_one = str(input("P1: "))
    # user_input_two = str(input("P2: "))
    # user_input_three = str(input("P3: "))
    # user_input_four = str(input("P4: "))
    # user_input_five = str(input("P5: "))
    # user_input_six = str(input("L1: "))
    # user_input_seven = str(input("L2: "))
    # user_input_eight = str(input("A1: "))
    # user_input_nine = str(input("A2: "))
    # user_input_ten = str(input("A3: "))
    # user_input_eleven = str(input("F1: "))
    # user_input_twelve = str(input("F2: "))
    # user_input_thirteen = str(input("F3: "))
    # user_input_fourteen = str(input("F4: "))
    # user_input_fourteen = str(input("SA1: "))
    # user_input_fourteen = str(input("SA2: "))


    with open(r'..\Data\output_lecture_seminar_processed.csv') as csvfile:  # Testing Input Data
        reader = csv.DictReader(csvfile)
        for row in reader:
            # print(row['date'])
            #date = datetime.datetime(row['date'])
            week = int(row['week'])
            day = int(row['day'])
            time_of_day = int(row['time_of_day'])
            class_type = int(row['class_type'])
            faculty = int(row['faculty'])
            semester = str(row['semester'])
            school = int(row['school'])
            joint = int(row['joint'])
            status = int(row['status'])
            degree = int(row['degree'])
            enrollment = int(row['enrollment'])
            class_duration = int(row['class_duration'])  # time variable
            room_name = int(row['room_name'])
            date_year = int(row['date-year'])
            date_month = int(row['date-month'])
            date_day = int(row['date-day'])

print("Attributes predicting attendance")

print("Class type(lecture/laboratory/tutorial)*: ", class_type, )
print("Course is combined or with other courses or not*", joint, )
print("Faculty*: ", faculty, )
print("Enrollment*", enrollment, )
print("School* ", school, )
print("Degree* ", degree, )
print("Status* (open or full) ", status, )
# print("Room name: ", room_name, )
print("Time of the day: ", time_of_day, )
print("Day", day, )
print("Week ", week, )
print("Enrollment", enrollment)
#print("Attendance", attendance)
print( "Date-year", date_year )
print( "Date-month", date_month )
print( "Date-day", date_day )

final_result = neural_network.think(np.array(
    [week,
     # date,
     day, room_name, time_of_day, class_type, faculty, school, joint, status, degree, enrollment, class_duration, date_year, date_month, date_day
     ])),
final_result = np.round(final_result, 5)
print("The Result: ", final_result)

#writing result of neural network to csv file. this result is not used anywhere

