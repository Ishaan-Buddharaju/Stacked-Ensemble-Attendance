# python -m pip install -U matplotlib
# pip install numpy
# pip install pandas
#pip install -U scikit-learn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

Data = pd.read_csv('StudentData.csv')
before_processed_shape = Data.shape[0]
Data = Data.dropna()
after_processed_shape = Data.shape[0]

DiscardedCount = open("KNNIndividualDiscardCount.txt", 'a')
DiscardedCount.write("Original Count/Shape:" + str(before_processed_shape))
DiscardedCount.write("\n\nNew Count/Shape:" + str(after_processed_shape))
DiscardedCount.close()

Data['#AP/Honors'] = Data['#Honors semesters'] + Data['#AP semesters']
Data = Data[['Student', 'Grade', 'GPA', '#AP/Honors', '#Honors semesters', '#AP semesters', 'Days Enrolled', 'Days Present', 'Att Rate']]
Data['Days Absent'] = Data['Days Enrolled'] - Data['Days Present']
Data = Data[['Student', 'Grade', 'GPA', '#AP/Honors', '#Honors semesters', '#AP semesters', 'Days Absent', 'Days Enrolled', 'Days Present', 'Att Rate']]
Data.drop(labels=['#Honors semesters', '#AP semesters','Days Present'], axis=1)

TrainingDF = Data.drop(labels=["Student"], axis=1)
X = TrainingDF[['Grade', 'GPA', '#AP/Honors', 'Days Absent', 'Days Enrolled']]
y = TrainingDF[['Att Rate']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 44, test_size = 0.25, shuffle= True)
y_train = y_train.values.ravel()

k_values = []
mae_scores = []

for x in range(3,50):
    KNN = KNeighborsRegressor(x)
    KNN.fit(X_train, y_train)
    KNN_y_pred = KNN.predict(X_test)
    KNN_MAE = mean_absolute_error(y_test, KNN_y_pred)
    k_values.append(x)
    mae_scores.append(KNN_MAE)


plt.plot(k_values, mae_scores, marker = 'o', linestyle='-')
plt.xlabel("K-value")
plt.ylabel('Mean Absolute Error (MAE)')
plt.title("K-value (3-50) v. Mean Absolute Error of KNN model")
plt.grid(True)
plt.savefig(fname= 'KNN Grid Search')

print('Done!')

