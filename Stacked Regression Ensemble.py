#pip install numpy
#pip install pandas
#pip install -U scikit-learn

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

Data = pd.read_csv('StudentData.csv')

Data['#AP/Honors'] = Data['#Honors semesters'] + Data['#AP semesters']
Data = Data[['Student', 'Grade', 'GPA', '#AP/Honors', '#Honors semesters', '#AP semesters', 'Days Enrolled', 'Days Present', 'Att Rate']]
Data['Days Absent'] = Data['Days Enrolled'] - Data['Days Present']
Data = Data[['Student', 'Grade', 'GPA', '#AP/Honors', '#Honors semesters', '#AP semesters', 'Days Absent', 'Days Enrolled', 'Days Present', 'Att Rate']]
Data.drop(labels=['#Honors semesters', '#AP semesters','Days Present'], axis=1)

TrainingDF = Data.drop(labels=["Student"], axis=1)
X = TrainingDF[['Grade', 'GPA', '#AP/Honors', 'Days Absent', 'Days Enrolled']]
y = TrainingDF[['Att Rate']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 44, test_size = 0.25, shuffle= True)


etr = ExtraTreesRegressor(random_state=19) #Extra Trees
etr.fit(X_train, y_train)

etr_y_pred = etr.predict(X_test)
etr_r2 = r2_score(y_test, etr_y_pred)
etr_MAE = mean_absolute_error(y_test, etr_y_pred)
etr_MSE = mean_squared_error(y_test, etr_y_pred)

KNN = KNeighborsRegressor(3) #KNN
KNN.fit(X_train, y_train)

KNN_y_pred = KNN.predict(X_test)
KNN_r2 = r2_score(y_test, KNN_y_pred)
KNN_MAE = mean_absolute_error(y_test, KNN_y_pred)
KNN_MSE = mean_squared_error(y_test, KNN_y_pred)

GBR = GradientBoostingRegressor(loss = 'absolute_error', subsample=0.5) # Gradient Boosting
GBR.fit(X_train, y_train)

GBR_y_pred = GBR.predict(X_test)
GBR_r2 = r2_score(y_test, GBR_y_pred)
GBR_MAE = mean_absolute_error(y_test, GBR_y_pred)
GBR_MSE = mean_squared_error(y_test, GBR_y_pred)

#Averaging layer 2

Layer_1_preds = pd.DataFrame({'Extra Trees Y predictions': etr_y_pred.flatten(), 'KNN Y predictions': KNN_y_pred.flatten(), 'GBR Y predictions': GBR_y_pred.flatten()})
y_pred_avgs = Layer_1_preds.mean(axis=1)
Averaged_r2 = r2_score(y_test, y_pred_avgs)
Averaged_MAE = mean_absolute_error(y_test, y_pred_avgs)
Averaged_MSE = mean_squared_error(y_test, y_pred_avgs)

ErrorMetric_df = pd.DataFrame(data=[[etr_r2, etr_MAE, etr_MSE], 
                                    [KNN_r2, KNN_MAE, KNN_MSE], 
                                    [GBR_r2, GBR_MAE, GBR_MSE], 
                                    [Averaged_r2, Averaged_MAE, Averaged_MSE]], 
                                    index=['Extra Trees', 'KNN', 'Gradient Boosting', 'Layer 2 Averaging'], 
                                    columns=["R^2", "MAE", "MSQ"])

ErrorMetric_df.to_csv('Error Metrics.csv')

print('Done!')