from sklearn.model_selection import (train_test_split, KFold, GridSearchCV)
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

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

grid = {
    'learning_rate': [0.0001, 0.001, 0.01, 0.05, 0.1, 1],  # eta
    'n_estimators': [10, 50, 100, 150],       # Number of trees
    'max_depth': [3, 5, 7, 9],                # Maximum depth of trees
    'min_samples_split': [2, 5, 10],       # Minimum samples for split
    'min_samples_leaf': [1, 2, 4],         # Minimum samples per leaf
    'subsample': [0.5, 0.7, 1.0],          # Subsample ratio
}

model = GradientBoostingRegressor()
CrossVal = KFold(n_splits= 3, shuffle= True, random_state= 444)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=1, cv=CrossVal)
grid_result = grid_search.fit(X_train, y_train)

best_result = grid_result.best_score_
best_params = grid_result.best_params_

Tuning_results = open("GBRTuningResults.txt", 'a')
Tuning_results.write("Best Results:" + str(best_result))
Tuning_results.write("\n\nNew Count/Shape:" + str(best_params))
Tuning_results.close()

print("Done!")