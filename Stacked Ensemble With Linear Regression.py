from sklearn.model_selection import (train_test_split,GroupKFold, StratifiedGroupKFold, KFold)
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import numpy as np

Data = pd.read_csv('PseudoData.csv')  #Read in CSV

before_processed_shape = Data.shape[0]
Data = Data.dropna()
after_processed_shape = Data.shape[0]
DiscardedCount = open("StackedIndividualDiscardCount.txt", 'a')
DiscardedCount.write("Original Count/Shape:" + str(before_processed_shape))
DiscardedCount.write("\n\nNew Count/Shape:" + str(after_processed_shape))
DiscardedCount.close()

Data['#AP/Honors'] = Data['#Honors semesters'] + Data['#AP semesters']
Data = Data[['Student', 'Grade', 'GPA', '#AP/Honors', '#Honors semesters', '#AP semesters', 'Days Enrolled', 'Days Present', 'Att Rate']]
Data['Days Absent'] = Data['Days Enrolled'] - Data['Days Present']
Data = Data[['Student', 'Grade', 'GPA', '#AP/Honors', '#Honors semesters', '#AP semesters', 'Days Absent', 'Days Enrolled', 'Days Present', 'Att Rate']]
Data.drop(labels=['#Honors semesters', '#AP semesters','Days Present'], axis=1)

Features = ['Grade', 'GPA', '#AP/Honors', 'Days Absent', 'Days Enrolled']
Target = "Att Rate"

TrainingDF = Data.drop(labels=["Student"], axis=1)  #Change b4 push
X = TrainingDF[Features]
y = TrainingDF[Target]    

etr_r2_scores = []
etr_MAE_scores = []
etr_MSE_scores = []

KNN_r2_scores = []
KNN_MAE_scores = []
KNN_MSE_scores = []

GBR_r2_scores = []
GBR_MAE_scores = []
GBR_MSE_scores = []

Avg_r2_scores = []
Avg_MAE_scores = []
Avg_MSE_scores = []

lr_r2_scores = []
lr_MAE_scores = []
lr_MSE_scores = []

Kf = KFold(n_splits= 3, shuffle= True, random_state= 444) # set splits to 4 so that folds are proportional to class size
Kf2 = KFold(n_splits= 3, shuffle= True, random_state= 926)

for train_idx, test_idx in Kf.split(X, y):
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    # ETR Model
    etr = ExtraTreesRegressor(random_state=19)
    etr.fit(X_train, y_train)
    etr_y_pred = etr.predict(X_test)

    etr_r2_scores.append(r2_score(y_test, etr_y_pred))
    etr_MAE_scores.append(mean_absolute_error(y_test, etr_y_pred))
    etr_MSE_scores.append(mean_squared_error(y_test, etr_y_pred))

    #KNN Model
    KNN = KNeighborsRegressor(4)
    KNN.fit(X_train, y_train)
    KNN_y_pred = KNN.predict(X_test)

    KNN_r2_scores.append(r2_score(y_test, KNN_y_pred))
    KNN_MAE_scores.append(mean_absolute_error(y_test, KNN_y_pred))
    KNN_MSE_scores.append(mean_squared_error(y_test, etr_y_pred))

    #GBR Model
    GBR = GradientBoostingRegressor(loss = 'absolute_error', subsample=0.5)
    GBR.fit(X_train, y_train)
    GBR_y_pred = GBR.predict(X_test)
    
    GBR_r2_scores.append(r2_score(y_test, GBR_y_pred))
    GBR_MAE_scores.append(mean_absolute_error(y_test, GBR_y_pred))
    GBR_MSE_scores.append(mean_squared_error(y_test, GBR_y_pred))

    avg_y_pred = (etr_y_pred + KNN_y_pred + GBR_y_pred)/3
    Avg_r2_scores.append(r2_score(y_test, avg_y_pred))
    Avg_MAE_scores.append(mean_absolute_error(y_test, avg_y_pred))
    Avg_MSE_scores.append(mean_squared_error(y_test, avg_y_pred))

    #Layer two linear regression 
    Layer2Data = {'ETR Pred': etr_y_pred, 'KNN Pred': KNN_y_pred, 'GBR Pred': GBR_y_pred, 'True Att Rate': y_test} #Model will be trained on layer 2 test data so validate on training data
    Layer2DF = pd.DataFrame(Layer2Data)
    Features2 = ['ETR Pred', 'KNN Pred', 'GBR Pred']
    X2 = Layer2DF[Features2]
    Target2 = "True Att Rate"
    y2 = Layer2DF[Target2]
    
    for train_idx2, test_idx2 in Kf2.split(X2, y2):
        X_train2 = X2.iloc[train_idx2]
        y_train2 = y2.iloc[train_idx2]

        X_test2 = X2.iloc[test_idx2]
        y_test2 = y2.iloc[test_idx2]    
        lr = LinearRegression()
        lr.fit(X_train2, y_train2)
        lr_y_pred = lr.predict(X_test2)

        lr_r2_scores.append(r2_score(y_test2, lr_y_pred))
        lr_MAE_scores.append(mean_absolute_error(y_test2, lr_y_pred))
        lr_MSE_scores.append(mean_absolute_error(y_test2, lr_y_pred))

r2_etrFolds = sum(etr_r2_scores)/len(etr_r2_scores)
MAE_etrFolds = sum(etr_MAE_scores)/len(etr_MAE_scores)
MSE_etrFolds = sum(etr_MSE_scores)/len(etr_MSE_scores)

r2_KNNFolds = sum(KNN_r2_scores)/len(KNN_r2_scores)
MAE_KNNFolds = sum(KNN_MAE_scores)/len(KNN_MAE_scores)
MSE_KNNFolds = sum(KNN_MSE_scores)/len(KNN_MSE_scores)

r2_GBRFolds = sum(GBR_r2_scores)/len(GBR_r2_scores)
MAE_GBRFolds = sum(GBR_MAE_scores)/len(GBR_MAE_scores)
MSE_GBRFolds = sum(GBR_MSE_scores)/len(GBR_MSE_scores)

r2_AvgEnsemble = sum(Avg_r2_scores)/len(Avg_r2_scores)
MAE_AvgEnsemble = sum(Avg_MAE_scores)/len(Avg_MAE_scores)
MSE_AvgEnsemble = sum(Avg_MSE_scores)/len(Avg_MSE_scores)

r2_lrStackEnsemble = sum(lr_r2_scores)/len(lr_r2_scores)
MAE_lrStackEnsemble = sum(lr_MAE_scores)/len(lr_MAE_scores)
MSE_lrStackEnsemble = sum(lr_MSE_scores)/len(lr_MSE_scores)

data = {
    'Model Type': ['Extra Trees', 'KNN', 'Gradient Boosting', 'Averaged Ensemble', 'Stacked Ensemble w/ LR'],
    'r2': [r2_etrFolds, r2_KNNFolds, r2_GBRFolds, r2_AvgEnsemble, r2_lrStackEnsemble],
    'MAE': [MAE_etrFolds, MAE_KNNFolds, MAE_GBRFolds, MAE_AvgEnsemble, MAE_lrStackEnsemble],
    'MSE': [MSE_etrFolds, MSE_KNNFolds, MSE_GBRFolds, MSE_AvgEnsemble, MSE_lrStackEnsemble]
}

ModelMetrics = pd.DataFrame(data)

ModelMetrics.to_csv('CrossValidMetricsLR.csv')

print("Done!")