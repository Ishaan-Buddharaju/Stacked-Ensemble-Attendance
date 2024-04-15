import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# python -m pip install -U pip
# python -m pip install -U matplotlib
# pip install numpy
# pip install pandas
# pip install seaborn


Data = pd.read_csv('PseudoData.csv')

before_processed_shape = Data.shape[0]
Data = Data.dropna()
after_processed_shape = Data.shape[0]

DiscardedCount = open("HeatMapIndividualDiscardCount.txt", 'a')
DiscardedCount.write("Original Count/Shape:" + str(before_processed_shape))
DiscardedCount.write("\n\nNew Count/Shape:" + str(after_processed_shape))
DiscardedCount.close()

Data['#AP/Honors'] = Data['#Honors semesters'] + Data['#AP semesters']
Data = Data[['Student', 'Grade', 'GPA', '#AP/Honors', '#Honors semesters', '#AP semesters', 'Days Enrolled', 'Days Present', 'Att Rate']]
Data['Days Absent'] = Data['Days Enrolled'] - Data['Days Present']
Data = Data[['Student', 'Grade', 'GPA', '#AP/Honors', '#Honors semesters', '#AP semesters', 'Days Absent', 'Days Enrolled', 'Days Present', 'Att Rate']]
Data = Data.drop(labels=['#Honors semesters', '#AP semesters','Days Present'], axis=1)

CorrelationDF = Data.drop(labels=["Student", "Att Rate"], axis=1).corr()

plt.figure(figsize=(20,12))
FeatureHeatMap = sns.heatmap(CorrelationDF.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG', annot_kws={'size': 20})
FeatureHeatMap.set_title('Feature Correlation Heatmap', fontsize=28)
plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.savefig(fname= 'Feature Correlation Heatmap')


print('Done!')