import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# pip install seaborn
# python -m pip install -U pip
# python -m pip install -U matplotlib
# pip install numpy

Data = pd.read_csv('StudentData.csv')

Data['#AP/Honors'] = Data['#Honors semesters'] + Data['#AP semesters']
Data = Data[['Student', 'Grade', 'GPA', '#AP/Honors', '#Honors semesters', '#AP semesters', 'Days Enrolled', 'Days Present', 'Att Rate']]
Data['Days Absent'] = Data['Days Enrolled'] - Data['Days Present']
Data = Data[['Student', 'Grade', 'GPA', '#AP/Honors', '#Honors semesters', '#AP semesters', 'Days Absent', 'Days Enrolled', 'Days Present', 'Att Rate']]
Data.drop(labels=['#Honors semesters', '#AP semesters','Days Present'], axis=1)

CorrelationDF = Data.drop(labels=["Student"], axis=1).corr()

plt.figure(figsize=(16,6))
FeatureHeatMap = sns.heatmap(CorrelationDF.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
FeatureHeatMap.set_title('Feature Correlation Heatmap')
plt.savefig(fname= 'Feature Correlation Heatmap')