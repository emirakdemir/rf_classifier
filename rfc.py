# --Random Forest Classifier-- #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

df.head()
df.info()
df.shape
df.dtypes
df.tail()
df.isnull()
df.isnull().sum()
df.describe()
df.columns


# split data into input and target variable
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]





# Feature Scaling #
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.20, random_state=42)





# Create Classifier #
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)

# predictin on the test set
y_pred = classifier.predict(X_test)





# Calculate Model Accuracy #
"Accuracy:", accuracy_score(y_test, y_pred)
#  0.8

confusion_matrix(y_test,y_pred)
#    [[38  3]
#    [ 9 10]]

classification_report(y_test,y_pred)
#                     precision    recall  f1-score   support
#
#              0       0.81      0.93      0.86        41
#              1       0.77      0.53      0.62        19
#   
#       accuracy                           0.80        60
#      macro avg       0.79      0.73      0.74        60
#   weighted avg       0.80      0.80      0.79        60



feature_importances_df = pd.DataFrame({"feature": list(X.columns), "importance": classifier.feature_importances_}).sort_values("importance", ascending=False)

feature_importances_df
#                        feature  importance
#   11                      time       0.370
#   7           serum_creatinine       0.146
#   4          ejection_fraction       0.119
#   2   creatinine_phosphokinase       0.086
#   6                  platelets       0.078
#   0                        age       0.073
#   8               serum_sodium       0.065
#   5        high_blood_pressure       0.015
#   10                   smoking       0.014
#   9                        sex       0.012
#   1                    anaemia       0.012
#   3                   diabetes       0.010





# Viusalization Important Features #

sns.barplot(x=feature_importances_df.feature, y=feature_importances_df.importance)

plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Visualizing Important Features")
plt.xticks(rotation=45, horizontalalignment="right", fontweight="light", fontsize="x-large")
plt.show()