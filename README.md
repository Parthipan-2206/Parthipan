# Parthipan
#AI-based-diabetes-prediction-system
DATA PREPROCESSING:
%matplotlib inline import pandas as pd import numpy as np
import matplotlib.pyplot as plt
import seaborn as snsdiabetes = pd.read_csv('datasets/diabetes.csv') diabetes.columns
Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'], dtype='object')
diabetes.head()
print("Diabetes data set dimensions : {}".format(diabetes.shape))
diabetes.groupby('Outcome').size()
diabetes.isnull().sum() diabetes.isna().sum()
print("Total : ", diabetes[diabetes.BloodPressure == 0].shape[0])Total : 35print(diabetes[diabetes.BloodPressure == 0].groupby('Outcome')['Age'].count())
print("Total : ", diabetes[diabetes.Glucose == 0].shape[0])Total : 5print(diabetes[diabetes.Glucose
== 0].groupby('Outcome')['Age'].count())
print("Total : ", diabetes[diabetes.SkinThickness == 0].shape[0])Total : 227print(diabetes[diabetes.SkinThickness == 0].groupby('Outcome')['Age'].count())
print("Total : ", diabetes[diabetes.SkinThickness == 0].shape[0])Total : 227print(diabetes[diabetes.SkinThickness == 0].groupby('Outcome')['Age'].count())feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']X = diabetes_mod[feature_names]
y = diabetes_ mod.
. neighbors import KNeighborsClassifier from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression from sklearn.tree import DecisionTreeClassifier from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier from sklearn.ensemble import GradientBoostingClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = diabetes_mod.Outcome, random_state=0)
names = []
scores = []for name, model in models: model.fit(X_train, y_train) y_pred = model.predict(X_test)
scores.append(accuracy_score(y_test, y_pred)) names.append(name)tr_split = pd.DataFrame({'Name': names,
'Score': scores}) print(tr_split)
K-FOLD CROSS VALIDATION:
names = []
scores = []for name, model in models:

kfold = KFold(n_splits=10, random_state=10) score = cross_val_score(model, X, y, cv=kfold,
scoring='accuracy').mean()

names.append(name)
scores.append(score)kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
print(kf_cross_val)
dataset_new = dataset
dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN) 
dataset_new["Glucose"].fillna(dataset_new["Glucose"].mean(), inplace = True)
dataset_new["BloodPressure"].fillna(dataset_new["BloodPressure"].mean(), inplace = True)
dataset_new["SkinThickness"].fillna(dataset_new["SkinThickness"].mean(), inplace = True)
dataset_new["Insulin"].fillna(dataset_new["Insulin"].mean(), inplace = True)
dataset_new["BMI"].fillna(dataset_new["BMI"].mean(), inplace = True)
y = dataset_new['Outcome']
X = dataset_new.drop('Outcome', axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.20, random_state = 42, stratify = dataset_new['Outcome'] )

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)
y_predict = model.predict(X_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)
y_predict = model.predict(X_test)
# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_predict)
cm

array([[84, 16],
       [25, 29]])

# Heatmap of Confusion matrix
sns.heatmap(pd.DataFrame(cm), annot=True)
from sklearn.metrics import accuracy_score
accuracy =accuracy_score(Y_test, y_predict)
accuracy

0.7337662337662337
y_predict = model.predict([[1,148,72,35,79.799,33.6,0.627,50]])
print(y_predict)
if y_predict==1:
    print("Diabetic")
else:
    print("Non Diabetic")
[1]
Diabetic
/opt/conda/lib/python3.10/site-packages/sklearn/base.py:439: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names
  warnings.warn(

Import Required Libraries:
# Ignore warning messages to prevent them from being displayed during code execution
import warnings
warnings.filterwarnings('ignore')

import numpy as np    # Importing the NumPy library for linear algebra operations
import pandas as pd   # Importing the Pandas library for data processing and CSV file handling

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import seaborn as sns                   # Importing the Seaborn library for statistical data visualization
import matplotlib.pyplot as plt         # Importing the Matplotlib library for creating plots and visualizations
import plotly.express as px             # Importing the Plotly Express library for interactive visualizations

Load and Prepare Data:
df=pd.read_csv('/kaggle/input/diabetes-data-set/diabetes.csv')

UnderStanding the Variables:
df.head(10)
df.describe()
Data Visualization:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df' is your DataFrame containing the dataset
# If you haven't imported your dataset yet, import it here

# Create subplots
f, ax = plt.subplots(1, 2, figsize=(10, 5))

# Pie chart for Outcome distribution
df['Outcome'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Outcome')
ax[0].set_ylabel(' ')

# Count plot for Outcome distribution
sns.countplot(x='Outcome', data=df, ax=ax[1])  # Use 'x' instead of 'Outcome'
ax[1].set_title('Outcome')

# Count plot for Outcome distribution
sns.countplot(x='Outcome', data=df, ax=ax[1])  # Use 'x' instead of 'Outcome'
ax[1].set_title('Outcome')

# Display class distribution
N, P = df['Outcome'].value_counts()
print('Negative (0):', N)
print('Positive (1):', P)
# Adding grid and showing plots
plt.grid()
plt.show()
Negative (0): 500
Positive (1): 268
Scatter Plot:
from pandas.plotting import scatter_matrix
scatter_matrix(df, figsize =(20, 20))
array([[<Axes: xlabel='Pregnancies', ylabel='Pregnancies'>,
        <Axes: xlabel='Glucose', ylabel='Pregnancies'>,
        <Axes: xlabel='BloodPressure', ylabel='Pregnancies'>,
        <Axes: xlabel='SkinThickness', ylabel='Pregnancies'>,
        <Axes: xlabel='Insulin', ylabel='Pregnancies'>,
        <Axes: xlabel='BMI', ylabel='Pregnancies'>,
        <Axes: xlabel='DiabetesPedigreeFunction', ylabel='Pregnancies'>,
        <Axes: xlabel='Age', ylabel='Pregnancies'>,
        <Axes: xlabel='Outcome', ylabel='Pregnancies'>],
       [<Axes: xlabel='Pregnancies', ylabel='Glucose'>,
        <Axes: xlabel='Glucose', ylabel='Glucose'>,
        <Axes: xlabel='BloodPressure', ylabel='Glucose'>,
        <Axes: xlabel='SkinThickness', ylabel='Glucose'>,
        <Axes: xlabel='Insulin', ylabel='Glucose'>,
        <Axes: xlabel='BMI', ylabel='Glucose'>,
        <Axes: xlabel='DiabetesPedigreeFunction', ylabel='Glucose'>,
        <Axes: xlabel='Age', ylabel='Glucose'>,
        <Axes: xlabel='Outcome', ylabel='Glucose'>],
       [<Axes: xlabel='Pregnancies', ylabel='BloodPressure'>,
        <Axes: xlabel='Glucose', ylabel='BloodPressure'>,
        <Axes: xlabel='BloodPressure', ylabel='BloodPressure'>,
        <Axes: xlabel='SkinThickness', ylabel='BloodPressure'>,
        <Axes: xlabel='Insulin', ylabel='BloodPressure'>,
        <Axes: xlabel='BMI', ylabel='BloodPressure'>,
        <Axes: xlabel='DiabetesPedigreeFunction', ylabel='BloodPressure'>,
        <Axes: xlabel='Age', ylabel='BloodPressure'>,
        <Axes: xlabel='Outcome', ylabel='BloodPressure'>],
       [<Axes: xlabel='Pregnancies', ylabel='SkinThickness'>,
        <Axes: xlabel='Glucose', ylabel='SkinThickness'>,
        <Axes: xlabel='BloodPressure', ylabel='SkinThickness'>,
        <Axes: xlabel='SkinThickness', ylabel='SkinThickness'>,
        <Axes: xlabel='Insulin', ylabel='SkinThickness'>,
        <Axes: xlabel='BMI', ylabel='SkinThickness'>,
        <Axes: xlabel='DiabetesPedigreeFunction', ylabel='SkinThickness'>,
        <Axes: xlabel='Age', ylabel='SkinThickness'>,
        <Axes: xlabel='Outcome', ylabel='SkinThickness'>],
       [<Axes: xlabel='Pregnancies', ylabel='Insulin'>,
        <Axes: xlabel='Glucose', ylabel='Insulin'>,
        <Axes: xlabel='BloodPressure', ylabel='Insulin'>,
        <Axes: xlabel='SkinThickness', ylabel='Insulin'>,
        <Axes: xlabel='Insulin', ylabel='Insulin'>,
        <Axes: xlabel='BMI', ylabel='Insulin'>,
        <Axes: xlabel='DiabetesPedigreeFunction', ylabel='Insulin'>,
        <Axes: xlabel='Age', ylabel='Insulin'>,
        <Axes: xlabel='Outcome', ylabel='Insulin'>],
       [<Axes: xlabel='Pregnancies', ylabel='BMI'>,
        <Axes: xlabel='Glucose', ylabel='BMI'>,
        <Axes: xlabel='BloodPressure', ylabel='BMI'>,
        <Axes: xlabel='SkinThickness', ylabel='BMI'>,
        <Axes: xlabel='Insulin', ylabel='BMI'>,
        <Axes: xlabel='BMI', ylabel='BMI'>,
        <Axes: xlabel='DiabetesPedigreeFunction', ylabel='BMI'>,
        <Axes: xlabel='Age', ylabel='BMI'>,
        <Axes: xlabel='Outcome', ylabel='BMI'>],
       [<Axes: xlabel='Pregnancies', ylabel='DiabetesPedigreeFunction'>,
        <Axes: xlabel='Glucose', ylabel='DiabetesPedigreeFunction'>,
        <Axes: xlabel='BloodPressure', ylabel='DiabetesPedigreeFunction'>,
        <Axes: xlabel='SkinThickness', ylabel='DiabetesPedigreeFunction'>,
        <Axes: xlabel='Insulin', ylabel='DiabetesPedigreeFunction'>,
        <Axes: xlabel='BMI', ylabel='DiabetesPedigreeFunction'>,
        <Axes: xlabel='DiabetesPedigreeFunction', ylabel='DiabetesPedigreeFunction'>,
        <Axes: xlabel='Age', ylabel='DiabetesPedigreeFunction'>,
        <Axes: xlabel='Outcome', ylabel='DiabetesPedigreeFunction'>],
       [<Axes: xlabel='Pregnancies', ylabel='Age'>,
        <Axes: xlabel='Glucose', ylabel='Age'>,
        <Axes: xlabel='BloodPressure', ylabel='Age'>,
        <Axes: xlabel='SkinThickness', ylabel='Age'>,
        <Axes: xlabel='Insulin', ylabel='Age'>,
        <Axes: xlabel='BMI', ylabel='Age'>,
        <Axes: xlabel='DiabetesPedigreeFunction', ylabel='Age'>,
        <Axes: xlabel='Age', ylabel='Age'>,
<Axes: xlabel='Outcome', ylabel='Age'>],
       [<Axes: xlabel='Pregnancies', ylabel='Outcome'>,
        <Axes: xlabel='Glucose', ylabel='Outcome'>,
        <Axes: xlabel='BloodPressure', ylabel='Outcome'>,
        <Axes: xlabel='SkinThickness', ylabel='Outcome'>,
        <Axes: xlabel='Insulin', ylabel='Outcome'>,
        <Axes: xlabel='BMI', ylabel='Outcome'>,
        <Axes: xlabel='DiabetesPedigreeFunction', ylabel='Outcome'>,
        <Axes: xlabel='Age', ylabel='Outcome'>,
        <Axes: xlabel='Outcome', ylabel='Outcome'>]], dtype=object)
        Pair plot:
sns.pairplot(data=df, hue='Outcome')
plt.show()

plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap='Reds')
plt.plot()
linkcode
# For Decesion Tree:
print("Train Accuracy of Decesion Tree: ", dt.score(X_train, y_train)*100)
print("Accuracy (Test) Score of Decesion Tree: ", dt.score(X_test, y_test)*100)
print("Accuracy Score of Decesion Tree: ", accuracy_score(y_test, dt_pred)*100)

Train Accuracy of Decesion Tree:  100.0
Accuracy (Test) Score of Decesion Tree:  80.51948051948052
Accuracy Score of Decesion Tree:  80.51948051948052

from sklearn.metrics import precision_score
print("Precision Score is: ", precision_score(y_test, lr_pred)*100)
print("Micro Average Precision Score is: ", precision_score(y_test, lr_pred, average='micro')*100)
print("Macro Average Precision Score is: ", precision_score(y_test, lr_pred, average='macro')*100)
print("Weighted Average Precision Score is: ", precision_score(y_test, lr_pred, average='weighted')*100)
print("precision Score on Non Weighted score is: ", precision_score(y_test, lr_pred, average=None)*100)
Precision Score is:  75.0
Micro Average Precision Score is:  77.27272727272727
Macro Average Precision Score is:  76.5909090909091
Weighted Average Precision Score is:  77.00413223140497
precision Score on Non Weighted score is:  [78.18181818 75.      

print('Classification Report of Logistic Regression: \n', classification_report(y_test, lr_pred, digits=4))

Classification Report of Logistic Regression: 
               precision    recall  f1-score   support

           0     0.7818    0.8866    0.8309        97
           1     0.7500    0.5789    0.6535        57

    accuracy                         0.7727       154
   macro avg     0.7659    0.7328    0.7422       154
weighted avg     0.7700    0.7727    0.7652       154

True Positive Rate(TPR)

Recall = True Positive/True Positive + False Negative
Recall = TP/TP+FN
In [65]:
recall_score = TP/ float(TP+FN)*100
print('recall_score', recall_score)
recall_score 57.89473684210527
In [66]:
TP, FN
Out[66]:
(33, 24)

from sklearn.metrics import recall_score
print('Recall or Sensitivity_Score: ', recall_score(y_test, lr_pred)*100)
Recall or Sensitivity_Score:  57.89473684210527
In [69]:
linkcode
print("recall Score is: ", recall_score(y_test, lr_pred)*100)
print("Micro Average recall Score is: ", recall_score(y_test, lr_pred, average='micro')*100)
print("Macro Average recall Score is: ", recall_score(y_test, lr_pred, average='macro')*100)
print("Weighted Average recall Score is: ", recall_score(y_test, lr_pred, average='weighted')*100)
recall Score is:  57.89473684210527
Micro Average recall Score is:  77.27272727272727
Macro Average recall Score is:  73.27726532826912
Weighted Average recall Score is:  77.27272727272727
recall Score on Non Weighted score is:  [88.65979381 57.89473684]
print('Classification Report of Logistic Regression: \n', classification_report(y_test, lr_pred, digits=4))

Classification Report of Logistic Regression: 
               precision    recall  f1-score   support

           0     0.7818    0.8866    0.8309        97
           1     0.7500    0.5789    0.6535        57

    accuracy                         0.7727       154
   macro avg     0.7659    0.7328    0.7422       154
weighted avg     0.7700    0.7727    0.7652       154

ROC Curve& ROC AUC
# Area under Curve:
auc= roc_auc_score(y_test, lr_pred)
print("ROC AUC SCORE of logistic Regression is ", auc)
ROC AUC SCORE of logistic Regression is  0.7327726532826913
linkcode:
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, lr_pred)
plt.plot(fpr, tpr, color='orange', label="ROC")
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristics (ROC) Curve of Logistic Regression")
plt.legend()
plt.grid()
plt.show()
%matplotlib inline

# Start Python Imports
import math, time, datetime import random as rd
# Data Manipulation import numpy as np import pandas as pd

# Visualization
import matplotlib.pyplot as plt import missingno as msno import seaborn as sns plt.style.use('seaborn-whitegrid')

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

# Machine learning
import catboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier from sklearn.neighbors import KNeighborsClassifier from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool, cv

# Let's be rebels and ignore warnings for now import warnings warnings.filterwarnings('ignore')
HELPING FUNCTIONS:
def systematic_sample(df, size): length = len(df)
interval = length // size rd.seed(None)
first = rd.randint(0, interval)
indexes = np.arange(first, length, step = interval) return df.iloc[indexes]
def missing_values_table(df):
# Total missing values
mis_val = df.isnull().sum()

# Percentage of missing values
mis_val_percent = 100 * df.isnull().sum() / len(df)

# Make a table with the results
mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

# Rename the columns
mis_val_table_ren_columns = mis_val_table.rename( columns = {0 : 'Missing Values', 1 : '% of Total Values'})

# Sort the table by percentage of missing descending
mis_val_table_ren_columns = mis_val_table_ren_columns[ mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
'% of Total Values', ascending=False).round(1)

# Print some summary information
print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n" "There are " + str(mis_val_table_ren_columns.shape[0]) +
" columns that have missing values.")

# Return the dataframe with missing information
return mis_val_table_ren_columns def fill_na(df):
for col in df.columns:
if df[col].isnull().any():
if df[col].dtypes in ["float", "int"]: df[col].fillna(df[col].mean(), inplace=True)
else:
df[col].fillna(df[col].mode()[0], inplace=True)
def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20, 5)
, use_bin_df=False):
"""
Function to plot counts and distributions of a label variable and target variable side by side.
::param_data:: = target dataframe
::param_bin_df:: = binned dataframe for countplot
::param_label_column:: = binary labelled column
param_target_column:: = column you want to view counts and distributions
::param_figsize:: = size of figure (width, height)
::param_use_bin_df:: = whether or not to use the bin_df, default False """
if use_bin_df:
fig = plt.figure(figsize=figsize) plt.subplot(1, 2, 1)
sns.countplot(y=target_column, data=bin_df); plt.subplot(1, 2, 2)
sns.distplot(data.loc[data[label_column] == 1][target_column], kde_kws={"label": "Survived"});
sns.distplot(data.loc[data[label_column] == 0][target_column], kde_kws={"label": "Did not survive"});
else:
fig = plt.figure(figsize=figsize) plt.subplot(1, 2, 1) sns.countplot(y=target_column, data=data); plt.subplot(1, 2, 2)
sns.distplot(data.loc[data[label_column] == 1][target_column], kde_kws={"label": "Survived"});
sns.distplot(data.loc[data[label_column] == 0][target_column], kde_kws={"label": "Did not survive"});
# Function that runs the requested algorithm and returns the accuracy metrics
def fit_ml_algo(algo, X_train, y_train, cv):

# One Pass
model = algo.fit(X_train, y_train)
acc = round(model.score(X_train, y_train) * 100, 2)

# Cross Validation
train_pred = model_selection.cross_val_predict(algo,
X_train, y_train, cv=cv, n_jobs = -1)
# Cross-validation accuracy metric
acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)
return train_pred, acc, acc_cv linkcode
# Feature Importance
def feature_importance(model, data):
"""
Function to show which features are most important in the model.
::param_model:: Which model to use?
::param_data:: What data to use? """
fea_imp = pd.DataFrame({'imp': model.feature_importances_, 'col': data.columns}) fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
_ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))

return fea_imp
#plt.savefig('catboost_feature_importance.png')

# Visualise the Fare bin counts as well as the Fare distribution versus Survived.
plot_count_dist(data=df,
bin_df=df_dis, label_column='Outcome', target_column='BloodPressure', figsize=(20,10), use_bin_df=True)
# Visualise the Fare bin counts as well as the Fare distribution versus Survived.
plot_count_dist(data=df,
bin_df=df_dis, label_column='Outcome', target_column='SkinThickness',

figsize=(20,10), use_bin_df=True)
# Visualise the Fare bin counts as well as the Fare distribution versus Survived.
plot_count_dist(data=df,
bin_df=df_dis, label_column='Outcome', target_column='Insulin', figsize=(20,10), use_bin_df=True)
# Visualise the Fare bin counts as well as the Fare distribution versus Survived.
plot_count_dist(data=df,
bin_df=df_dis, label_column='Outcome',
target_column='BMI', figsize=(20,10), use_bin_df=True)# Visualise the Fare bin counts as well as the Fare distribution versus Survived.
plot_count_dist(data=df,
bin_df=df_dis, label_column='Outcome',
target_column='DiabetesPedigreeFunction', figsize=(20,10),
use_bin_df=True)
Split the dataframe into data and labels
X_train = selected_df.drop('Outcome', axis=1) # data
y_train = selected_df.Outcome # labels
# Shape of the data (without labels)
X_train.shape

(768, 1254)
linkcode X_train.head()
# Shape of the labels
y_train.shape
(768,)
# Logistic Regression
start_time = time.time()
train_pred_log, acc_log, acc_cv_log = fit_ml_algo(LogisticRegression(),
X_train, y_train,
10)
log_time = (time.time() - start_time) print("Accuracy: %s" % acc_log) print("Accuracy CV 10-Fold: %s" % acc_cv_log)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))
Accuracy: 96.09
Accuracy CV 10-Fold: 67.45 Running Time: 0:00:02.451250 # k-Nearest Neighbours start_time = time.time()
train_pred_knn, acc_knn, acc_cv_knn = fit_ml_algo(KNeighborsClassifier(),
X_train, y_train, 10)
knn_time = (time.time() - start_time) print("Accuracy: %s" % acc_knn) print("Accuracy CV 10-Fold: %s" % acc_cv_knn)
print("Running Time: %s" % datetime.timedelta(seconds=knn_time))
Accuracy: 75.26
Accuracy CV 10-Fold: 66.8 Running Time: 0:00:00.358875

# Gaussian Naive Bayes
start_time = time.time()
train_pred_gaussian, acc_gaussian, acc_cv_gaussian = fit_ml_algo(GaussianNB(),
X_train, y_train, 10)
gaussian_time = (time.time() - start_time) print("Accuracy: %s" % acc_gaussian) print("Accuracy CV 10-Fold: %s" % acc_cv_gaussian)

print("Running Time: %s" % datetime.timedelta(seconds=gaussian_time)) Accuracy: 94.79
Accuracy CV 10-Fold: 59.9
Running Time: 0:00:00.219412
# Linear SVC
start_time = time.time()
train_pred_svc, acc_linear_svc, acc_cv_linear_svc = fit_ml_algo(LinearSVC(),
X_train, y_train, 10)
linear_svc_time = (time.time() - start_time) print("Accuracy: %s" % acc_linear_svc) print("Accuracy CV 10-Fold: %s" % acc_cv_linear_svc)
print("Running Time: %s" % datetime.timedelta(seconds=linear_svc_time))
Accuracy: 100.0
Accuracy CV 10-Fold: 65.23 Running Time: 0:00:00.268296 # Stochastic Gradient Descent start_time = time.time()
train_pred_sgd, acc_sgd, acc_cv_sgd = fit_ml_algo(SGDClassifier(), X_train,
y_train, 10)
sgd_time = (time.time() - start_time) print("Accuracy: %s" % acc_sgd) print("Accuracy CV 10-Fold: %s" % acc_cv_sgd)
print("Running Time: %s" % datetime.timedelta(seconds=sgd_time))
Accuracy: 100.0
Accuracy CV 10-Fold: 63.93 Running Time: 0:00:00.437200 # Decision Tree Classifier start_time = time.time()
train_pred_dt, acc_dt, acc_cv_dt = fit_ml_algo(DecisionTreeClassifier(),
X_train, y_train,
)
dt_time = (time.time() - start_time) print("Accuracy: %s" % acc_dt) print("Accuracy CV 10-Fold: %s" % acc_cv_dt)
print("Running Time: %s" % datetime.timedelta(seconds=dt_time))
Accuracy: 100.0
Accuracy CV 10-Fold: 61.85 Running Time: 0:00:00.596504 # Gradient Boosting Trees start_time = time.time()
train_pred_gbt, acc_gbt, acc_cv_gbt = fit_ml_algo(GradientBoostingClassifier(),
X_train, y_train, 10)

gbt_time = (time.time() - start_time) print("Accuracy: %s" % acc_gbt) print("Accuracy CV 10-Fold: %s" % acc_cv_gbt)
print("Running Time: %s" % datetime.timedelta(seconds=gbt_time))
Accuracy: 81.9
Accuracy CV 10-Fold: 63.54 Running Time: 0:00:06.448709
# Define the categorical features for the CatBoost model cat_features = np.where(X_train.dtypes != np.float)[0] cat_features
array([	0,	1,	2, ..., 1251, 1252, 1253])
# Use the CatBoost Pool() function to pool together the training data and categori cal feature labels
train_pool = Pool(X_train,
y_train, cat_features)
linkcode
# CatBoost model definition

catboost_model = CatBoostClassifier(iterations=1000,
custom_loss=['Accuracy'], loss_function='Logloss')

# Fit CatBoost model
catboost_model.fit(train_pool,
plot=True)

# CatBoost accuracy
acc_catboost = round(catboost_model.score(X_train, y_train) * 100, 2)

print("---CatBoost Metrics---") print("Accuracy: {}".format(acc_catboost))
print("Accuracy cross-validation 10-Fold: {}".format(acc_cv_catboost)) print("Running Time: {}".format(datetime.timedelta(seconds=catboost_time)))
---CatBoost Metrics--- Accuracy: 81.64
Accuracy cross-validation 10-Fold: 66.54 Running Time: 0:00:59.138368
linkcode
models = pd.DataFrame({
'Model': ['KNN', 'Logistic Regression', 'Naive Bayes', 'Stochastic Gradient Decent', 'Linear SVC', 'Decision Tree', 'Gradient Boosting Trees', 'CatBoost'],
'Score': [ acc_knn, acc_log, acc_gaussian, acc_sgd, acc_linear_svc, acc_dt, acc_gbt,

acc_catboost
]})
print("---Reuglar Accuracy Scores---") models.sort_values(by='Score', ascending=False)
cv_models = pd.DataFrame({
'Model': ['KNN', 'Logistic Regression', 'Naive Bayes', 'Stochastic Gradient Decent', 'Linear SVC', 'Decision Tree', 'Gradient Boosting Trees', 'CatBoost'],
'Score': [ acc_cv_knn, acc_cv_log, acc_cv_gaussian, acc_cv_sgd, acc_cv_linear_svc, acc_cv_dt, acc_cv_gbt, acc_cv_catboost
]})
print('---Cross-validation Accuracy Scores---')
cv_models.sort_values(by='Score', ascending=False)
# Plot the feature importance scores
feature_importance(catboost_model, X_train)
