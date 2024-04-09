import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
import tensorflow as tf



data = pd.read_csv('heart.csv');
#print(data);

#---------------------------------------------------------------Data Cleaning----------------------------------------------------------------
#Removing Duplicates
data = data.drop_duplicates()

# Removing Missing Values
data = data.dropna()

#check if dataset is empty or not
if data.shape[0] == 0:
    print("Error: The datset is empty")
    exit()

# Remove outliers
target_column = 'target' if 'target' in data.columns else 'Disease'
if target_column in data.columns:
    features = data.drop(target_column, axis=1)
    z_scores = np.abs((features - features.mean()) / features.std())
    outliers_mask = (z_scores >= 3).any(axis=1)
    data_outliers = data[outliers_mask]
    data_no_outliers = data[~outliers_mask]
else:
    data_no_outliers = data
    data_outliers = pd.DataFrame()

# print(data)

