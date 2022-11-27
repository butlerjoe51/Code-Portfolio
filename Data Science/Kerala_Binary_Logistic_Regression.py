# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Read from Dataset - We use the Kerala dataset in this scenario
df = pd.read_csv('https://raw.githubusercontent.com/amandp13/Flood-Prediction-Model/master/kerala.csv') #Rainfall dataset in Kerala from 1905 to 2018

# Explore dataset
df.info()

df.shape

df.describe()

df.corr()

# We need values of target output to be between 0 and 1. Replace YES and NO with 1 and 0
df['FLOODS'].replace(['YES', 'NO'], [1,0], inplace=True) # 1 is YES, 0 is NO
df.head(5)

# Feature Selection - pick the ones that contribute most to target output

#Define X and Y
X= df.iloc[:,1:14]   #all features
Y= df.iloc[:,-1]   #target output (floods)

# Select top 3 features
best_features= SelectKBest(score_func=chi2, k=3)
fit= best_features.fit(X,Y)

# Create Dataframes for features and scores of each feature
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)

# Combine features and scores into one dataframe
features_scores= pd.concat([df_columns, df_scores], axis=1)
features_scores.columns= ['Features', 'Score']
features_scores.sort_values(by = 'Score') # Higher score means more rain 

# Build the model - SEPT, JUN, JUL had the most rainfall
X= df[['SEP', 'JUN', 'JUL']]  #the top 3 features
Y= df[['FLOODS']]  #the target output

# Split into test and train
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.4,random_state=100)

# Create Logistic Regression Body and fit it
logreg= LogisticRegression()
logreg.fit(X_train,y_train)

# Predict using our model
y_pred=logreg.predict(X_test)
print (X_test) # test dataset
print (y_pred) # predicted values

# Evaulate model performance

# Method 1: Classification report
from sklearn import metrics
from sklearn.metrics import classification_report
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
print("Recall: ",metrics.recall_score(y_test, y_pred, zero_division=1))
print("Precision:",metrics.precision_score(y_test, y_pred, zero_division=1))
print("CL Report:",metrics.classification_report(y_test, y_pred, zero_division=1))

# Method 2: ROC Curve
y_pred_proba= logreg.predict_proba(X_test) [::,1] # Define metrics
false_positive_rate, true_positive_rate, _ = metrics.roc_curve(y_test, y_pred_proba) #Calculate true positive and false positive rates

auc= metrics.roc_auc_score(y_test, y_pred_proba) # Calculate AUC to see model performance

# Plot ROC Curve
plt.plot(false_positive_rate, true_positive_rate,label="AUC="+str(auc))
plt.title('ROC Curve')
plt.ylabel('True Positive Rate')
plt.xlabel('false Positive Rate')
plt.legend(loc=4) # AUC is ~0.94 => We did a good job!
