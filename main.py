import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
import keras
from tensorflow.keras import Sequential
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

#load data
pima = pd.read_csv("C:/Users/sandu/Desktop/Python Projects/diabetes.csv")

# print first couple rows
print(pima.head())

# get histograms
pima.hist()
plt.show()

plt.subplots(3,3,figsize = (15,15))

for idx, col in enumerate(pima.columns):
    ax = plt.subplot(3,3,idx+1)
    ax.yaxis.set_ticklabels([])
    sns.distplot(pima.loc[pima.Outcome == 0][col],hist = False,axlabel=False,
                 kde_kws={
                     'linestyle':'-',
                     'color': 'black',
                     'label': 'No Diabetes'
                 })
    sns.distplot(pima.loc[pima.Outcome == 1][col],hist = False,axlabel=False,
                 kde_kws={
                     'linestyle':'-',
                     'color': 'black',
                     'label': 'Diabetes'
                 })
plt.subplot(3,3,9).set_visible(False)
plt.show()

#Check for missing values in the data
print(pima.isnull().any()) #looks like none, but some colums have minimums of 0 which isn't possible
print(pima.describe())

for col in pima.columns:
    missing_rows = pima.loc[pima[col]==0].shape[0]
    print(col + " : " + str(missing_rows))

#Replace missing values with NaNs, so python knows these are missing values
pima['Glucose'] = pima['Glucose'].replace(0,np.nan)
pima['BloodPressure'] = pima['BloodPressure'].replace(0,np.nan)
pima['SkinThickness'] = pima['SkinThickness'].replace(0,np.nan)
pima['Insulin'] = pima['Insulin'].replace(0,np.nan)
pima['BMI'] = pima['BMI'].replace(0,np.nan)

#Check to make sure replacements worked
for col in pima.columns:
    missing_rows = pima.loc[pima[col]==0].shape[0]
    print(col + " : " + str(missing_rows))

#impute missing values using mean
pima['Glucose'] = pima['Glucose'].fillna(pima['Glucose'].mean())
pima['BloodPressure'] = pima['BloodPressure'].fillna(pima['BloodPressure'].mean())
pima['SkinThickness'] = pima['SkinThickness'].fillna(pima['SkinThickness'].mean())
pima['Insulin'] = pima['Insulin'].fillna(pima['Insulin'].mean())
pima['BMI'] = pima['BMI'].fillna(pima['BMI'].mean())

#Scale Data
pima_scale = preprocessing.scale(pima)
pima_scale = pd.DataFrame(pima_scale,columns = pima.columns)
pima_scale['Outcome'] = pima['Outcome']

#Split data into training and testing sets
from sklearn.model_selection import train_test_split

x = pima_scale.loc[:,pima_scale.columns != 'Outcome']
y = pima_scale.loc[:,'Outcome']

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size= .2)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size= .2)

#Build Model Architecture
model = Sequential()
model.add(Dense(32,activation= "relu", input_dim=8))
model.add(Dense(16,activation= "relu"))
model.add(Dense(1,activation= "sigmoid"))
model.compile(optimizer= "adam", metrics = ["accuracy"], loss= ["binary_crossentropy"] )

#Fit model on training data
model.fit(X_train,y_train, epochs = 200)

#evaluate model on train data
scores = model.evaluate(X_train, y_train)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))

#Evaluate on test data
scores = model.evaluate(X_test, y_test)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))

#Create confusion matrix
y_test_pred = model.predict_classes(X_test)
c_matrix = confusion_matrix(y_test, y_test_pred)

#Create heatmap of matrix
ax = sns.heatmap(
    c_matrix,
    annot = True,
    xticklabels = ['No Diabetes','Diabetes'],
    yticklabels = ['No Diabetes','Diabetes'],
    cbar = False,
    cmap = 'Blues'
)
plt.show()

y_test_pred_probs = model.predict(X_test)
FPR,TPR, _ = roc_curve(y_test,y_test_pred_probs)
plt.plot(FPR,TPR)
plt.plot([0,1],[0,1], '--',color = 'black')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Pos Rate')
plt.show()



