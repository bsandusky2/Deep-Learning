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
from sklearn.model_selection import train_test_split

class MLP:
    def __init__(self, data):
        self.data = data

    def createsubplots(self, data):
        for idx, col in enumerate(data.columns):
            ax = plt.subplot(3, 3, idx + 1)
            ax.yaxis.set_ticklabels([])
            sns.distplot(data.loc[data.Outcome == 0][col], hist=False, axlabel=False,
                         kde_kws={
                             'linestyle': '-',
                             'color': 'black',
                             'label': 'No Diabetes'
                         })
            sns.distplot(data.loc[data.Outcome == 1][col], hist=False, axlabel=False,
                         kde_kws={
                             'linestyle': '-',
                             'color': 'black',
                             'label': 'Diabetes'
                         })

    def imputemissing(self, data):
        data['Glucose'] = data['Glucose'].replace(0, np.nan)
        data['BloodPressure'] = data['BloodPressure'].replace(0, np.nan)
        data['SkinThickness'] = data['SkinThickness'].replace(0, np.nan)
        data['Insulin'] = data['Insulin'].replace(0, np.nan)
        data['BMI'] = data['BMI'].replace(0, np.nan)
        data['Glucose'] = data['Glucose'].fillna(data['Glucose'].mean())
        data['BloodPressure'] = data['BloodPressure'].fillna(data['BloodPressure'].mean())
        data['SkinThickness'] = data['SkinThickness'].fillna(data['SkinThickness'].mean())
        data['Insulin'] = data['Insulin'].fillna(data['Insulin'].mean())
        data['BMI'] = data['BMI'].fillna(data['BMI'].mean())
        data_scale = preprocessing.scale(data)
        data_scale = pd.DataFrame(data_scale, columns=data.columns)
        data_scale['Outcome'] = data['Outcome']
        return data_scale

    def splitdata(self,data):
        x = data.loc[:, data.columns != 'Outcome']
        y = data.loc[:, 'Outcome']
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2)
        df = [X_train,y_train,X_val,y_val,X_test,y_test]
        return df

    def buildann(self, datalist):
        # Build Model Architecture
        model = Sequential()
        model.add(Dense(32, activation="relu", input_dim=8))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", metrics=["accuracy"], loss=["binary_crossentropy"])
        # Fit model on training data
        model.fit(datalist[0], datalist[1], epochs=200)
        # evaluate model on train data
        trainscores = model.evaluate(datalist[0], datalist[1])
        # Evaluate on test data
        testscores = model.evaluate(datalist[4], datalist[5])
        trainacc = trainscores[1] * 100
        testacc = testscores[1] * 100
        return [trainacc, testacc, model]

    def confmatrix(self, list,datalist):
        y_test_pred = list[2].predict_classes(datalist[4])
        c_matrix = confusion_matrix(datalist[5], y_test_pred)

        # Create heatmap of matrix
        ax = sns.heatmap(
            c_matrix,
            annot=True,
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'],
            cbar=False,
            cmap='Blues'
        )
        return ax

    def ROC(self, list,datalist):
        y_test_pred_probs = list[2].predict(datalist[4])
        FPR, TPR, _ = roc_curve(datalist[5], y_test_pred_probs)
        plt.plot(FPR, TPR)
        plt.plot([0, 1], [0, 1], '--', color='black')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Pos Rate')


