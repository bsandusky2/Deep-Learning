import pandas as pd
import matplotlib.pyplot as plt
from mlp import MLP

#Read in data
pima = pd.read_csv("C:/Users/sandu/Desktop/Python Projects/diabetes.csv")

#Show subplots
newMLP = MLP(pima)
plt.subplots(3,3,figsize = (15,15))
newMLP.createsubplots(pima)
plt.subplot(3,3,9).set_visible(False)
plt.show()

#Preprocess Data
pima_scale = MLP.imputemissing(self = MLP, data = pima)

#Build train/test sets
split = MLP.splitdata(self = MLP,data = pima_scale)

#Build MLP
deeplearn = MLP.buildann(self = MLP,datalist = split)
#Print trainign error
print(deeplearn[0])
#Print testing error
print(deeplearn[1])

#Build confustion matrix
MLP.confmatrix(self= MLP, list = deeplearn,datalist = split)
plt.show

MLP.ROC(self=MLP, list = deeplearn,datalist = split)
plt.show()
