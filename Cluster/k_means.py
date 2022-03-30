'''
@author: CSEMN(Mahmoud Nasser)
@since : 30 MAR 2022
'''

import pandas as pd
import matplotlib.pyplot as plt

col_names=['timeStamp','Gender','Grade','Age','Length','Weight','ShoesSize']
dataset = pd.read_csv("../human_features.csv",names= col_names,skiprows=(0,))

#Pick Features and label
x = dataset.iloc[:,5].values  # weight
y = dataset.iloc[:, 4].values # length

#Visualize data
plt.scatter(x,y, label='True Position' )
plt.xlabel("Weight")
plt.ylabel("Lenght")
plt.show()
