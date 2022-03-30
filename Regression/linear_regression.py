'''
@author: CSEMN(Mahmoud Nasser)
@since : 30 MAR 2022
'''
#Read Dataset from CSV file
import pandas as pd
col_names=['timeStamp','Gender','Grade','Age','Length','Weight','ShoesSize']
dataset = pd.read_csv("../human_features.csv",names= col_names,skiprows=(0,))

#Pick Features and label
x = dataset.iloc[:,[1,5,6]].values #gender,weight,ShoeSize
y = dataset.iloc[:, 4].values # length

#Encode Gender
from sklearn.preprocessing import LabelEncoder
gender_col=x[:,0]
genderEncoder = LabelEncoder()
genderEncoder.fit(gender_col[:2]) # give it different values of gender to fit
gender_encoded = genderEncoder.transform(gender_col) # encode
x[:,0]= gender_encoded # replace

# split data into train & test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0)

# Training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Making Predictions
y_pred = regressor.predict(x_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

#Evaluation 
from sklearn.metrics import mean_absolute_error,mean_squared_error
from math import sqrt
print('Mean Absolute Error    :', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error     :', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', sqrt(mean_squared_error(y_test, y_pred)))