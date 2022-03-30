'''
@author: CSEMN(Mahmoud Nasser)
@since : 30 MAR 2022
'''
#Read Dataset from CSV file
import pandas as pd
col_names=['timeStamp','Gender','Grade','Age','Length','Weight','ShoesSize']
dataset = pd.read_csv("../human_features.csv",names= col_names,skiprows=(0,))
