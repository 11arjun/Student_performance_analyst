import pandas as pd
from pandas import read_csv

# Loading the data set
Data = read_csv('student-por.csv',sep=';')
pd.set_option('display.max_columns', None)  # Shows all columns
pd.set_option('display.max_rows', None)     # Shows all rows
pd.set_option('display.max_colwidth', None) # Shows full column content
# checking if we have any missing data
missing_data =  Data.isnull().sum();
print(" \n The no of missing data \n" , missing_data)
# We have no missing data so lets identify Nominal attributes
# Identifying nominal attributes
nominal_columns = Data.select_dtypes(include=['object']).columns
print("Nominal Attributes" , nominal_columns)
# setting the attributes with it's datas
nominal_datas = Data[nominal_columns]
print(" Nominal Datas \n ", nominal_datas)
# Converting Nominal attributes to numerical



