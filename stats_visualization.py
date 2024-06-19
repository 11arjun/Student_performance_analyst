import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder

# Loading the data set
Data = read_csv('student-por.csv', sep=';')
pd.set_option('display.max_columns', None)  # Shows all columns
pd.set_option('display.max_rows', None)     # Shows all rows
pd.set_option('display.max_colwidth', None) # Shows full column content

# Checking if we have any missing data
missing_data = Data.isnull().sum()
# print(" \n The number of missing data \n", missing_data)

# Identifying nominal attributes
nominal_columns = Data.select_dtypes(include=['object']).columns
# print("Nominal Attributes:", nominal_columns)
# Converting boolean columns to integers
for column in Data.select_dtypes(include=['bool']).columns:
    Data[column] = Data[column].astype(int)

# Label encoding for nominal attributes, nominal to numerical
label_encoders = {}
for column in nominal_columns:
    le = LabelEncoder()
    Data[column] = le.fit_transform(Data[column])
    label_encoders[column] = le

# Print to verify nominal conversion
print("Data after label encoding nominal attributes:\n", Data.head())
# Computing the mean, median, and standard deviation for all attributes
mean_data = Data.mean()
median_data = Data.median()
std_dev_data = Data.std()
print("Mean for all attributes:\n", mean_data)
print("Median for all attributes:\n", median_data)
print("Standard Deviation for all attributes:\n", std_dev_data)
# computing the  first and third quartiles for all attributes
quartiles = Data.describe().loc[['25%', '75%']]
print("\nFirst and Third Quartiles for all attributes:\n", quartiles)
# Computing the range and variance for all attributes
# using the max() and min() methods to compute the range,
range_data = Data.max() - Data.min()
# Using var() method to compute the variance for all attributes.
variance_data = Data.var()
print("\nRange for all attributes:\n", range_data)
print("\nVariance for all attributes:\n", variance_data)