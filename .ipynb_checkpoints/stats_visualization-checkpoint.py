import numpy as np
import pandas as pd
from pandas import read_csv
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

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
# Computing Mean
mean_data = Data.mean()
# Computing Median
median_data = Data.median()
# Computing Standard Deviation
std_dev_data = Data.std()
# print("Mean for all attributes:\n", mean_data)
# print("Median for all attributes:\n", median_data)
# print("Standard Deviation for all attributes:\n", std_dev_data)
mean_data.to_csv('mean_data.csv', index=True)
print("The Mean has been written")
median_data.to_csv('median_data.csv', index=True)
print("The Median  has been written")
std_dev_data.to_csv('std_dev_data.csv', index=True)
print("The  Standard Deviation has been written")
# computing the  first and third quartiles for all attributes
quartiles = Data.describe().loc[['25%', '75%']]
quartiles.to_csv('quartiles.csv',index=True)
print("The Quartiles has been written")
print("\nFirst and Third Quartiles for all attributes:\n", quartiles)
# Computing the range and variance for all attributes
# using the max() and min() methods to compute the range,
range_data = Data.max() - Data.min()
range_data.to_csv('range.csv',index = True)
print("The Range has been written")
# Using var() method to compute the variance for all attributes.
variance_data = Data.var()
variance_data.to_csv('Variance.csv',index=True)
print("The Variance has been written")
# Displaying Range and Variance .
print("\nRange for all attributes:\n", range_data)
print("\nVariance for all attributes:\n", variance_data)
# calculating AAD , as per required slide formula, calculating mean of the column and computing mean of the AAD
def calculate_aad(column):
    mean_val = column.mean()
    aad = (column - mean_val).abs().mean()
    return  aad
# calculating  MAD , as per required slides formula, calculating the mean of the column and computing the median.
def calculate_mad(column):
    mean_val = column.mean()
    mad = np.median(np.abs(column - mean_val))
    return mad
# Initializing two empty dictionaries aad_values and mad_values to store the  AAD and MAD for each  attributes
aad_values = {}
mad_values = {}
# Starting for loop to iterate over each column in the dataset
for column in Data.select_dtypes(include=[np.number]).columns:
# This will call the  calculate_add and calculate _mad  passing the current column as argument
    aad_values[column] = calculate_aad(Data[column])
    mad_values[column] = calculate_mad(Data[column])
#
aad_df = pd.DataFrame(list(aad_values.items()), columns=['Attribute', 'AAD'])
mad_df = pd.DataFrame(list(mad_values.items()), columns=['Attribute', 'MAD'])
aad_df.to_csv('ADD.csv',index=True)
print("The ADD has been written")
mad_df.to_csv('MAD.csv' , index=True)
print("The MAD has been written")
# print("\nAAD for all numerical attributes:\n", aad_values)
# print("\nMAD for all numerical attributes:\n", mad_values)
# Histograms for each attribute
for column in Data.columns:
    plt.figure(figsize=(10, 6))
    Data[column].hist(bins=20, edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.savefig(f'histogram_{column}.png')
    plt.close()
# Box Plots for each attribute
for column in Data.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=Data[column])
    plt.title(f'Box Plot of {column}')
    plt.ylabel(column)
    plt.savefig(f'boxplot_{column}.png')
    plt.close()
# Scatter Plots for each pair of attributes
for i, column1 in enumerate(Data.columns):
    for column2 in Data.columns[i+1:]:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=Data[column1], y=Data[column2])
        plt.title(f'Scatter Plot of {column1} vs. {column2}')
        plt.xlabel(column1)
        plt.ylabel(column2)
        plt.savefig(f'scatterplot_{column1}_vs_{column2}.png')
        plt.close()
# Contour Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(x=Data['age'], y=Data['G3'], fill=True)
plt.title('Contour Plot of Age vs. Final Grade (G3)')
plt.savefig('contourplot_age_g3.png')
plt.close()

# Matrix Plot (Correlation Heatmap)
plt.figure(figsize=(12, 8))
sns.heatmap(Data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.savefig('correlation_matrix_heatmap.png')
plt.close()

# Pair Plot (Correlation Plot)
sns.pairplot(Data)
plt.suptitle('Pair Plot of All Attributes', y=1.02)
plt.savefig('pairplot_all_attributes.png')
plt.close()

# Parallel Coordinates Plot
plt.figure(figsize=(12, 8))
parallel_coordinates(Data, 'sex', color=('#556270', '#4ECDC4'))
plt.title('Parallel Coordinates Plot')
plt.savefig('parallel_coordinates_plot.png')
plt.close()
# 2D Histogram
plt.figure(figsize=(10, 6))
plt.hist2d(Data['age'], Data['G3'], bins=30, cmap='Blues')
plt.colorbar()
plt.title('2D Histogram of Age vs. Final Grade (G3)')
plt.xlabel('Age')
plt.ylabel('Final Grade (G3)')
plt.savefig('2d_histogram_age_g3.png')
plt.close()
