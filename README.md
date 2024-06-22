 Arjun Shrestha 
# Data science  
# Dataset  : https://archive.ics.uci.edu/dataset/320/student+performance
# Requirements
 Pycharm
- Python 3.x
- pandas
- numpy
- scikit-learn
- import pandas as pd
- import numpy as np
- from sklearn.metrics.pairwise import manhattan_distances
- from sklearn.preprocessing import LabelEncoder
- import matplotlib.pyplot as plt
- import seaborn as sns

# Loading the data set
Data = read_csv('student-por.csv', sep=';')
pd.set_option('display.max_columns', None)  # Shows all columns
pd.set_option('display.max_rows', None)     # Shows all rows
pd.set_option('display.max_colwidth', None) # Shows full column content
# Getting Data info
data_info = Data.info()
print(data_info)
# Identifying nominal attributes
nominal_columns = Data.select_dtypes(include=['object']).columns
print("Nominal Attributes:", nominal_columns)
#  Converting boolean columns to integers
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
print("Mean for all attributes:\n", mean_data)
print("Median for all attributes:\n", median_data)
print("Standard Deviation for all attributes:\n", std_dev_data)
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
#  calculating AAD , as per required slide formula, calculating mean of the column and computing mean of the AAD
def calculate_aad(column):
    mean_val = column.mean()
    aad = (column - mean_val).abs().mean()
    return  aad
#  calculating  MAD , as per required slides formula, calculating the mean of the column and computing the median.
def calculate_mad(column):
    mean_val = column.mean()
    mad = np.median(np.abs(column - mean_val))
    return mad
#  Initializing two empty dictionaries aad_values and mad_values to store the  AAD and MAD for each  attributes
aad_values = {}
mad_values = {}
#  Starting for loop to iterate over each column in the dataset
for column in Data.select_dtypes(include=[np.number]).columns:
# This will call the  calculate_add and calculate _mad  passing the current column as argument
    aad_values[column] = calculate_aad(Data[column])
    mad_values[column] = calculate_mad(Data[column])
aad_df = pd.DataFrame(list(aad_values.items()), columns=['Attribute', 'AAD'])
mad_df = pd.DataFrame(list(mad_values.items()), columns=['Attribute', 'MAD'])
aad_df.to_csv('ADD.csv',index=True)
print("The ADD has been written")
mad_df.to_csv('MAD.csv' , index=True)
print("The MAD has been written")
print("\nAAD for all numerical attributes:\n", aad_values)
print("\nMAD for all numerical attributes:\n", mad_values)
# Visualizing histograms for specific columns to analyze
specific_columns = ['absences', 'freetime', 'studytime','traveltime','sex','age','G1','G2','G3','health']
column_scatter = [('absences','freetime'),('studytime','traveltime'),('sex','age'),('G1','G2'),('G3','health')]
for column in specific_columns:
    plt.figure()
    sns.histplot(Data[column], kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.savefig(f'{column}_histogram.png')
    plt.show()
# Visualizing Outliers using box plots
for column in specific_columns:
    plt.figure()
    sns.boxplot(y=Data[column])
    plt.title(f'Box Plot of {column}')
    plt.ylabel(column)
    plt.savefig(f'{column}_boxplot.png')
    plt.show()

#  visualizing pairplot for overall distribution
#  Defining the color palette
# Ensure the 'sex' column contains only 0 and 1
Data = Data[Data['sex'].isin([0, 1])]

# Creating the pair plot using Seaborn's built-in palette functionality
sns.pairplot(Data[specific_columns], hue='sex', palette="husl")

# Adding a title and saving the plot
plt.suptitle('Pair Plot of Selected Attributes', y=1.02)
plt.savefig('pair_plot.png')
plt.show()
# Computing the correlation matrix
selected_data = Data[specific_columns]
correlation_matrix = selected_data.corr()
# Creating a heatmap to visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')  # Save the plot as a PNG file
plt.show()
# Creating scatter plots to represent Relationships
for x_col, y_col in column_scatter:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=Data, x=x_col, y=y_col)
    plt.title(f'Scatter Plot of {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.savefig(f'scatter_{x_col}_vs_{y_col}.png')  # Saving the plot as a PNG file
    plt.show()

