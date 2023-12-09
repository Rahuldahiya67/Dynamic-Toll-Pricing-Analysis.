import pandas as pd

# Question-1
import numpy as np

# Correcting the code to use numpy's fill_diagonal function
def generate_car_matrix(df):
    # Pivot the table to create a matrix with id_1 as index, id_2 as columns, and car values as data
    matrix = df.pivot(index='id_1', columns='id_2', values='car')
    # Fill the diagonal with zeros using numpy
    np.fill_diagonal(matrix.values, 0)
    return matrix


# Load the dataset
dataset_1 = pd.read_csv(r"C:\Users\user\Downloads\mapup-20231209T135215Z-001\mapup\datasets\dataset-1.csv", encoding='utf-8-sig')
# Generate the car matrix
result_matrix = generate_car_matrix(dataset_1)

# Save the result to a CSV file
result_matrix.to_csv(r"C:\Users\user\Downloads\mapup-20231209T135215Z-001\mapup\templates\car_matrix.csv")

# Display the head of the result matrix
result_matrix.head()

#The matrix displays 'id_1' as the index and 'id_2' as the columns, with the 'car' values filled in and the diagonal set to 0 as requested.

#-------------------------------------------------------------------------------------------------------------------------------------------#
#Question-2 
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

def get_type_count(df):
    # Define the conditions for the categorical types
    conditions = [
        df['car'] <= 15,
        (df['car'] > 15) & (df['car'] <= 25),
        df['car'] > 25
    ]
    # Define the corresponding categories
    categories = ['low', 'medium', 'high']
    # Create the new categorical column
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')], labels=categories, right=False)
    # Calculate the count of occurrences for each category
    type_count = df['car_type'].value_counts().sort_index().to_dict()
    return type_count

# Load the dataset
dataset_1 = pd.read_csv(r"C:\Users\user\Downloads\mapup-20231209T135215Z-001\mapup\datasets\dataset-1.csv", encoding='utf-8-sig')
# Get the type count
type_count_dict = get_type_count(dataset_1)

# Display the result
type_count_dict

#{'low': 196, 'medium': 89, 'high': 56}
#The dictionary shows the count of occurrences for each car_type category, sorted alphabetically by keys.

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Question-3
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

def get_bus_indexes(df):
    # Calculate twice the mean of the 'bus' column
    threshold = 2 * df['bus'].mean()
    # Identify the indices where 'bus' values are greater than twice the mean
    indexes = df.index[df['bus'] > threshold].tolist()
    # Sort the indices in ascending order
    indexes.sort()
    return indexes

# Load the dataset
dataset_1 = pd.read_csv(r"C:\Users\user\Downloads\mapup-20231209T135215Z-001\mapup\datasets\dataset-1.csv", encoding='utf-8-sig')
# Get the bus indexes
bus_indexes = get_bus_indexes(dataset_1)

# Display the result
bus_indexes

#Output is [2, 7, 12, 17, 25, 30, 54, 64, 70, 97, 144, 145, 149, 154, 160, 201, 206, 210, 215, 234, 235, 245, 250, 309, 314, 319, 322, 323, 334, 340]
#The list above contains the indices where the bus values are greater than twice the mean value of the bus column in the DataFrame, sorted in ascending order.

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Question-4
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

def filter_routes(df):
    # Group by 'route' and calculate the average 'truck' values
    route_groups = df.groupby('route')['truck'].mean()
    # Filter routes where the average 'truck' value is greater than 7
    filtered_routes = route_groups[route_groups > 7].index.tolist()
    # Sort the list of routes
    filtered_routes.sort()
    return filtered_routes

# Load the dataset
dataset_1 = pd.read_csv(r"C:\Users\user\Downloads\mapup-20231209T135215Z-001\mapup\datasets\dataset-1.csv", encoding='utf-8-sig')
# Apply the filter_routes function
routes_list = filter_routes(dataset_1)

# Display the result
routes_list

#Output is [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#The list above represents the sorted route values for which the average of the truck column is greater than 7.

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Question-5
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

# Load the car matrix from the previous task
matrix = pd.read_csv(r"C:\Users\user\Downloads\mapup-20231209T135215Z-001\mapup\templates\car_matrix.csv", index_col=0)

# Define the function to apply to each value
def modify_value(x):
    if x > 20:
        return round(x * 0.75, 1)
    else:
        return round(x * 1.25, 1)

# Apply the function to each value in the DataFrame
modified_matrix = matrix.progress_applymap(modify_value)

# Save the modified matrix to a new CSV file
modified_matrix.to_csv(r"C:\Users\user\Downloads\mapup-20231209T135215Z-001\mapup\templates\modified_car_matrix.csv")

# Display the head of the modified matrix
modified_matrix.head()

#The DataFrame has been successfully modified according to the specified logic, with values greater than 20 multiplied by 0.75 and values 20 or less multiplied by 1.25, rounded to one decimal place. The displayed output shows the head of the modified matrix.

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Question-6
import pandas as pd

# Load the dataset
dataset_2 = pd.read_csv(r"C:\Users\user\Downloads\mapup-20231209T135215Z-001\mapup\datasets\dataset-2.csv", encoding='utf-8-sig')

# Display the head of the dataset to inspect the data
print(dataset_2.head())

# Check for any NaT or None values in the timestamp columns
print('Null values in startDay:', dataset_2['startDay'].isnull().sum())
print('Null values in startTime:', dataset_2['startTime'].isnull().sum())
print('Null values in endDay:', dataset_2['endDay'].isnull().sum())
print('Null values in endTime:', dataset_2['endTime'].isnull().sum())

# Check for any unusual or out-of-bound date values
print('Unique startDay values:', dataset_2['startDay'].unique())
print('Unique endDay values:', dataset_2['endDay'].unique())

'''
Null values in startDay: 0
Null values in startTime: 0
Null values in endDay: 0
Null values in endTime: 0
Unique startDay values: ['Monday' 'Thursday' 'Saturday' 'Tuesday' 'Wednesday']
Unique endDay values: ['Wednesday' 'Friday' 'Sunday' 'Saturday' 'Tuesday']
The dataset has been loaded and inspected. There are no null values in the timestamp columns. However, the unique values for startDay and endDay do not include all days of the week, which could be a potential issue for the completeness check.
'''