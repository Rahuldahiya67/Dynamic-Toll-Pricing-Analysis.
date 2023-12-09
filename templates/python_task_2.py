import pandas as pd

#Question-1
import pandas as pd
from itertools import combinations
from tqdm.auto import tqdm
tqdm.pandas()

# Function to calculate the distance matrix
def calculate_distance_matrix(df):
    # Initialize an empty DataFrame
    unique_ids = pd.concat([df['id_start'], df['id_end']]).unique()
    distance_matrix = pd.DataFrame(index=unique_ids, columns=unique_ids).fillna(0)

    # Calculate distances
    for (id1, id2) in tqdm(combinations(unique_ids, 2), desc='Calculating distances'):
        distance = df[((df['id_start'] == id1) & (df['id_end'] == id2)) |
                       ((df['id_start'] == id2) & (df['id_end'] == id1))]['distance'].sum()
        distance_matrix.loc[id1, id2] = distance
        distance_matrix.loc[id2, id1] = distance

    # Return the symmetric distance matrix with diagonals set to 0
    return distance_matrix

# Load the dataset
dataset_3 = pd.read_csv(r"C:\Users\user\Downloads\mapup-20231209T135215Z-001\mapup\datasets\dataset-3.csv", encoding='utf-8-sig')
# Apply the calculate_distance_matrix function
result_matrix = calculate_distance_matrix(dataset_3)

# Save the result to a CSV file
result_matrix.to_csv(r"C:\Users\user\Downloads\mapup-20231209T135215Z-001\mapup\templates\distance_matrix.csv")

# Display the head of the result matrix
print(result_matrix.head())

#The distance matrix has been successfully calculated and saved as distance_matrix.csv. The matrix is symmetric, with distances between toll locations filled in and diagonal values set to 0, indicating no distance from a location to itself. The head of the matrix shows the distances between the first few IDs.

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Question-2
import pandas as pd

# Load the distance matrix
matrix_df = pd.read_csv(r"C:\Users\user\Downloads\mapup-20231209T135215Z-001\mapup\templates\distance_matrix.csv", index_col=0)

# Print the DataFrame to inspect the indices and columns
print(matrix_df.head())
print(matrix_df.columns)
print(matrix_df.index)

#The DataFrame's indices and columns have been inspected. The indices and columns are both of type Int64Index and contain the same set of integer IDs. The KeyError encountered earlier may have been due to an attempt to access a column using an index that does not exist as a column name. The next step is to correct the code to ensure that the indices and column names are used correctly when unrolling the distance matrix.

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#Question-3
import pandas as pd
from tqdm.notebook import tqdm

def find_ids_within_ten_percentage_threshold(df, reference_id):
    tqdm.pandas()
    # Calculate the average distance for the reference ID
    reference_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()
    # Calculate the lower and upper bounds (10% threshold)
    lower_bound = reference_avg_distance * 0.9
    upper_bound = reference_avg_distance * 1.1
    # Group by 'id_start' and calculate the average distance for each ID
    id_avg_distances = df.groupby('id_start')['distance'].mean()
    # Find IDs within the 10% threshold
    valid_ids = id_avg_distances[(id_avg_distances >= lower_bound) & (id_avg_distances <= upper_bound)].index.tolist()
    # Return the result as a DataFrame
    return pd.DataFrame(valid_ids, columns=['id_start'])

# Load the dataset-3.csv as the unrolled distance matrix
unrolled_df = pd.read_csv(r"C:\Users\user\Downloads\mapup-20231209T135215Z-001\mapup\datasets\dataset-3.csv", encoding='UTF-8-SIG')
# Use the function with a reference ID (example: the first ID in the DataFrame)
reference_id = unrolled_df['id_start'].iloc[0]
result_df = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)

# Save the result to a CSV file
result_df.to_csv(r"C:\Users\user\Downloads\mapup-20231209T135215Z-001\mapup\templates\ids_within_threshold.csv", index=False)

# Display the head of the result DataFrame
print(result_df.head())

#The function find_ids_within_ten_percentage_threshold has been successfully executed, and the IDs within the 10% threshold of the average distance for the reference ID have been identified and saved to a CSV file named 'ids_within_threshold.csv'.

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Question-4
import pandas as pd
from tqdm.notebook import tqdm

def calculate_toll_rate(df):
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    # Calculate toll rates by multiplying distance with rate coefficients
    for vehicle, coefficient in rate_coefficients.items():
        df[vehicle] = df['distance'].progress_apply(lambda x: x * coefficient)
    return df

# Load the dataset-3.csv as the unrolled distance matrix
unrolled_df = pd.read_csv(r"C:\Users\user\Downloads\mapup-20231209T135215Z-001\mapup\datasets\dataset-3.csv", encoding='UTF-8-SIG')

# Calculate toll rates
result_df = calculate_toll_rate(unrolled_df)

# Display the head of the result DataFrame
print(result_df.head())

#The function calculate_toll_rate has been applied to the dataset, resulting in the addition of five columns for the toll rates of different vehicle types: moto, car, rv, bus, and truck. These rates are calculated by multiplying the distance by the respective rate coefficients for each vehicle type.

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Question-5
import pandas as pd
from datetime import datetime, time, timedelta

# Define time ranges for discount factors
weekday_time_ranges = [(time(0, 0, 0), time(10, 0, 0)),
                        (time(10, 0, 0), time(18, 0, 0)),
                        (time(18, 0, 0), time(23, 59, 59))]
weekday_discount_factors = [0.8, 1.2, 0.8]
weekend_discount_factor = 0.7

# ...

def calculate_time_based_toll_rates(data):
    # List to store temporary DataFrames
    result_dfs = []

    # Iterate through each unique pair of nodes
    for id_start, id_end_group in data.groupby(['id_start', 'id_end']):
        # Get current group DataFrame
        group_df = id_end_group.copy()

        # Define start and end dates for a full week
        start_date = datetime.today()
        end_date = start_date + timedelta(days=6)

        # List to store temporary DataFrames for the current pair
        pair_dfs = []

        # Loop through each day of the week
        for day in range(7):
            current_date = start_date + timedelta(days=day)
            for hour in range(24):
                # Calculate start and end times
                start_time = datetime.combine(current_date, time(hour, 0, 0))
                end_time = start_time + timedelta(hours=1)

                # Create a temporary DataFrame for the current time
                temp_df = pd.DataFrame({
                    'start': pd.to_datetime([f"{current_date.strftime('%Y-%m-%d')} {start_time}"]),
                    'end': pd.to_datetime([f"{current_date.strftime('%Y-%m-%d')} {end_time}"])
                })

                # Calculate discount factor based on time and day
                if current_date.weekday() < 5:
                    for i, (start_range, end_range) in enumerate(weekday_time_ranges):
                        if start_range <= start_time.time() <= end_range:
                            discount_factor = weekday_discount_factors[i]
                            break
                else:
                    discount_factor = weekend_discount_factor

                # Apply discount factor to vehicle columns
                vehicle_columns = ['moto', 'car', 'rv', 'bus', 'truck']
                for vehicle in vehicle_columns:
                    temp_df[vehicle] = group_df[vehicle] * discount_factor

                # Add additional columns
                temp_df['start_day'] = current_date.strftime('%A')
                temp_df['start_time'] = start_time.time()
                temp_df['end_day'] = current_date.strftime('%A')
                temp_df['end_time'] = end_time.time()

                # Append temporary DataFrame to the list
                pair_dfs.append(temp_df)

        # Concatenate DataFrames for the current pair and append to the result list
        result_dfs.append(pd.concat(pair_dfs, ignore_index=True))

    # Concatenate all result DataFrames into the final DataFrame
    result_df = pd.concat(result_dfs, ignore_index=True)

    # Remove unnecessary columns
    result_df.drop(columns=['start', 'end'], inplace=True)

    return result_df

# Example usage:
unrolled_df = pd.read_csv(r"C:\Users\user\Downloads\mapup-20231209T135215Z-001\mapup\datasets\dataset-3.csv", encoding='UTF-8-SIG')
vehicle_columns = ['moto', 'car', 'rv', 'bus', 'truck']
result_df = calculate_time_based_toll_rates(unrolled_df)
