import pandas as pd
import numpy as np
from datetime import time

df = pd.read_csv(r"C:\Users\user\Downloads\mapup-20231209T135215Z-001\mapup\datasets\dataset-3.csv")
df



#TASK 2 - Q1
def calculate_distance_matrix(dataset_path):
    # Write your logic here
    distances = {}
    for index, row in df.iterrows():
        start_location = row['id_start']
        end_location = row['id_end']
        distance = row['distance']
        distances[(start_location, end_location)] = distance
        distances[(end_location, start_location)] = distance
    toll_locations = df['id_start'].unique()
    distance_matrix = pd.DataFrame(0, index=toll_locations, columns=toll_locations)
    for i in toll_locations:
        for j in toll_locations:
            if i != j:
                direct_distance = distances.get((i, j), None)
                if direct_distance is not None:
                    distance_matrix.loc[i, j] = direct_distance
                else:
                    for k in toll_locations:
                        if i != k and j != k:
                            cumulative_distance = distance_matrix.loc[i, k] + distance_matrix.loc[k, j]
                            if distance_matrix.loc[i, j] == 0 or cumulative_distance < distance_matrix.loc[i, j]:
                                distance_matrix.loc[i, j] = cumulative_distance

    return distance_matrix

dataset_path = r"C:\Users\user\Downloads\mapup-20231209T135215Z-001\mapup\datasets\dataset-3.csv"
resulting_matrix = calculate_distance_matrix(dataset_path)
print(resulting_matrix)


#TASK 2 - Q2
def unroll_distance_matrix(distance_matrix):
    # Write your logic here
    upper_triangle = distance_matrix.where(np.triu(np.ones(distance_matrix.shape), k=1).astype(bool))
    unrolled_df = upper_triangle.stack().reset_index()
    unrolled_df.columns = ['id_start', 'id_end', 'distance']

    return unrolled_df

unrolled_df = unroll_distance_matrix(resulting_matrix)
print(unrolled_df)



#TASK 2 - Q3
def find_ids_within_ten_percentage_threshold(df, reference_value):
    # Write your logic here
    reference_rows = df[df['id_start'] == reference_value]
    average_distance = reference_rows['distance'].mean()
    lower_threshold = average_distance - 0.1 * average_distance
    upper_threshold = average_distance + 0.1 * average_distance
    within_threshold_rows = df[(df['distance'] >= lower_threshold) & (df['distance'] <= upper_threshold)]
    result_ids = sorted(within_threshold_rows['id_start'].unique())

    return result_ids

reference_value = 123  # Replace with the actual reference value
result_ids = find_ids_within_ten_percentage_threshold(unrolled_df, reference_value)
print(result_ids)



#TASK 2 - Q4
def calculate_toll_rate(df):
    # Wrie your logic here
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient
    return df

result_with_toll_rates = calculate_toll_rate(unrolled_df)
print(result_with_toll_rates)



#TASK 2 - Q5
def calculate_time_based_toll_rates(df):
    # Write your logic here
    time_ranges_weekdays = [(time(0, 0, 0), time(10, 0, 0)),
                            (time(10, 0, 0), time(18, 0, 0)),
                            (time(18, 0, 0), time(23, 59, 59))]

    time_ranges_weekends = [(time(0, 0, 0), time(23, 59, 59))]
    df['start_day'] = df['end_day'] = df['start_time'] = df['end_time'] = None

    def map_time_range(start, end, time_ranges):
        for time_range in time_ranges:
            if start >= time_range[0] and end <= time_range[1]:
                return time_range
        return None

    def apply_time_based_rates(row, time_ranges, discount_factor):
        start_day = row['start_day']
        end_day = row['end_day']
        start_time = row['start_time']
        end_time = row['end_time']

        for time_range in time_ranges:
            if start_time >= time_range[0] and end_time <= time_range[1]:
                row['start_time'] = time_range[0]
                row['end_time'] = time_range[1]
                row['start_day'] = start_day
                row['end_day'] = end_day
                row[['moto', 'car', 'rv', 'bus', 'truck']] *= discount_factor
                return row

    for index, row in df.iterrows():
        start_day = row['start_day']
        end_day = row['end_day']

        if start_day == end_day:
            if start_day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                row = apply_time_based_rates(row, time_ranges_weekdays, 0.8)
            elif start_day in ['Saturday', 'Sunday']:
                row = apply_time_based_rates(row, time_ranges_weekends, 0.7)

    return df

result_with_time_based_rates = calculate_time_based_toll_rates(result_with_toll_rates)
print(result_with_time_based_rates)


