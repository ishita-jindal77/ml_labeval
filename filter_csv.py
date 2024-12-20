import pandas as pd

# Specify the dataset location
dataset_path = 'Fuel_cell_performance_data-Full.csv'

# Load the dataset into a DataFrame
fuel_data = pd.read_csv(dataset_path)

# Display the column names for review
print("Dataset columns:")
print(fuel_data.columns)

# Identify and remove target columns, retaining only 'Target4'
columns_to_remove = [column for column in fuel_data.columns if column.startswith('Target') and column != 'Target4']
filtered_data = fuel_data.drop(columns=columns_to_remove)

# Check the remaining columns
print("Columns retained after filtering:")
print(filtered_data.columns)

# Save the processed dataset to a new file
filtered_data.to_csv('Filtered_Fuel_cell_performance_data.csv', index=False)
print("Processed dataset has been saved as 'Filtered_Fuel_cell_performance_data.csv'.")
