#To know mapping index

import pandas as pd

# Load the dataset
file_path = '/Users/jiahui/helpdesk/MainApp/issues_dataset_balanced.csv'
data = pd.read_csv(file_path)

# Display the first few rows to confirm the data structure
print("First few rows of the dataset:")
print(data.head())

# Check if 'issue_type' column exists
if 'issue_type' not in data.columns:
    print("Error: The column 'issue_type' is not present in the dataset.")
else:
    # Mapping from the original dataset
    label_to_index = {label: idx for idx, label in enumerate(data['issue_type'].unique())}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    # Print to verify the mappings
    print("Label to Index mapping:", label_to_index)
    print("Index to Label mapping:", index_to_label)
