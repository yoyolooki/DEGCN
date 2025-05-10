import pandas as pd
import numpy as np

# Read sample_classes.csv, assuming the first column contains sample names
df = pd.read_csv('data/KCdata/sample_classes.csv')
sample_names = df.iloc[:, 0].tolist()

# Get the number of samples
n = len(sample_names)

# Create an identity matrix of size n x n
identity_matrix = np.eye(n)
identity_df = pd.DataFrame(identity_matrix, index=sample_names, columns=sample_names)
identity_df.insert(0, 'SampleName', sample_names)
identity_df.to_csv('result/identity_matrix.csv', index=False)
