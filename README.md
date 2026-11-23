# Binary-Classification-with-Neural-Networks-on-the-Census-Income-Dataset

## AIM

To develop a binary classification model using PyTorch to predict whether an individual's annual income is greater than $50,000 or less than or equal to $50,000 using the Census Income Dataset.

## OBJECTIVES

1. To load and understand the Census Income Dataset.
2. To separate categorical, continuous and label columns.
3. To preprocess the dataset and handle missing values.
4. To convert data into PyTorch tensors.
5. To design a neural network using PyTorch.
6. To train the model using Adam optimizer.
7. To evaluate the model using test loss and accuracy.
   
## SOFTWARE REQUIREMENTS

- Operating System: Windows 11
- Programming Language: Python
- IDE: Visual Studio Code
- Framework: PyTorch
- Libraries: NumPy, Pandas, Scikit-learn
- Notebook: Jupyter Notebook
  
## DATASET DESCRIPTION
- Dataset Name: Census Income Dataset
- File Name: income.csv
- Total Records: 30,000
- Target Column: label
- Class 0: Income less than or equal to $50,000
- Class 1: Income greater than $50,000
  
## PROCEDURE
1. Import the required libraries.
2. Set random seed for reproducibility.
3. Load the Census Income Dataset.
4. Separate categorical, continuous and label columns.
5. Handle missing values.
6. Encode categorical features.
7. Create arrays and tensors.
8. Split data into training and testing sets.
9. Define the TabularModel class.
10. Initialize loss function and optimizer.
11. Train the model for 300 epochs.
12. Evaluate test loss and accuracy.
13. Predict income for new user inputs.
    
## Program
```
import os
import pandas as pd
from pathlib import Path
possible_paths = [
    Path("../income.csv"),
    Path("income.csv"),
    Path("../Census-Income-Classification/income.csv"),
    Path.cwd() / "income.csv",
    Path.home() / "Downloads" / "income.csv"
]
csv_path = next((path for path in possible_paths if path.exists()), None)
if csv_path is None:
    raise FileNotFoundError("income.csv not found. Please place it in the project folder or Downloads folder.")
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
print("Dataset Loaded Successfully!")
print("Dataset Location:", csv_path.resolve())
print("Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())
print("\nDataset Columns:")
print(df.columns.tolist())
```
## Output

<img width="1375" height="525" alt="image" src="https://github.com/user-attachments/assets/c06e7f8c-c74a-4573-a419-702c7ccf5417" />


## Result
Thus, the Census Income Classification project was successfully implemented and executed using Python and PyTorch. The neural network model was trained for 300 epochs and successfully performed binary classification to predict whether an individual's annual income is greater than $50,000 or less than or equal to $50,000.

The program executed successfully and produced the test loss, test accuracy, and income prediction results.
