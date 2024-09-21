#### Employee Attrition Dataset

This synthetic dataset simulates employee information and their attrition status within a fictional company. All attributes are represented in numeric format for ease of analysis and modeling.

#### Python Code
```python
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters for the synthetic dataset
num_records = 1470  # Number of records to generate

# Generate random data
ages = np.random.randint(22, 60, size=num_records)  # Age between 22 and 60
genders = np.random.choice([0, 1], size=num_records)  # 0 for Male, 1 for Female
marital_statuses = np.random.choice([0, 1, 2], size=num_records)  # 0: Single, 1: Married, 2: Divorced
job_roles = np.random.choice([0, 1, 2, 3], size=num_records)  # 0: Sales, 1: R&D, 2: Management, 3: HR
departments = np.random.choice([0, 1, 2, 3], size=num_records)  # Same encoding as job roles
job_levels = np.random.randint(1, 6, size=num_records)  # Job level between 1 and 5
job_satisfaction = np.random.randint(1, 5, size=num_records)  # Satisfaction score between 1 and 4
monthly_income = np.random.randint(3000, 12000, size=num_records)  # Monthly income between $3000 and $12000
overtime = np.random.choice([0, 1], size=num_records)  # 0 for No, 1 for Yes

# Generate attrition based on some random logic (for demonstration purposes)
attrition_probabilities = (0.3 * (job_satisfaction < 3)).astype(int) + (0.2 * (overtime == 1)).astype(int)
attrition = np.random.choice([0, 1], size=num_records, p=[attrition_probabilities.mean(), 
                                                           1 - attrition_probabilities.mean()]) 

# Create a DataFrame
data = pd.DataFrame({
    'Age': ages,
    'Gender': genders,
    'MaritalStatus': marital_statuses,
    'JobRole': job_roles,
    'Department': departments,
    'JobLevel': job_levels,
    'JobSatisfaction': job_satisfaction,
    'MonthlyIncome': monthly_income,
    'OverTime': overtime,
    'Attrition': attrition
})

# Display the first few rows of the generated dataset
print("Generated Dataset:")
print(data.head())

# Save the dataset to a CSV file
output_file = input("Enter the filename to save the dataset (e.g., employee_attrition.csv): ")
data.to_csv(output_file, index=False)
print(f"Dataset saved as {output_file}")
```

#### Features

- **Age**: The age of the employee (integer between **22** and **60**).
- **Gender**: The gender of the employee (integer: **0** for Male and **1** for Female).
- **MaritalStatus**: The marital status of the employee (integer: **0** for Single, **1** for Married, **2** for Divorced).
- **JobRole**: The role of the employee within the company (integer: **0** for Sales, **1** for Research & Development, **2** for Management, **3** for HR).
- **Department**: The department in which the employee works (same encoding as JobRole).
- **JobLevel**: A numeric representation of the employee's job level (integer between **1** and **5**).
- **JobSatisfaction**: A score representing the employee's job satisfaction (integer between **1** and **4**).
- **MonthlyIncome**: The monthly income of the employee (integer between **$3,000** and **$12,000**).
- **OverTime**: Indicates whether the employee works overtime (integer: **0** for No and **1** for Yes).
- **Attrition**: Indicates whether the employee has left the company (integer: **0** for No and **1** for Yes).

#### Data Generation Logic

The dataset was generated using random sampling methods to create a synthetic dataset. The attrition status was influenced by two main factors:
1. Job Satisfaction: Employees with a job satisfaction score less than **3** are more likely to leave.
2. Overtime: Employees who work overtime (**OverTime = 1**) are also more likely to leave.

#### Usage

This dataset can be used for various analyses:
- Predictive modeling for attrition.
- Exploratory data analysis to understand factors influencing employee turnover.
- Training machine learning models for classification tasks.

#### Saving the Dataset

The dataset can be saved as a CSV file by providing a filename when prompted in the code. For example, you can save it as `employee_attrition.csv`.

### Conclusion

This synthetic dataset serves as a valuable resource for understanding and analyzing employee attrition dynamics in a corporate setting. It can be used for educational purposes and practice in data science and machine learning applications.

Feel free to adjust any sections according to your specific requirements!
