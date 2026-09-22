import pandas as pd

sd_data = pd.read_csv("smoking_drinking.csv", header=0, sep=",")

# 1a
# First rows
print(sd_data.head())
print("\n\n\n")

# Summary statistics
print(sd_data.describe())
print("\n\n\n")

# Data types
print(sd_data.dtypes)
print("\n\n\n")


# 1b
# Missing values
print(sd_data.isnull().sum())
print("\n\n\n")

# Unique values
print(sd_data['sex'].unique())
print(sd_data['DRK_YN'].unique())
print(sd_data['SMK_stat_type_cd'].unique())
print("\n\n\n")

# Outliers (for height)
Q1 = sd_data['waistline'].quantile(0.10)
Q3 = sd_data['waistline'].quantile(0.90)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = sd_data[(sd_data['waistline'] < lower_bound) | (sd_data['waistline'] > upper_bound)]
print(f"Number of outliers in waistline: {len(outliers)}")



# Scatter plot
"""
sd_data.plot(kind='scatter', x = 'SBP', y = 'DBP')
plt.show()
"""
# Noen verdier med sykt høye max verdier (som 999 waistline)