import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("/mnt/data/wage_dataset/wages_by_education.csv")

# Show basic info
print(df.info())
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Convert year to integer if needed
df['year'] = df['year'].astype(int)

# Basic EDA: Plot wage trends over time by education level
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='year', y='hourly_wage', hue='education_level', marker='o')
plt.title('Average Hourly Wage by Education Level (1973–2022)')
plt.xlabel('Year')
plt.ylabel('Hourly Wage (USD)')
plt.legend(title='Education Level')
plt.grid(True)
plt.tight_layout()
plt.show()
