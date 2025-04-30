import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# Initialize W&B
wandb.init(project="wage-education-analysis", name="education_xlsx_analysis")

# Set path to Excel file
xlsx_path = "/Users/johnross/Analysis-of-Wages-by-Education-in-the-U.S.A-1973-2022-/Datasets/education.xlsx"


# Load Table 5.1: Unemployment rates and earnings by educational attainment
table_51 = pd.read_excel(xlsx_path, sheet_name='Table 5.1', skiprows=3, nrows=13)

# Rename columns for convenience
table_51.columns = ['Education Level', 'Unemployment Rate (%)', 'Median Weekly Earnings ($)']

# Drop rows with NaN if present
table_51.dropna(inplace=True)

# Plot histogram of earnings
plt.figure(figsize=(10, 6))
sns.barplot(data=table_51, x='Education Level', y='Median Weekly Earnings ($)', palette='Blues_d')
plt.xticks(rotation=45, ha='right')
plt.title("Median Weekly Earnings by Education Level (2023)")
plt.tight_layout()
plt.savefig("median_earnings_histogram.png")
wandb.log({"Median Earnings Histogram": wandb.Image("median_earnings_histogram.png")})
plt.close()

# Plot unemployment rates
plt.figure(figsize=(10, 6))
sns.barplot(data=table_51, x='Education Level', y='Unemployment Rate (%)', palette='Reds')
plt.xticks(rotation=45, ha='right')
plt.title("Unemployment Rate by Education Level (2023)")
plt.tight_layout()
plt.savefig("unemployment_rate_histogram.png")
wandb.log({"Unemployment Rate Histogram": wandb.Image("unemployment_rate_histogram.png")})
plt.close()

# Linear Regression: Predict earnings based on education rank
edu_order = table_51['Education Level'].rank(method='dense').values.reshape(-1, 1)
earnings = table_51['Median Weekly Earnings ($)'].values.reshape(-1, 1)

model = LinearRegression().fit(edu_order, earnings)
predicted = model.predict(edu_order)
r2 = r2_score(earnings, predicted)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=edu_order.flatten(), y=earnings.flatten(), s=100, label='Actual', color='blue')
plt.plot(edu_order, predicted, color='black', linestyle='--', label=f'Linear Fit (R²={r2:.2f})')
plt.title("Linear Regression: Education Rank vs. Earnings")
plt.xlabel("Education Rank")
plt.ylabel("Median Weekly Earnings ($)")
plt.legend()
plt.tight_layout()
plt.savefig("education_regression.png")
wandb.log({"Earnings Regression": wandb.Image("education_regression.png")})
plt.close()

# Log metrics
wandb.log({
    "Regression Slope": model.coef_[0][0],
    "Regression Intercept": model.intercept_[0],
    "R² Score": r2
})

wandb.finish()