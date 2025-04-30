import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import os

# Initialize Weights & Biases
wandb.init(project="wage-education-analysis", name="detailed_analysis")

# Load dataset
df = pd.read_csv("../Datasets/wages_by_education.csv")

# Define education level columns
edu_cols = ['less_than_hs', 'high_school', 'some_college', 'bachelors_degree', 'advanced_degree']

# Create directory for saving plots
os.makedirs("plots", exist_ok=True)

# Bar Chart: Hourly wages in 2022
df_2022 = df[df['year'] == 2022]
plt.figure(figsize=(10, 6))
sns.barplot(x=edu_cols, y=df_2022[edu_cols].values[0])
plt.title("Hourly Wages by Education Level (2022)")
plt.ylabel("Hourly Wage (USD)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/2022_snapshot.png")
wandb.log({"2022 Snapshot": wandb.Image("plots/2022_snapshot.png")})
plt.close()

# Linear Regression Analysis and Trendlines
metrics_table = []
plt.figure(figsize=(12, 7))
for edu in edu_cols:
    edu_df = df[['year', edu]].dropna()
    x = edu_df['year'].values.reshape(-1, 1)
    y = edu_df[edu].values.reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    y_pred = reg.predict(x)
    slope = reg.coef_[0][0]
    intercept = reg.intercept_[0]
    r2 = r2_score(y, y_pred)
    growth = ((y[-1][0] - y[0][0]) / y[0][0]) * 100

    plt.plot(df['year'], y, label=f'{edu} Actual')
    plt.plot(df['year'], y_pred, linestyle='--', label=f'{edu} Trend (R²={r2:.2f})')
    metrics_table.append([edu, slope, intercept, r2, float(growth)])

plt.title("Wage Trends and Linear Regression by Education Level (1973–2022)")
plt.xlabel("Year")
plt.ylabel("Hourly Wage (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/regression_detailed.png")
wandb.log({"Regression Trendlines": wandb.Image("plots/regression_detailed.png")})
plt.close()

# Year-over-Year Change
plt.figure(figsize=(12, 7))
for edu in edu_cols:
    df[f'{edu}_delta'] = df[edu].diff()
    sns.lineplot(x=df['year'], y=df[f'{edu}_delta'], label=edu)
plt.title("Year-over-Year Wage Change by Education Level")
plt.xlabel("Year")
plt.ylabel("Δ Hourly Wage (USD)")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/yearly_deltas.png")
wandb.log({"YOY Delta Plot": wandb.Image("plots/yearly_deltas.png")})
plt.close()

# Growth from 1973 to 2022
growth_percentages = [((df[edu].iloc[-1] - df[edu].iloc[0]) / df[edu].iloc[0]) * 100 for edu in edu_cols]
plt.figure(figsize=(10, 6))
sns.barplot(x=edu_cols, y=growth_percentages)
plt.title("Total Wage Growth by Education Level (1973–2022)")
plt.ylabel("Growth %")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/growth_percentages.png")
wandb.log({"Wage Growth Chart": wandb.Image("plots/growth_percentages.png")})
plt.close()

# Log regression metrics to W&B
wandb_table = wandb.Table(data=metrics_table, columns=["Education Level", "Slope", "Intercept", "R²", "Growth %"])
wandb.log({"Regression Metrics": wandb_table})

wandb.finish()
