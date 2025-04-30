import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

wandb.init(project="wage-education-analysis", name="updated_analysis")

# Load dataset
df = pd.read_csv("../Datasets/wages_by_education.csv")

# Reshape for main education levels
edu_cols = ['less_than_hs', 'high_school', 'some_college', 'bachelors_degree', 'advanced_degree']
df_melted = df.melt(id_vars='year', value_vars=edu_cols,
                    var_name='education_level', value_name='hourly_wage')

# Line Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_melted, x='year', y='hourly_wage', hue='education_level', marker='o')
plt.title('Average Hourly Wage by Education Level (1973–2022)')
plt.xlabel('Year')
plt.ylabel('Hourly Wage (USD)')
plt.grid(True)
plt.tight_layout()
plt.savefig("line_plot.png")
wandb.log({"Line Plot": wandb.Image("line_plot.png")})
plt.close()

# Regression analysis
plt.figure(figsize=(12, 6))
metrics_table = []
for edu in edu_cols:
    edu_df = df[['year', edu]].dropna()
    x = edu_df['year'].values.reshape(-1, 1)
    y = edu_df[edu].values.reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    y_pred = reg.predict(x)
    slope = reg.coef_[0][0]
    intercept = reg.intercept_[0]
    r2 = r2_score(y, y_pred)
    growth = ((y[-1] - y[0]) / y[0]) * 100

    plt.plot(df['year'], y, label=f'{edu} Actual')
    plt.plot(df['year'], y_pred, linestyle='--', label=f'{edu} Predicted')
    metrics_table.append([edu, slope, intercept, r2, float(growth)])

plt.title("Wage Trends with Linear Regression")
plt.xlabel("Year")
plt.ylabel("Hourly Wage (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("regression_plot.png")
wandb.log({"Regression Plot": wandb.Image("regression_plot.png")})
plt.close()

# Year-over-year changes
plt.figure(figsize=(12, 6))
for edu in edu_cols:
    df[f'{edu}_delta'] = df[edu].diff()
    sns.lineplot(x=df['year'], y=df[f'{edu}_delta'], label=edu)
plt.title("Year-over-Year Wage Change by Education Level")
plt.xlabel("Year")
plt.ylabel("Δ Hourly Wage (USD)")
plt.grid(True)
plt.tight_layout()
plt.savefig("delta_plot.png")
wandb.log({"Delta Plot": wandb.Image("delta_plot.png")})
plt.close()

# Log regression metrics
wandb_table = wandb.Table(data=metrics_table, columns=["Education Level", "Slope", "Intercept", "R²", "Growth %"])
wandb.log({"Regression Metrics": wandb_table})

wandb.finish()
