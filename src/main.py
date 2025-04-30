import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

wandb.init(project="wage-education-analysis", name="detailed_analysis")

# Load dataset
df = pd.read_csv("../Datasets/wages_by_education.csv")

# Define education level columns
edu_cols = ['less_than_hs', 'high_school', 'some_college', 'bachelors_degree', 'advanced_degree']
recession_years = [2001, 2008, 2020]

# ========== Line Plot with Recession Markers ==========
df_melted = df.melt(id_vars='year', value_vars=edu_cols,
                    var_name='education_level', value_name='hourly_wage')
plt.figure(figsize=(14, 7))
sns.lineplot(data=df_melted, x='year', y='hourly_wage', hue='education_level', marker='o')
for year in recession_years:
    plt.axvline(x=year, color='gray', linestyle='--', alpha=0.3, label=f'Recession {year}')
for edu in edu_cols:
    max_year = df.loc[df[edu].idxmax(), 'year']
    max_wage = df[edu].max()
    plt.text(max_year, max_wage + 0.5, f"{max_wage:.2f}", fontsize=9, ha='center')
plt.title('Average Hourly Wage by Education Level (1973–2022)')
plt.xlabel('Year')
plt.ylabel('Hourly Wage (USD)')
plt.grid(True)
plt.tight_layout()
plt.savefig("line_plot_detailed.png")
wandb.log({"Line Plot (Detailed)": wandb.Image("line_plot_detailed.png")})
plt.close()

# ========== Regression Analysis ==========
plt.figure(figsize=(14, 7))
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
    growth = ((y[-1][0] - y[0][0]) / y[0][0]) * 100
    plt.plot(df['year'], y, label=f'{edu} Actual')
    plt.plot(df['year'], y_pred, linestyle='--', label=f'{edu} Predicted (R²={r2:.2f})')
    metrics_table.append([edu, slope, intercept, r2, float(growth)])
plt.title("Wage Trends with Linear Regression")
plt.xlabel("Year")
plt.ylabel("Hourly Wage (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("regression_plot_detailed.png")
wandb.log({"Regression Plot (Detailed)": wandb.Image("regression_plot_detailed.png")})
plt.close()

# ========== Year-over-Year Wage Change ==========
plt.figure(figsize=(14, 7))
for edu in edu_cols:
    df[f'{edu}_delta'] = df[edu].diff()
    sns.lineplot(x=df['year'], y=df[f'{edu}_delta'], label=edu)
for year in recession_years:
    plt.axvline(x=year, color='gray', linestyle='--', alpha=0.3)
plt.title("Year-over-Year Wage Change by Education Level")
plt.xlabel("Year")
plt.ylabel("Δ Hourly Wage (USD)")
plt.grid(True)
plt.tight_layout()
plt.savefig("delta_plot_detailed.png")
wandb.log({"Delta Plot (Detailed)": wandb.Image("delta_plot_detailed.png")})
plt.close()

# ========== Bar Chart: 2022 Snapshot ==========
df_2022 = df[df['year'] == 2022]
wages_2022 = df_2022[edu_cols].values[0]
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=edu_cols, y=wages_2022)
for bar in ax.patches:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', (bar.get_x() + bar.get_width() / 2, height),
                ha='center', va='bottom')
plt.title("Hourly Wages by Education Level (2022)")
plt.ylabel("Hourly Wage (USD)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("bar_2022_detailed.png")
wandb.log({"2022 Wage Bar Chart (Detailed)": wandb.Image("bar_2022_detailed.png")})
plt.close()

# ========== Bar Chart: Growth (1973–2022) ==========
growth_percentages = [((df[edu].iloc[-1] - df[edu].iloc[0]) / df[edu].iloc[0]) * 100 for edu in edu_cols]
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=edu_cols, y=growth_percentages)
for bar in ax.patches:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%', (bar.get_x() + bar.get_width() / 2, height),
                ha='center', va='bottom')
plt.title("Percentage Growth in Wages by Education Level (1973–2022)")
plt.ylabel("Growth %")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("growth_chart_detailed.png")
wandb.log({"Growth Bar Chart (Detailed)": wandb.Image("growth_chart_detailed.png")})
plt.close()

# ========== Log Regression Metrics ==========
wandb_table = wandb.Table(data=metrics_table, columns=["Education Level", "Slope", "Intercept", "R²", "Growth %"])
wandb.log({"Regression Metrics": wandb_table})

wandb.finish()