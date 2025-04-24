import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("wages_by_education.csv")

# Reshape for main education levels (5 columns)
edu_cols = ['less_than_hs', 'high_school', 'some_college', 'bachelors_degree', 'advanced_degree']
df_melted = df.melt(id_vars='year', value_vars=edu_cols,
                    var_name='education_level', value_name='hourly_wage')

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_melted, x='year', y='hourly_wage', hue='education_level', marker='o')
plt.title('Average Hourly Wage by Education Level (1973–2022)')
plt.xlabel('Year')
plt.ylabel('Hourly Wage (USD)')
plt.grid(True)
plt.tight_layout()
plt.show()
