import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from sklearn.linear_model import LinearRegression


df = pd.read_csv("C:\\Users\\Sharan\\OneDrive\\Desktop\\int 375\\data\\hotstar.csv")

print("rows:", len(df))
print("columns:", len(df.columns))
print("\nmissing values:\n", df.isnull().sum())
print("duplicate rows:", df.duplicated().sum())

# data cleaning
df['running_time'] = df['running_time'].fillna(df['running_time'].median())
df['seasons'] = df['seasons'].fillna(0)
df['episodes'] = df['episodes'].fillna(0)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.drop_duplicates()
df = df.dropna(subset=['genre', 'type', 'year'])

print("\ncleaned shape:", df.shape)
print("\nmissing values after cleaning:\n", df.isnull().sum())

print(df.describe())

# iqr outlier detection
num_cols = ['running_time', 'seasons', 'episodes']
print("\niqr outlier detection")
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(col, "outliers:", len(outliers))

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("rows after cleaning:", len(df))


print(df.corr(numeric_only=True))

# objective 1
# to analyze the relationship between running_time and year using correlation analysis.
corr = df['running_time'].corr(df['year'])
print("\ncorrelation (running_time vs year):", round(corr, 4))

# objective 2
# to identify the best-performing content type based on average running_time.
print("\ncontent type avg running time:\n", df.groupby('type')['running_time'].mean())
print("\ntop genres by count:\n", df['genre'].value_counts().head(10))

# objective 3: visualization
# 1. scatter plot
plt.figure()
sns.scatterplot(x='year', y='running_time', hue='type', data=df)
plt.title("Year vs Running Time")
plt.xlabel("Year")
plt.ylabel("Running Time (min)")
plt.tight_layout()
plt.show()

# 2. histogram
plt.figure()
sns.histplot(df['running_time'], kde=True, color='steelblue')
plt.title("Running Time Distribution")
plt.xlabel("Running Time (min)")
plt.tight_layout()
plt.show()

# 3. line plot
yearly = df.groupby('year')['running_time'].mean().reset_index()
plt.figure()
sns.lineplot(x='year', y='running_time', data=yearly, marker='o')
plt.title("Average Running Time per Year")
plt.xlabel("Year")
plt.ylabel("Avg Running Time (min)")
plt.tight_layout()
plt.show()

# 4. bubble plot

tv_df = df[df['type'] == 'tv'].copy()
plt.figure()
plt.scatter(tv_df['year'], tv_df['running_time'], s=tv_df['episodes'] * 0.5 + 10, alpha=0.5, color='coral')
plt.title("Bubble Plot (Year vs Running Time, bubble = episodes, TV only)")
plt.xlabel("Year")
plt.ylabel("Running Time (min)")
plt.tight_layout()
plt.show()

# 5. box plot
plt.figure()
sns.boxplot(x='type', y='running_time', data=df)
plt.title("Content Type vs Running Time")
plt.xlabel("Type")
plt.ylabel("Running Time (min)")
plt.tight_layout()
plt.show()

# 6. bar plot
top_genres = df['genre'].value_counts().head(10)
plt.figure()
top_genres.plot(kind='bar', color='mediumseagreen')
plt.title("Top 10 Genres by Count")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 7. heatmap
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 8. pie chart

plt.figure()
df['type'].value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue', 'salmon'], startangle=90)
plt.title("Content Type Distribution")
plt.ylabel('')
plt.tight_layout()
plt.show()

# objective 4
# h0: mean running_time = 90 min
t_stat, p_val = ttest_1samp(df['running_time'], 90)
print("\nt-test p-value:", p_val)
mean = df['running_time'].mean()
std = df['running_time'].std()
n = len(df)

if p_val < 0.05:
    print("result: reject h0 – average running time is significantly different from 90 min.")
else:
    print("result: fail to reject h0 – no significant difference from 90 min.")

# objective 5
# predicting running_time from year
x = df['year']
y = df['running_time']
mean_x = x.mean()
mean_y = y.mean()
num = ((x - mean_x) * (y - mean_y)).sum()
den = ((x - mean_x) ** 2).sum()
slope = num / den
intercept = mean_y - slope * mean_x

print("\nregression equation:")
print("y =", slope, "* x +", intercept)

plt.figure()
sns.regplot(x='year', y='running_time', data=df, line_kws={"color": "red"})
plt.title("Regression Plot – Year vs Running Time")
plt.xlabel("Year")
plt.ylabel("Running Time (min)")
plt.tight_layout()
plt.show()