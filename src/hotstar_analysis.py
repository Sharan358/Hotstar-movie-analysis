
# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

from scipy.stats import shapiro, chi2_contingency, ttest_ind
from statsmodels.stats.weightstats import ztest

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sns.set(style="whitegrid")

# 2. LOAD DATA

df = pd.read_csv("C:\\Users\\Sharan\\OneDrive\\Desktop\\int 375\\data\\hotstar.csv")

print("Initial Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

# 3. DATA CLEANING

df['running_time'] = df['running_time'].fillna(df['running_time'].median())

df['seasons']  = df['seasons'].fillna(0)
df['episodes'] = df['episodes'].fillna(0)

df['age_rating'] = df['age_rating'].fillna('Unknown')

df = df.drop_duplicates()
df = df.dropna()

print("\nCleaned Shape:", df.shape)
print("\nAfter Cleaning Missing Values:\n", df.isnull().sum())
print("\nDuplicates After Cleaning:", df.duplicated().sum())

df.to_csv("C:\\Users\\Sharan\\OneDrive\\Desktop\\int 375\\data\\hotstar_cleaned.csv", index=False)

# 4. EXPLORATORY DATA ANALYSIS (EDA)

# --- 4.1 Top 10 Genres ---
genre_counts = df['genre'].value_counts()
plt.figure(figsize=(10,6))
sns.barplot(x=genre_counts.index[:10], y=genre_counts.values[:10])
plt.xticks(rotation=90)
plt.title("Top 10 Genres on Hotstar")
plt.show()

# --- 4.2 Movies vs TV Shows ---
plt.figure()
sns.countplot(x='type', data=df)
plt.title("Movies vs TV Shows")
plt.show()

# --- 4.3 Running Time Distribution ---
plt.figure()
sns.histplot(df['running_time'], kde=True)
plt.title("Running Time Distribution")
plt.show()

# --- 4.4 Year vs Running Time ---
plt.figure()
sns.scatterplot(x='year', y='running_time', data=df)
plt.title("Year vs Running Time")
plt.show()

# --- 4.5 Top Genres Share (Pie Chart) ---
top_genres = df['genre'].value_counts().head(6)
plt.figure(figsize=(6,6))
plt.pie(top_genres.values, labels=top_genres.index, autopct='%1.1f%%')
plt.title("Top Genres Share on Hotstar")
plt.show()


# 5. TREND ANALYSIS

# --- 5.1 Content Added Over Years ---
df['year'].value_counts().sort_index().plot(marker='o')
plt.title("Content Added Over Years")
plt.xticks(rotation=90)
plt.show()

# --- 5.2 Average Running Time Trend ---
avg_runtime = df.groupby('year')['running_time'].mean()
avg_runtime.plot(marker='o')
plt.title("Average Running Time Trend")
plt.xticks(rotation=90)
plt.show()

# --- 5.3 Top 5 Content Producing Years ---
top_years = df['year'].value_counts().head(5)
top_years.plot(kind='bar', color='coral')
plt.title("Top 5 Years by Content Added")
plt.xlabel("Year")
plt.ylabel("Number of Titles")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 6. NORMALITY TEST


stat, p = shapiro(df['running_time'].sample(min(5000, len(df)), random_state=42))

print("\nNormality Test P-value:", p)

if p > 0.05:
    print("Normal Distribution")
else:
    print("Not Normal Distribution")

sns.histplot(df['running_time'], kde=True)
plt.title("Running Time Distribution")
plt.show()

# 7. HYPOTHESIS TESTING

# --- 7.1 T-Test ---
drama  = df[df['genre'] == 'Drama']['running_time']
comedy = df[df['genre'] == 'Comedy']['running_time']
t_stat, t_p = ttest_ind(drama, comedy)
print("\nT-Test P-value:", t_p)

# --- 7.2 Z-Test ---
z_stat, z_p = ztest(df['running_time'], value=100)
print("Z-Test P-value:", z_p)

# --- 7.3 Chi-Square Test ---
table = pd.crosstab(df['genre'], df['type'])
chi_stat, chi_p, _, _ = chi2_contingency(table)
print("Chi-Square P-value:", chi_p)

# 8. CORRELATION HEATMAP

num_df = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(10,6))
sns.heatmap(num_df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# 9. CLUSTERING

# --- 9.1 Data Preparation ---
data_cluster = df[['running_time', 'year']].dropna()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_cluster)

# --- 9.2 K-Means ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(scaled_data)

# --- 9.3 PCA Visualization ---
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(8,5))
sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], hue=clusters)
plt.title("PCA Cluster Visualization")
plt.show()

# 10. MODEL TRAINING — PREPARING DATA

data = df[['genre', 'age_rating', 'year', 'running_time', 'type']].copy()
le_genre      = LabelEncoder()
le_age_rating = LabelEncoder()
le_type       = LabelEncoder()
 
data['genre']      = le_genre.fit_transform(data['genre'])
data['age_rating'] = le_age_rating.fit_transform(data['age_rating'])
data['type']       = le_type.fit_transform(data['type'])
 
X = data.drop('type', axis=1)
y = data['type']
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
# 11. RANDOM FOREST CLASSIFIER

print("\nStarting Random Forest training...\n")

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

rf_pred     = model_rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_accuracy)
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_pred))

# --- 11.1 Confusion Matrix — Random Forest ---
cm_rf = confusion_matrix(y_test, rf_pred)
plt.figure()
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 12. LOGISTIC REGRESSION

print("\nStarting Logistic Regression training...\n")

model_lr = LogisticRegression(max_iter=500, random_state=42)
model_lr.fit(X_train, y_train)

lr_pred     = model_lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

print("Logistic Regression Accuracy:", lr_accuracy)
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, lr_pred))

# --- 12.1 Confusion Matrix — Logistic Regression ---
cm_lr = confusion_matrix(y_test, lr_pred)
plt.figure()
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 13. MODEL COMPARISON

print("\n" + "="*45)
print("         MODEL COMPARISON SUMMARY")
print("="*45)
print(f"  Random Forest Accuracy      : {rf_accuracy:.4f}")
print(f"  Logistic Regression Accuracy: {lr_accuracy:.4f}")
print("-"*45)
if rf_accuracy > lr_accuracy:
    print("  Best Model : Random Forest")
elif lr_accuracy > rf_accuracy:
    print("  Best Model : Logistic Regression")
else:
    print("  Both models performed equally")
print("="*45)

# 14. OUTLIER DETECTION

df['z_running_time'] = zscore(df['running_time'])

outliers = df[(df['z_running_time'] > 3) | (df['z_running_time'] < -3)]
print("\nOutliers:\n", outliers[['running_time', 'z_running_time']])

# --- 14.1 Z-Score Distribution ---
sns.histplot(df['z_running_time'], kde=True)
plt.title("Z-Score Distribution (Running Time)")
plt.show()