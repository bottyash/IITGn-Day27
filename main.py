# ===============================
# ZeptoFresh Case Study Solution
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("zepto_orders.csv")

print("Initial Shape:", df.shape)
print(df.info())

# -------------------------------
# Data Cleaning
# -------------------------------

# 1. Remove invalid delivery_time = 0
df = df[df['delivery_time_mins'] > 0]

# 2. Handle outlier in order_value
df = df[df['order_value_Rs'] < 10000]

# 3. Fix negative prep_time
df['prep_time_mins'] = df['prep_time_mins'].apply(lambda x: np.nan if x < 0 else x)
df['prep_time_mins'].fillna(df['prep_time_mins'].median(), inplace=True)

# 4. Fix customer_rating
df['customer_rating'] = df['customer_rating'].replace(0, np.nan)
df['customer_rating'].fillna(df['customer_rating'].median(), inplace=True)

# -------------------------------
# Feature Engineering
# -------------------------------

df['delivery_speed'] = df['rider_distance_km'] / df['delivery_time_mins']
df['order_complexity'] = df['items_count'] * df['prep_time_mins']
df['value_per_item'] = df['order_value_Rs'] / df['items_count']

# Target Variable (Late Delivery Risk)
df['late_delivery'] = df['delivery_time_mins'].apply(lambda x: 1 if x >= 30 else 0)

# -------------------------------
# Encoding
# -------------------------------

categorical_cols = ['city', 'order_category']

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# -------------------------------
# EDA
# -------------------------------

plt.figure()
sns.histplot(df['delivery_time_mins'], bins=50)
plt.title("Delivery Time Distribution")
plt.show()

plt.figure()
sns.boxplot(x=df['delivery_time_mins'])
plt.title("Delivery Time Boxplot")
plt.show()

plt.figure()
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# -------------------------------
# Model Training
# -------------------------------

features = [
    'order_value_Rs',
    'items_count',
    'prep_time_mins',
    'rider_distance_km',
    'order_hour',
    'is_weekend',
    'rain_flag',
    'customer_rating',
    'delivery_speed',
    'order_complexity',
    'value_per_item'
]

X = df[features]
y = df['late_delivery']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Performance:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# Feature Importance
# -------------------------------

importance = pd.Series(model.feature_importances_, index=features)
importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()