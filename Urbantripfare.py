# 1. Import Liabraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Load data

df = pd.read_csv("taxi_fare_dataset.csv")
print(df.head())

# 3. Data Understanding

print("Shape:", df.shape)

print("\nInfo:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicates:", df.duplicated().sum())

# 4. Feature Engineering

df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

df['hour'] = df['tpep_pickup_datetime'].dt.hour
df['day'] = df['tpep_pickup_datetime'].dt.day_name()

# 5. Time Features

df['am_pm'] = df['hour'].apply(lambda x: 'AM' if x < 12 else 'PM')
df['is_night'] = df['hour'].apply(lambda x: 1 if x >= 22 or x <= 5 else 0)
df['is_weekend'] = df['day'].isin(['Saturday', 'Sunday']).astype(int)

# 6. Timezone Conversation

df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], utc=True)
df['tpep_pickup_datetime'] = df['tpep_pickup_datetime'].dt.tz_convert('America/New_York')

# Recalculate
df['hour'] = df['tpep_pickup_datetime'].dt.hour
df['day'] = df['tpep_pickup_datetime'].dt.day_name()
df['am_pm'] = df['hour'].apply(lambda x: 'AM' if x < 12 else 'PM')
df['is_night'] = df['hour'].apply(lambda x: 1 if x >= 22 or x <= 5 else 0)
df['is_weekend'] = df['day'].isin(['Saturday', 'Sunday']).astype(int)

# 7. Trip Distance

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

df['trip_distance'] = haversine(
    df['pickup_latitude'],
    df['pickup_longitude'],
    df['dropoff_latitude'],
    df['dropoff_longitude']
)

# 8. Exploratory Data Analysis

plt.figure(figsize=(6,4))
sns.scatterplot(x='trip_distance', y='total_amount', data=df)
plt.title("Fare vs Distance")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='passenger_count', y='total_amount', data=df)
plt.title("Passenger vs Fare")
plt.show()

plt.figure(figsize=(8,4))
sns.countplot(x='hour', data=df)
plt.title("Trips by Hour")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='is_weekend', y='total_amount', data=df)
plt.title("Weekend vs Fare")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='is_night', y='total_amount', data=df)
plt.title("Night vs Fare")
plt.show()

# 9. Data Transformation

Q1 = df['total_amount'].quantile(0.25)
Q3 = df['total_amount'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['total_amount'] >= Q1 - 1.5*IQR) &
        (df['total_amount'] <= Q3 + 1.5*IQR)]

print("After removing outliers:", df.shape)

df = pd.get_dummies(df, columns=['am_pm'], drop_first=True)
df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'Y':1, 'N':0})
df = pd.get_dummies(df, columns=['day'], drop_first=True)

# 10. Feature Selection

numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

df = df.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime'], axis=1)

# 11. Train-Test Split

from sklearn.model_selection import train_test_split

X = df[['passenger_count', 'trip_distance', 'hour', 'is_night', 'is_weekend']]
y = df['total_amount']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Split:", X_train.shape, X_test.shape)

# 12. Model Building + Evaluation

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

best_model = None
best_r2 = -1

for name, model in models.items():
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = model.score(X_test, y_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"{name}")
    print(f"R2: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    print("-"*40)
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = model

print("\nBest Model:", best_model)
print("Best R2:", best_r2)

import joblib
joblib.dump(best_model, "fare_model.pkl")



























