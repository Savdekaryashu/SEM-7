import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import zscore

# ------------------- 1. Load dataset -------------------
df = pd.read_csv("uber.csv")  # Change to your file path

# ------------------- 2. Preprocessing -------------------
df.dropna(inplace=True)  # Remove missing values
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
df = df[df['pickup_datetime'].notnull()]

# Filter invalid coordinates
df = df[
    (df['pickup_latitude'].between(-90, 90)) &
    (df['pickup_longitude'].between(-180, 180)) &
    (df['dropoff_latitude'].between(-90, 90)) &
    (df['dropoff_longitude'].between(-180, 180))
    ]

# ------------------- 3. Feature Engineering -------------------
df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.dayofweek


# Haversine distance
def haversine(lon1, lat1, lon2, lat2):
    from math import radians, cos, sin, asin, sqrt
    R = 6371
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


df['distance_km'] = df.apply(lambda row: haversine(
    row['pickup_longitude'], row['pickup_latitude'],
    row['dropoff_longitude'], row['dropoff_latitude']), axis=1
                             )

# ------------------- 4. Outlier Detection -------------------
plt.figure(figsize=(14, 10))

# Boxplots
plt.subplot(3, 2, 1)
sns.boxplot(y=df['fare_amount'])
plt.title("Boxplot: Fare Amount")
plt.subplot(3, 2, 2)
sns.boxplot(y=df['distance_km'])
plt.title("Boxplot: Distance (km)")

# Violin plots
plt.subplot(3, 2, 3)
sns.violinplot(y=df['fare_amount'])
plt.title("Violin Plot: Fare Amount")
plt.subplot(3, 2, 4)
sns.violinplot(y=df['distance_km'])
plt.title("Violin Plot: Distance (km)")

# Histograms
plt.subplot(3, 2, 5)
plt.hist(df['fare_amount'], bins=50, color='skyblue', edgecolor='black')
plt.title("Histogram: Fare Amount")
plt.subplot(3, 2, 6)
plt.hist(df['distance_km'], bins=50, color='lightgreen', edgecolor='black')
plt.title("Histogram: Distance (km)")

plt.tight_layout()
plt.show()

# Z-score highlighting
df['fare_z'] = zscore(df['fare_amount'])
df['dist_z'] = zscore(df['distance_km'])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(df.index, df['fare_amount'], c=np.where(np.abs(df['fare_z']) > 3, 'red', 'blue'), alpha=0.5)
plt.title("Z-Score Highlighted: Fare Amount")
plt.xlabel("Index")
plt.ylabel("Fare Amount")

plt.subplot(1, 2, 2)
plt.scatter(df.index, df['distance_km'], c=np.where(np.abs(df['dist_z']) > 3, 'red', 'green'), alpha=0.5)
plt.title("Z-Score Highlighted: Distance (km)")
plt.xlabel("Index")
plt.ylabel("Distance (km)")

plt.tight_layout()
plt.show()


# ------------------- 5. Outlier Removal using IQR -------------------
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df_filtered


df = remove_outliers(df, 'fare_amount')
df = remove_outliers(df, 'distance_km')

# Filter remaining invalid distances
df = df[(df['distance_km'] > 0) & (df['distance_km'] < 100)]

# Drop unused columns
df.drop(['key', 'pickup_datetime', 'fare_z', 'dist_z'], axis=1, inplace=True)

# ------------------- 6. Correlation Heatmap -------------------
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# ------------------- 7. Model Training and Evaluation -------------------
X = df[['distance_km', 'passenger_count', 'hour', 'day']]
y = df['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {'R2 Score': round(r2, 4), 'RMSE': round(rmse, 4)}

    # Predicted vs Actual Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([0, 200], [0, 200], 'r--')
    plt.xlabel("Actual Fare Amount")
    plt.ylabel("Predicted Fare Amount")
    plt.title(f"{name}: Actual vs Predicted")
    plt.xlim(0, 200)
    plt.ylim(0, 200)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------- 8. Model Comparison -------------------
results_df = pd.DataFrame(results).T
print("\nModel Evaluation Comparison:")
print(results_df)

results_df.plot(kind='bar', figsize=(8, 5))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
plt.show()
