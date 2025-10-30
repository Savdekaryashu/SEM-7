# Agglomerative Hierarchical Clustering Example
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# We’ll use only two features for easy visualization
X = df[['sepal length (cm)', 'petal length (cm)']]

# -------------------------------
# Step 2: Standardize Data
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Step 3: Plot Dendrogram
# -------------------------------
plt.figure(figsize=(8, 5))
plt.title("Dendrogram for Iris Dataset")
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

# -------------------------------
# Step 4: Apply Agglomerative Clustering
# -------------------------------
# Let's assume we want 3 clusters (Iris has 3 species)
model = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')

labels = model.fit_predict(X_scaled)

# Add cluster labels to DataFrame
df['Cluster'] = labels

# -------------------------------
# Step 5: Visualize the Clusters
# -------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='rainbow')
plt.title("Agglomerative Clustering (Iris Dataset)")
plt.xlabel("Sepal Length (standardized)")
plt.ylabel("Petal Length (standardized)")
plt.show()

# -------------------------------
# Step 6: Display Cluster Info
# -------------------------------
print(df.head())
