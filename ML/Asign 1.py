import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load the dataset
url = "Wine.csv"
df = pd.read_csv(url)

# Step 2: Explore dataset
print("Dataset Head:\n", df.head(10))
print("\nColumn Names:", df.columns.tolist())

# Step 3: Separate features and target
X = df.drop('Customer_Segment', axis=1) # assuming Customer_Segment is target
y = df['Customer_Segment']

# Step 4: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply PCA
pca = PCA(n_components=2) # reduce to 2 components for visualization
X_pca = pca.fit_transform(X_scaled)

# Step 6: Create DataFrame for visualization
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Target'] = y

# Step 7: Plot PCA result
plt.figure(figsize=(10, 7))
sns.scatterplot(x='PC1', y='PC2', hue='Target', data=pca_df, palette='Set1')
plt.title('PCA of Wine Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# Step 8: Explained variance
print("\nExplained variance ratio:", pca.explained_variance_ratio_)
