import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 1. Import the Excel file
file_path = '/path/to/your/directory/PCA_male_female_4-16.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Filter data for IDs between 1 and 55
df_filtered = df[df['ID'].between(1, 55)]

# 2. Normalization
features = ['Activity_L', 'Activity_D', 'Sociability_L', 'Sociability_D']
scaler = MinMaxScaler()  # Scaling to the range [0, 1]
df_filtered[features] = scaler.fit_transform(df_filtered[features])

# 3. Perform PCA
x = df_filtered[features].values
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Calculate vector loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Optimize the number of clusters using the Silhouette method
k_values = range(2, 16)
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(df_pca[['PC1', 'PC2']])
    score = silhouette_score(df_pca[['PC1', 'PC2']], kmeans.labels_)
    silhouette_scores.append(score)

# Determine the optimal number of clusters
best_k = k_values[np.argmax(silhouette_scores)]
best_score = max(silhouette_scores)

# KMeans++ is used for clustering
print(f"KMeans++ is used with {best_k} clusters for the best silhouette score.")

# Color mapping for clusters
cluster_color_map = {
    1: '#FF0000',  # Red
    2: '#00FF00',  # Green
    3: '#0000FF',  # Blue
    4: '#FFFF00',  # Yellow
    5: '#FF00FF',  # Pink
    6: '#00FFFF',  # Cyan
    7: '#800080',  # Purple
    8: '#808000'   # Olive
}

# Graph A: PCA plot with feature vectors
plt.figure(figsize=(6, 6))
plt.scatter(df_pca['PC1'], df_pca['PC2'], c='black')
plt.quiver(np.zeros(len(features)), np.zeros(len(features)), loadings[:, 0], loadings[:, 1],
           angles='xy', scale_units='xy', scale=1, color=['r'])
for i, feature in enumerate(features):
    plt.text(loadings[i, 0], loadings[i, 1], feature, color='r', ha='center', va='center')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.title('PCA Plot with Feature Vectors (ID 1-55)')
plt.grid(True)
plt.savefig('/path/to/your/directory/pca_plot_A.png')
plt.show()

# Graph B: PCA plot with K-Means clustering results (Cluster numbering starts from 1)
kmeans = KMeans(n_clusters=best_k, init='k-means++', random_state=42)
df_pca['Cluster'] = kmeans.fit_predict(df_pca[['PC1', 'PC2']]) + 1  # Start cluster numbering from 1

plt.figure(figsize=(6, 6))
for cluster in sorted(df_pca['Cluster'].unique()):
    subset = df_pca[df_pca['Cluster'] == cluster]
    plt.scatter(subset['PC1'], subset['PC2'], color=cluster_color_map[cluster], label=f'Cluster {cluster}')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.title(f'PCA Plot with K-Means++ Clusters (ID 1-55)')
plt.legend()
plt.grid(True)
plt.savefig('/path/to/your/directory/pca_plot_B.png')
plt.show()

# Graph C: PCA plot with color-coded group IDs (specified color codes)
plt.figure(figsize=(6, 6))
color_map = {
    'Male4': '#0000FF',   # Blue
    'Male8': '#0F80FF',   # Light Blue
    'Male15': '#21FFFF',  # Cyan
    'Female4': '#FB02FF', # Pink
    'Female8': '#FB0280', # Magenta
    'Female16': '#FC6666' # Red
}

groups = df_filtered['Group'].unique()
for group in groups:
    subset = df_filtered[df_filtered['Group'] == group]
    plt.scatter(df_pca.loc[subset.index, 'PC1'], df_pca.loc[subset.index, 'PC2'], 
                color=color_map[group], label=group)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.title('PCA Plot with Colored IDs (ID 1-55)')
plt.legend()
plt.grid(True)
plt.savefig('/path/to/your/directory/pca_plot_C.png')
plt.show()

# Plot and save the silhouette scores (same aspect ratio as other plots)
plt.figure(figsize=(6, 6))
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis with K-Means++')
plt.grid(True)
plt.savefig('/path/to/your/directory/silhouette_analysis.png')
plt.show()

# Save silhouette scores to an Excel file
results_df = pd.DataFrame({
    'Number of Clusters (k)': k_values,
    'Silhouette Score': silhouette_scores
})

# Add the best number of clusters and score
best_df = pd.DataFrame({
    'Number of Clusters (k)': ['Best k', 'Best Score'],
    'Silhouette Score': [best_k, best_score]
})

results_df = pd.concat([results_df, best_df], ignore_index=True)

# Specify the path for the output Excel file
output_file_path = '/path/to/your/directory/silhouette_scores.xlsx'
results_df.to_excel(output_file_path, index=False)
print(f'Silhouette scores have been saved to {output_file_path}')
