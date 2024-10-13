import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Create a directory to save the output files
output_dir = '/path/to/your/output_directory'
os.makedirs(output_dir, exist_ok=True)

# 1. Import the Excel file
file_path = '/path/to/your/directory/Elbow_Silhouette_male_female_4-16xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Filter data for IDs between 1 and 55
df_filtered = df[df['ID'].between(1, 55)]

# 2. Normalize the features
features = ['Activity_L', 'Activity_D', 'Sociability_L', 'Sociability_D']
scaler = MinMaxScaler()  # Min-Max Scaling to [0, 1] range
df_filtered[features] = scaler.fit_transform(df_filtered[features])

# 3. Perform PCA
x = df_filtered[features].values
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Calculate loadings (contribution vectors)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Determine the optimal number of clusters using the Elbow Method
k_values = range(1, 11)  # Set k range from 1 to 10
sse = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(df_pca[['PC1', 'PC2']])
    sse.append(kmeans.inertia_)

# Plot the results of the Elbow Method
plt.figure(figsize=(8, 8))  # Set equal aspect ratio
plt.plot(k_values, sse, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'elbow_method.png'))
plt.show()

# Evaluate the number of clusters using Silhouette analysis
k_values = range(2, 7)  # Set k range from 2 to 6
color_map = ['red', 'green', 'blue', '#d4a017', 'pink', 'cyan']  # Color settings

for k in k_values:
    # Silhouette plot
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))  # Set equal aspect ratio

    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    cluster_labels = kmeans.fit_predict(df_pca[['PC1', 'PC2']])
    
    silhouette_avg = silhouette_score(df_pca[['PC1', 'PC2']], cluster_labels)
    sample_silhouette_values = silhouette_samples(df_pca[['PC1', 'PC2']], cluster_labels)

    y_lower = 10
    for i in range(k):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = color_map[i % len(color_map)]
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i + 1))  # Start labels from 1
        y_lower = y_upper + 10

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    plt.savefig(os.path.join(output_dir, f'silhouette_plot_k{k}.png'))
    plt.show()

    # Plot the clustering results
    fig, ax2 = plt.subplots(1, 1, figsize=(8, 8))  # Set equal aspect ratio
    
    colors = [color_map[label % len(color_map)] for label in cluster_labels]
    ax2.scatter(df_pca['PC1'], df_pca['PC2'], marker=".", s=100, lw=0, alpha=0.7, c=colors, edgecolor="k")

    centers = kmeans.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker="o", c="white", alpha=1, s=200, edgecolor="k")

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % (i + 1), alpha=1, s=50, edgecolor="k")  # Start labels from 1

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.savefig(os.path.join(output_dir, f'cluster_visualization_k{k}.png'))
    plt.show()

# Graph A: PCA plot with feature vectors
plt.figure(figsize=(8, 8))
plt.scatter(df_pca['PC1'], df_pca['PC2'], c='black', marker='o', s=60)

# Draw contribution vectors
for i, feature in enumerate(features):
    plt.quiver(0, 0, loadings[i, 0], loadings[i, 1], angles='xy', scale_units='xy', scale=1, color='r', width=0.005)
    plt.text(loadings[i, 0], loadings[i, 1], feature, color='r', ha='center', va='center')

# Set axis labels and title
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.title('PCA Plot with Feature Vectors')

# Set equal aspect ratio
plt.gca().set_aspect('equal', adjustable='box')

# Configure grid
plt.grid(True, linestyle='-', color='gray', alpha=0.5)

# Save the graph
plot_file_path_a = os.path.join(output_dir, 'pca_plot_A.png')
plt.savefig(plot_file_path_a)
plt.show()

# Graph B: PCA plot with color-coded IDs
plt.figure(figsize=(8, 8))

# Color mapping for each group
color_map_id = {
    'Male4': '#0000FF',
    'Male8': '#0F80FF',
    'Male15': '#21FFFF',
    'Female4': '#FB02FF',
    'Female8': '#FB0280',
    'Female16': '#FC6666'
}

for group in df_filtered['Group'].unique():
    subset = df_filtered[df_filtered['Group'] == group]
    plt.scatter(df_pca.loc[subset.index, 'PC1'], df_pca.loc[subset.index, 'PC2'], color=color_map_id[group], marker='o', label=group, s=60)

# Set axis labels and title
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.title('PCA Plot with Colored IDs')

# Set equal aspect ratio
plt.gca().set_aspect('equal', adjustable='box')

# Configure grid
plt.grid(True, linestyle='-', color='gray', alpha=0.5)

# Add legend
plt.legend()

# Save the graph
plot_file_path_b = os.path.join(output_dir, 'pca_plot_B.png')
plt.savefig(plot_file_path_b)
plt.show()

print(f'All graphs have been saved to {output_dir}')
