import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title='Select an Excel file',
    filetypes=[('Excel files', '*.xlsx *.xls')]
)

if not file_path:
    print("No file selected. Exiting...")
    exit()

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_dir = os.path.join(desktop_path, 'PCA_Output')
os.makedirs(output_dir, exist_ok=True)

df = pd.read_excel(file_path, sheet_name='Sheet1')
df_filtered = df[df['ID'].between(1, 55)].copy()

features = ['Activity_L', 'Activity_D', 'Sociability_L', 'Sociability_D']
scaler = MinMaxScaler()
df_filtered.loc[:, features] = scaler.fit_transform(df_filtered[features])

x = df_filtered[features].values
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

def draw_ellipse(x, y, ax, edgecolor='black'):
    if len(x) > 1:
        cov = np.cov(x, y)
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        ell = mpatches.Ellipse(xy=(np.mean(x), np.mean(y)),
                               width=lambda_[0]*4, height=lambda_[1]*4,
                               angle=np.rad2deg(np.arccos(v[0, 0])),
                               edgecolor=edgecolor, fc='none', lw=2, alpha=0.6)
        ax.add_patch(ell)

color_map = OrderedDict([
    ('Male4', '#0000FF'),   
    ('Male8', '#0F80FF'),   
    ('Male15', '#21FFFF'),  
    ('Female4', '#FB02FF'), 
    ('Female8', '#FB0280'), 
    ('Female16', '#FC6666') 
])

marker_map = {}
for group in color_map.keys():
    if 'female' in group.lower():
        marker_map[group] = 's'  
    else:
        marker_map[group] = 'o'  

# Graph A (1): PCA plot without ellipses
plt.figure(figsize=(6, 6))
for group, color in color_map.items():
    subset = df_filtered[df_filtered['Group'] == group]
    marker = marker_map[group]
    plt.scatter(df_pca.loc[subset.index, 'PC1'], df_pca.loc[subset.index, 'PC2'],
                color=color, label=group, marker=marker)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)')
plt.title('PCA Plot with Colored IDs')
plt.grid(True)
plt.axis('equal')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig(os.path.join(output_dir, 'pca_plot_A_markers.png'), bbox_inches='tight')
plt.show()

# Graph A (2): PCA plot with ellipses
plt.figure(figsize=(6, 6))
ax = plt.gca()
for group, color in color_map.items():
    subset = df_filtered[df_filtered['Group'] == group]
    marker = marker_map[group]
    x = df_pca.loc[subset.index, 'PC1']
    y = df_pca.loc[subset.index, 'PC2']
    plt.scatter(x, y, color=color, label=group, marker=marker)
    draw_ellipse(x, y, ax, edgecolor=color)
    plt.text(np.mean(x), np.mean(y), group, fontsize=9, ha='center', va='center')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)')
plt.title('PCA Plot with Colored IDs')
plt.grid(True)
plt.axis('equal')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig(os.path.join(output_dir, 'pca_plot_A_markers_ellipses.png'), bbox_inches='tight')
plt.show()

# Graph B: PCA plot with K-means clustering
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
df_pca['Cluster'] = kmeans.fit_predict(df_pca[['PC1', 'PC2']])
df_pca['Group'] = df_filtered['Group'].values  

plt.figure(figsize=(6, 6))
cluster_colors = ['red', 'green', 'blue', '#d4a017']
custom_handles = []

for cluster in range(4):
    cluster_data = df_pca[df_pca['Cluster'] == cluster]
    cluster_color = cluster_colors[cluster % len(cluster_colors)]

    for group in cluster_data['Group'].unique():
        subset = cluster_data[cluster_data['Group'] == group]
        marker = marker_map.get(group, 'o')
        plt.scatter(subset['PC1'], subset['PC2'],
                    color=cluster_color,
                    marker=marker)
    circle = Line2D([], [], color=cluster_color, marker='o', linestyle='None', markersize=6)
    square = Line2D([], [], color=cluster_color, marker='s', linestyle='None', markersize=6)
    custom_handles.append((circle, square, f'Cluster {cluster + 1}'))
handles = [(c, s) for c, s, _ in custom_handles]
labels = [lbl for _, _, lbl in custom_handles]
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)')
plt.title(f'PCA Plot with K-Means++ Clusters')
plt.grid(True)
plt.axis('equal')

plt.legend(handles=handles, labels=labels, handler_map={tuple: HandlerTuple(ndivide=None)},
           bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig(os.path.join(output_dir, 'pca_plot_B.png'), bbox_inches='tight')
plt.show()

# Graph C: PCA with feature vectors
plt.figure(figsize=(6, 6))
for group, color in color_map.items():
    subset = df_filtered[df_filtered['Group'] == group]
    marker = marker_map[group]
    plt.scatter(df_pca.loc[subset.index, 'PC1'], df_pca.loc[subset.index, 'PC2'],
                color='black', label=group, marker=marker)

plt.quiver(np.zeros(len(features)), np.zeros(len(features)), loadings[:, 0], loadings[:, 1],
           angles='xy', scale_units='xy', scale=1, color='r')
for i, feature in enumerate(features):
    plt.text(loadings[i, 0], loadings[i, 1], feature, color='red', ha='center', va='center')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)')
plt.title('PCA Plot with Feature Vectors')
plt.grid(True)
plt.axis('equal')
plt.savefig(os.path.join(output_dir, 'pca_plot_C.png'), bbox_inches='tight')
plt.show()

print('All graphs have been saved.')
