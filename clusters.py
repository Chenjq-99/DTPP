import json
import re
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 读取文件
file_path = "./optimized_parameters_2.json"

with open(file_path, "r") as file:
    data = file.read()

# 提取五维数据
pattern = r'\((\[[\-0-9.,\s]+\])'
matches = re.findall(pattern, data)
idm_params = [json.loads(match) for match in matches]

# 转换为矩阵格式
import numpy as np
X = np.array(idm_params)
print(len(X))
# 聚类
num_clusters = 8  # 设置目标簇数
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# 获取聚类标签
labels = kmeans.labels_

print(kmeans.cluster_centers_)

# 降维可视化
from sklearn.decomposition import PCA
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X)

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for the data points with cluster labels
scatter = ax.scatter(
    X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], 
    c=labels, cmap='viridis', s=10, alpha=0.8
)

# Add cluster centers to the plot
centers_pca_3d = pca_3d.transform(kmeans.cluster_centers_)
ax.scatter(
    centers_pca_3d[:, 0], centers_pca_3d[:, 1], centers_pca_3d[:, 2], 
    c='red', marker='X', s=30, label='Cluster Centers'
)

# Label the axes and add a title
ax.set_title("3D Cluster Visualization")
plt.legend()
plt.colorbar(scatter, label='Cluster')
plt.savefig("clusters_2.png", dpi=600)
plt.show()