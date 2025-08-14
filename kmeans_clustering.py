import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("dataset.csv")
print("📂 Original Dataset:")
print(df.head())


numeric_df = df.select_dtypes(include=[np.number])


scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)


inertia = []
K_range = range(1, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)


plt.figure(figsize=(6, 4))
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()


optimal_k = 3 
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_data)

print("\n✅ Clustered Data:")
print(df.head())


plt.figure(figsize=(6, 4))
sns.scatterplot(
    x=numeric_df.iloc[:, 0],
    y=numeric_df.iloc[:, 1],
    hue=df['Cluster'],
    palette='viridis',
    s=100
)


centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    color='red',
    marker='X',
    s=200,
    label='Centroids'
)

plt.xlabel(numeric_df.columns[0])
plt.ylabel(numeric_df.columns[1])
plt.title('K-Means Clustering')
plt.legend()
plt.show()
