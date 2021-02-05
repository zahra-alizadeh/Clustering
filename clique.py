import pandas as pd
from pyclustering.cluster.clique import clique, clique_visualizer
from pyclustering.cluster import cluster_visualizer_multidim
import numpy as np
import betacv

data = pd.read_csv("Mall_Customers.csv")
data.rename(columns={'Annual Income (k$)': 'Annual_Income', 'Spending Score (1-100)': 'Spending_Score'}, inplace=True)
data['Gender'] = data['Gender'].replace(['Male', 'Female'], [0, 1])
data.drop(["CustomerID"], axis=1, inplace=True)
data_values = data.values

# Define the number of grid cells in each dimension
intervals = 5
# Density threshold
threshold = 0
clique_instance = clique(data_values, intervals, threshold)

clique_instance.process()
clique_cluster = clique_instance.get_clusters()

noise = clique_instance.get_noise()
cells = clique_instance.get_cells()

print("Amount of clusters:", len(clique_cluster))
for cluster in clique_cluster:
    print(cluster)

labelList = [0] * 200
j = 1
for cluster in clique_cluster:
    for x in cluster:
        labelList[x] = j
    j = j + 1

labels = np.array(labelList)

clique_visualizer.show_grid(cells, data_values)
visualizer = cluster_visualizer_multidim()
visualizer.append_clusters(clique_cluster, data_values)

BETACV = betacv.betacv(data, labels)
print(BETACV)
