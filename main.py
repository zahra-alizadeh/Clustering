import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sklearn.metrics as sm
import betacv

data = pd.read_csv("Mall_Customers.csv")
data.rename(columns={'Annual Income (k$)': 'Annual_Income', 'Spending Score (1-100)': 'Spending_Score'}, inplace=True)
data['Gender'] = data['Gender'].replace(['Male', 'Female'], [0, 1])
data.drop(["CustomerID"], axis=1, inplace=True)

print("prepare data for training")

# ----------------------------------------------------

print("min age in dataset : ", data.Age.min())
print("max age in dataset : ", data.Age.max())

age18_25 = data.Age[(data.Age <= 25) & (data.Age >= 18)]
age26_35 = data.Age[(data.Age <= 35) & (data.Age >= 26)]
age36_45 = data.Age[(data.Age <= 45) & (data.Age >= 36)]
age46_55 = data.Age[(data.Age <= 55) & (data.Age >= 46)]
age55above = data.Age[data.Age >= 56]

x = ["18-25", "26-35", "36-45", "46-55", "55-70"]
y = [len(age18_25.values), len(age26_35.values), len(age36_45.values), len(age46_55.values), len(age55above.values)]
sns.barplot(x=x, y=y, palette="rocket")
plt.title("Number of Customer and Ages")
plt.xlabel("Age")
plt.ylabel("Number of Customer")
plt.show()

# ----------------------------------------------------

print("min annual income in dataset : ", data.Annual_Income.min())
print("max annual income in dataset : ", data.Annual_Income.max())

fig = plt.figure()
plt.hist(data.Annual_Income, histtype='bar', rwidth=0.8)
fig.suptitle('Annual Income distribution in Dataset', fontsize=15)
plt.xlabel('Annual Income')
plt.ylabel('count')
plt.show()

# ----------------------------------------------------

distortions = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    distortions.append(kmeans.inertia_)

print("The Elbow Method showing the optimal k")
plt.plot(range(1, 11), distortions, 'bx-')
plt.xticks(range(1, 11))
plt.xlabel('k')
plt.ylabel('SSE')
# plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.grid()
plt.show()

# ----------------------------------------------------

print("use Age, Annual Income and Spending Score for clustering customers.")
sns.set_style("white")
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.Age, data.Annual_Income, data.Spending_Score, c='blue', s=60)
ax.view_init(30, 185)
plt.title("plot without clustering")
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()

# ----------------------------------------------------

kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(data)
data["label"] = kmeans.labels_
print("The lowest SSE value : ", kmeans.inertia_)
# Calculate Silhoutte Score
score = silhouette_score(data, data["label"], metric='euclidean')
print('Silhouetter Score: %.3f' % score)

# ----------------------------------------------------

print("By elbow method the optimal number of clusters is 5.")
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.Age[data.label == 0], data.Annual_Income[data.label == 0], data.Spending_Score[data.label == 0],
           c='blue', s=60)
ax.scatter(data.Age[data.label == 1], data.Annual_Income[data.label == 1], data.Spending_Score[data.label == 1],
           c='red', s=60)
ax.scatter(data.Age[data.label == 2], data.Annual_Income[data.label == 2], data.Spending_Score[data.label == 2],
           c='green', s=60)
ax.scatter(data.Age[data.label == 3], data.Annual_Income[data.label == 3], data.Spending_Score[data.label == 3],
           c='orange', s=60)
ax.scatter(data.Age[data.label == 4], data.Annual_Income[data.label == 4], data.Spending_Score[data.label == 4],
           c='purple', s=60)
ax.view_init(30, 185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()

# ----------------------------------------------------

# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
# sns.scatterplot(data=data, x='Age', y='Spending_Score', ax=ax1, hue='label', palette='Set1')
# sns.scatterplot(data=data, x='Annual_Income', y='Spending_Score', ax=ax2, hue='label', palette='Set1')

# ----------------------------------------------------

x = data.values

plt.scatter(x[clusters == 0, 2], x[clusters == 0, 3], s=100, c='black', label='cluster1')
plt.scatter(x[clusters == 1, 2], x[clusters == 1, 3], s=100, c='red', label='cluster2')
plt.scatter(x[clusters == 2, 2], x[clusters == 2, 3], s=100, c='blue', label='cluster3')
plt.scatter(x[clusters == 3, 2], x[clusters == 3, 3], s=100, c='green', label='cluster4')
plt.scatter(x[clusters == 4, 2], x[clusters == 4, 3], s=100, c='purple', label='cluster5')

plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3], s=100, c='yellow', label='Centroids')
plt.show()

BETACV = betacv.betacv(data, clusters)
print("betacv : ", BETACV)
