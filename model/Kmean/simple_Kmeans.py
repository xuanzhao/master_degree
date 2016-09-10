from sklearn.cluster import KMeans,MeanShift
import numpy as np
from sklearn import preprocessing

df = pd.read_excel('Titanic.xls')
original_df = pd.DataFrame.copy(df)
df.drop(['name','body'], 1, inplace=True)
df = df.convert_objects(convert_numeric=True)
df.dtypes
df.fillna(0, inplace=True)

def handle_non_numerical_data(df):
	columns = df.columns.values

	for column in columns:
		text_digit_vals = {}
		def convert_to_int(val):
			return text_digit_vals[val]

		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			column_contents = df[column].values.tolist()
			unique_elements = set(column_contents)
			x = 0
			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique] = x
					x += 1

			df[column] = list(map(convert_to_int, df[column]))

	return df

df = handle_non_numerical_data(df)

X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)
labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan
for i in range(len(X)):
	original_df['cluster_group'].iloc[i] = labels[i]

n_clusters_  = len(np.unique(labels))
survival_rates = {}

for i in range(n_clusters_):
	temp_df = original_df[ (original_df['cluster_group']==float(i))]
	survival_cluster = temp_df[ (temp_df['survived']==1) ]
	survived_rate = len(survival_cluster)/len(temp_df)
	survival_rates[i] = survival_rates

print(survival_rates)

X = np.array([[1,2],
			  [1.5, 1.8],
			  [5,8],
			  [8,8],
			  [1,0.6],
			  [9,11]])
colors = ['g','r','c','b','k']

class K_means:
	def __init__(self, k=2, tol=0.001, max_iter=300):
		self.k = k
		self.tol = tol
		self.max_iter = max_iter

	def fit(self, data):

		self.centroids = {}

		for i in range(self.k):
			self.centroids[i] = data[i]

		for i in range(self.max_iter):
			self.classifications = {}

			for i in range(self.k):
				self.classifications[i] = []

			for featureset in X:
				distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset)

			prev_centroids = dict(self.centroids)

			for classification in self.classifications:
				pass
				self.centroids[classification] = np.average(self.classifications[classification], axis=0)

			optimized = True

			for c in self.centroids:
				original_centroid = prev_centroids[c]
				current_centroid = self.centroids[c]

				if  np.sum((current_centroid-original_centroid) / original_centroid*100.00) > self.tol:
					optimized= False
			if optimized:
				break

	def predict(self, data):
		distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
		classification = distances.index(min(distances))

		return classification

clf = K_means()

clf.fit(X)

for centroid in clf.centroids:
	plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
				marker='o', color='k', s=100, linewidths=5)


for classification in clf.classifications:
	color = colors[classification]
	for featureset in clf.classifications[classification]:
		plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150, linewidths=5)


unknowns = np.array([[1,3],
					 [8,9],
					 [0,3],
					 [5,4],
					 [6,4]])

for unknown in unknowns:
	classification = clf.predict(unknown)
	plt.scatter(unknown[0], unknown[1], marker='*', color=colors[classification])

plt.show()















