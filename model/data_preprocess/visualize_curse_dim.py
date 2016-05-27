# visualize The Curse of Dimension
import numpy as np
import matplotlib.pyplot as plt

def random_point(dim):
	"""
	Generating random points in special dimension
	"""
	return np.random.random(dim)

def distance(point1, point2):
	return np.sqrt(sum((point1-point2)**2))

def random_distances(dim, num_pairs):
	return [distance(random_point(dim), random_point(dim)) \
			for i in range(num_pairs)]

def compare_distance_by_dim(dim=100, num_pairs=10000):
	dimensions = range(1,dim)
	avg_distances = []
	min_distances = []

	for d in dimensions:
		distances = np.array(random_distances(d,num_pairs))
		avg_distances.append(np.mean(distances))
		min_distances.append(np.min(distances))

	#min_avg_ratio = np.array(min_distances) / np.array(avg_distances)
	min_avg_ratio = np.true_divide(min_distances, avg_distances)
	
	plt.plot(avg_distances,'b',label='average distance')
	plt.plot(min_distances,'g',label='minimum distance')
	plt.plot(min_avg_ratio,'r',label='min/avg ratio')
	plt.xlabel('# of dimensions')
	plt.title('10,000 random_distances')
	plt.legend(loc='best')
