import numpy as np

def split_data(data, prob):
	"""
	split data into factions [prob, 1-prob]
	return the dataset for subset which is index belong 0 or 1
	"""
	results = ([],[])
	for row in data:
		results[0 if np.random.random() < prob else 1].append(row)
	return results


def train_test_split(x, y, test_pct):
	data = zip(x, y)
	train, test = split_data(data, 1-test_pct)
	x_train, y_train = zip(*train)
	x_test, y_test   = zip(*test)
	return x_train, y_train, x_test, y_test


def accuracy(tp, fp, fn, tn):
	correct = tp + tn
	total = tp + fp + fn + tn
	return correct / total


def precision(tp, fp, fn, tn):
	return tp / (tp + fp)


def recall(tp, fp, fn, tn):
	return tp / (tp + fn)

def f1_score(tp, fp, fn, tn):
	"""
	This is the harmonic mean of precision and recall and necessarily lies between them.
	"""
	p = precision(tp, fp, fn, tn)
	r = recall(tp, fp, fn, tn)
	return 2*p*r / (p+r)

def bootstrap_sample(data):
	"""
	the type of data is ndarray.
	"""
	return np.random.choice(data.ravel(), size=data.shape, replace=True)

def bootstrap_statistic(data, stats_func, num_samples):
	"""
	evaluates stats_func on each of samples space which is from bootstrp_sample.
	"""
	return [stats_func(bootstrp_sample(data)) for _ in range(num_samples)]


def estimate_sample_beta(sample):
	"""
	sample is a list of pairs (x_i, y_i)

	bootstrap_betas = bootstrap_statistic(zip(x_i, y_i),
									  estimate_sample_beta,
									  100)
	bootstrap_standard_errors = [ standard_deviation([beta[i] for beta in bootstrap_betas])
								for i in range(4) ]
	"""
	x_sample, y_sample = zip(*sample)  #unzipping trick
	return estimate_beta(x_sample, y_sample)

def p_value(beta_hat_j, sigma_hat_j):
	if beta_hat_j > 0:
		return 2 * (1 - normal_cdf(beta_hat_j, sigma_hat_j))
	else:
		return 2 * normal_cdf(beta_hat_j, sigma_hat_j)


def f1_score(y_test, y_test):
	true_positives = false_negatives = false_positives = true_negatives = 0
	for x_i, y_i in zip(x_test, y_test):
		result = predict(x_i, y_i)

		if y_i == 1 and result >= 0.5
			true_positives += 1
		elif y_i == 1:
			false_negatives += 1
		elif result >= 0.5:
			false_positives += 1
		else:
			true_negatives += 1

	precision = true_positives / (true_positives + false_positives)
	recall = true_positives / (true_positives + false_negatives)
	f1_score = 2 * (precision * recall) / (precision + recall)
