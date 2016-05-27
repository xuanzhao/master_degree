import pandas as pd
import numpy  as np
import os


def get_data():
	"""Get data from local csv or pandas repo."""
	if os.path.exists('iris.csv')
		print '-- iris.csv found locally'
		df = pd.read_csv('iris.csv', index_col=0)
	else:
		print '-- trying to download from github'
		fn = ''
		try:
			df = pd.read_csv(fn)
		except:
			exit('-- Unable to download iris.csv')

		with open('iris.csv', 'w') as f:
			print '-- writing to local iris.csv file'
			df.to_csv(f)
	
	return f

