import random
import pandas
import numpy as np

class randomForest:

	def __init__(self, n_tree, data, data_desc):
		#Number of trees to be generated
		self.n_tree = n_tree
		self.data = data
		self.data_desc = data_desc

	def bootstrap(self, n_iter, data):
		#list of bootstraps ([training_data, test_data])
		bootstraps = []

		#All possible index in the dataset
		all_indexs = range(data.shape[0])

		for i in range(n_iter):
			indexs_used = []

			#Iterating over the number of elements of the dataset
			for x in range(data.shape[0]):
				#gererating random numbers between 0 and data.shape[0]-1
				indexs_used.append(random.randint(0,data.shape[0]-1))

			#removing repeated elements
			indexs_used = sorted(list(dict.fromkeys(indexs_used)))

			#getting elements not used
			indexs_not_used = list(np.setdiff1d(all_indexs, indexs_used))

			print(indexs_used,indexs_not_used)
			
			#Getting actual training and test data
			training_data = data.iloc[indexs_used]
			test_data = data.iloc[indexs_not_used]
			bootstraps.append([training_data, test_data])

		return bootstraps
