import random
import pandas
import numpy as np
from decisionTree import decisionTree as dt

class randomForest:

	def __init__(self, n_tree, data, data_desc):
		#Number of trees to be generated
		self.n_tree = n_tree
		self.data = data
		self.data_desc = data_desc
		self.decisionTrees = None

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

			#sorting
			#indexs_used = sorted(indexs_used)

			#getting elements not used
			indexs_not_used = list(np.setdiff1d(all_indexs, indexs_used))

			if len(indexs_not_used) == 0:
				return self.bootstrap(n_iter,data)
			
			#Getting actual training and test data
			training_data = data.iloc[indexs_used]
			test_data = data.iloc[indexs_not_used]
			bootstraps.append([training_data, test_data])

		return bootstraps

	def testDecisionTree(self, decisionTree, test_data):
		correct_predictions = 0

		for index, row in test_data.iterrows():
			#print(row)
			predicted_class = decisionTree.predict_instance(row)

			#print(predicted_class, row[-1])

			if predicted_class == row[-1]:
				correct_predictions += 1

		return correct_predictions/test_data.shape[0]

	def fit(self):
		########## Random Forest Creation ##########

		#List to save all the decision trees
		decisionTrees = []

		#creating bootstraps
		bootstraps = self.bootstrap(self.n_tree, self.data)

		for bootstrap in bootstraps:
			#print('New bootstrap ################################################')
			#Creating decisionTree object from training data.
			det = dt(bootstrap[0], self.data_desc)

			#building actual decisionTree
			root_node = det.generateDecisionTree()

			#Testing accuracy of the decisionTree against test data.
			#accuracy = self.testDecisionTree(det, bootstrap[1])

			#If accuracy is greater than 0.5 the decisionTree will be added
			#to the random forest, if not decisionTree is discarded.
			#if accuracy >= 0.5:
			#	decisionTrees.append(det)
			#else:
			#	print('Discarding tree with less than 0.5 of accuracy')

			##### Using all the decision Trees generated for the random forest.
			decisionTrees.append(det)

		self.decisionTrees = decisionTrees

	def predict(self, instances):
		########## Majority Voting ##########
		predictions = []

		#Getting predictions from the random forest
		for index,row in instances.iterrows():
			predictions_trees = []
			for decisionTree in self.decisionTrees:
				prediction_tree = decisionTree.predict_instance(row)
				predictions_trees.append(prediction_tree)

			#counting predictions made from the random forest into a dictionary
			count_predictions_trees = dict((x,predictions_trees.count(x)) for x in set(predictions_trees))

			#Ordering the dictionary in reverse order.
			count_predictions_trees = dict(sorted(count_predictions_trees.items(), key=lambda x: x[1], reverse=True))

			#After counting the votes
			final_prediction_instance = list(count_predictions_trees.keys())[0]

			#Adding prediction to the list of predictions
			predictions.append(final_prediction_instance)

		return predictions